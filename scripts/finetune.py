from gym.envs import __init__  # noqa: F401
from gym.utils import get_args, task_registry
from gym.utils.helpers import class_to_dict

from learning.algorithms import *  # noqa: F403
from learning.runners.finetune_runner import FineTuneRunner
from gym.envs.mini_cheetah.minimalist_cheetah import MinimalistCheetah

from gym import LEGGED_GYM_ROOT_DIR

import os
import torch
import scipy.io
import numpy as np
import pandas as pd
from tensordict import TensorDict

ROOT_DIR = f"{LEGGED_GYM_ROOT_DIR}/logs/mini_cheetah_ref/"
LOAD_RUN = "Jul13_01-49-59_PPO32_S16"
LOG_FILE = "PPO_701_a001_expl08.mat"
REWARDS_FILE = "rewards_test.csv"

TRAJ_LENGTH = 100  # split data into chunks of this length
EXPLORATION_SCALE = 0.8  # scale std in SmoothActor
ITERATION = 701  # load this iteration

# Data struct fields from Robot-Software logs
DATA_LIST = [
    "header",
    "base_height",  # 1 shape: (1, data_length)
    # the following are all shape: (data_length, n)
    "base_lin_vel",  # 2
    "base_ang_vel",  # 3
    "projected_gravity",  # 4
    "commands",  # 5
    "dof_pos_obs",  # 6
    "dof_vel",  # 7
    "phase_obs",  # 8
    "grf",  # 9
    "dof_pos_target",  # 10
    "exploration_noise",  # 11
    "footer",
]

# Finetune for these rewards instead of the ones in the cfg
REWARD_WEIGHTS = {
    "min_base_height": 1.5,
    # "tracking_lin_vel": 4.0,
    # "tracking_ang_vel": 2.0,
    "orientation": 1.0,
    "swing_grf": 1.0,
    "stance_grf": 1.0,
    "stand_still": 1.0,
    "action_rate": 0.001,
    "action_rate2": 0.001,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_obs(obs_list, data_struct, batches):
    obs_all = torch.empty(0).to(DEVICE)
    for obs_name in obs_list:
        data_idx = DATA_LIST.index(obs_name)
        obs = torch.tensor(data_struct[data_idx]).to(DEVICE)
        if obs.shape[0] == 1:
            obs = obs.T  # base_height has shape (1, data_length)
        obs = obs[: TRAJ_LENGTH * batches, :]
        obs = obs.reshape((TRAJ_LENGTH, batches, -1))
        obs_all = torch.cat((obs_all, obs), dim=-1)

    return obs_all.float()


def get_rewards(data_struct, num_trajs):
    mini_cheetah = MinimalistCheetah(device=DEVICE)
    rewards_dict = {name: [] for name in REWARD_WEIGHTS.keys()}  # for plotting
    rewards_all = torch.empty(0).to(DEVICE)

    for i in range(TRAJ_LENGTH * num_trajs):
        mini_cheetah.set_states(
            base_height=data_struct[1][:, i],  # shape (1, data_length)
            base_lin_vel=data_struct[2][i],
            base_ang_vel=data_struct[3][i],
            proj_gravity=data_struct[4][i],
            commands=data_struct[5][i],
            dof_pos_obs=data_struct[6][i],
            dof_vel=data_struct[7][i],
            phase_obs=data_struct[8][i],
            grf=data_struct[9][i],
            dof_pos_target=data_struct[10][i],
        )
        total_rewards = 0
        for name, weight in REWARD_WEIGHTS.items():
            reward = weight * eval(f"mini_cheetah._reward_{name}()")
            rewards_dict[name].append(reward.item())
            total_rewards += reward
        rewards_all = torch.cat((rewards_all, total_rewards), dim=0)
        # Post process mini cheetah
        mini_cheetah.post_process()

    rewards_dict["total"] = rewards_all.tolist()

    # Save rewards in dataframe
    rewards_path = os.path.join(ROOT_DIR, LOAD_RUN, REWARDS_FILE)
    if not os.path.exists(rewards_path):
        rewards_df = pd.DataFrame(columns=["iteration", "type", "mean", "std"])
    else:
        rewards_df = pd.read_csv(rewards_path)
    for name, rewards in rewards_dict.items():
        rewards = np.array(rewards)
        mean = rewards.mean()
        std = rewards.std()
        rewards_df = rewards_df._append(
            {
                "iteration": str(ITERATION) + "_expl",
                "type": name,
                "mean": mean,
                "std": std,
            },
            ignore_index=True,
        )
    rewards_df.to_csv(rewards_path, index=False)
    print(rewards_df)

    # Return total rewards
    rewards_all = rewards_all.reshape((TRAJ_LENGTH, num_trajs))
    return rewards_all.float()


def get_data_dict(train_cfg, name="SMOOTH_RL_CONTROLLER"):
    path = os.path.join(ROOT_DIR, LOAD_RUN, LOG_FILE)
    data = scipy.io.loadmat(path)
    data_struct = data[name][0][0]

    data_length = data_struct[1].shape[1]  # base_height: shape (1, data_length)
    num_trajs = data_length // TRAJ_LENGTH
    data_dict = TensorDict(
        {},
        device=DEVICE,
        batch_size=(TRAJ_LENGTH - 1, num_trajs),  # -1 for next_obs
    )
    actor_obs = get_obs(train_cfg["actor"]["obs"], data_struct, num_trajs)
    critic_obs = get_obs(train_cfg["critic"]["obs"], data_struct, num_trajs)

    data_dict["actor_obs"] = actor_obs[:-1]
    data_dict["next_actor_obs"] = actor_obs[1:]
    data_dict["critic_obs"] = critic_obs[:-1]
    data_dict["next_critic_obs"] = critic_obs[1:]

    actions_idx = DATA_LIST.index("dof_pos_target")
    actions = torch.tensor(data_struct[actions_idx]).to(DEVICE)
    actions = actions[: TRAJ_LENGTH * num_trajs, :]
    actions = actions.reshape((TRAJ_LENGTH, num_trajs, -1))
    data_dict["actions"] = actions[:-1]

    rewards = get_rewards(data_struct, num_trajs)
    data_dict["rewards"] = rewards[:-1]

    # Assume no termination
    data_dict["timed_out"] = torch.zeros(
        TRAJ_LENGTH - 1, num_trajs, device=DEVICE, dtype=bool
    )
    data_dict["terminal"] = torch.zeros(
        TRAJ_LENGTH - 1, num_trajs, device=DEVICE, dtype=bool
    )
    data_dict["dones"] = torch.zeros(
        TRAJ_LENGTH - 1, num_trajs, device=DEVICE, dtype=bool
    )

    return data_dict


def setup():
    args = get_args()

    _, train_cfg = task_registry.create_cfgs(args)
    train_cfg = class_to_dict(train_cfg)
    data_dict = get_data_dict(train_cfg)

    runner = FineTuneRunner(
        train_cfg, data_dict, exploration_scale=EXPLORATION_SCALE, device=DEVICE
    )
    runner._set_up_alg()

    return runner


def finetune(runner):
    load_path = os.path.join(ROOT_DIR, LOAD_RUN, "model_" + str(ITERATION) + ".pt")
    print("Loading model from: ", load_path)
    runner.load(load_path)

    # Perform a single update
    runner.learn()

    # Compare old and new actions
    scaling = torch.tensor([0.2, 0.3, 0.3], device=DEVICE).repeat(4)
    actions_old = runner.data_dict["actions"]
    actions_new = scaling * runner.alg.actor.act_inference(
        runner.data_dict["actor_obs"]
    )
    diff = actions_new - actions_old
    print("Mean action diff per actuator: ", diff.mean(dim=(0, 1)))

    save_path = os.path.join(ROOT_DIR, LOAD_RUN, "model_" + str(ITERATION + 1) + ".pt")
    runner.save(save_path)

    export_path = os.path.join(ROOT_DIR, LOAD_RUN, "exported_" + str(ITERATION + 1))
    runner.export(export_path)


if __name__ == "__main__":
    runner = setup()
    finetune(runner)
