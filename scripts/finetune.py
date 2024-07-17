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
from tensordict import TensorDict

ROOT_DIR = f"{LEGGED_GYM_ROOT_DIR}/logs/mini_cheetah_ref/"

USE_SIMULATOR = False

# Scales
EXPLORATION_SCALE = 0.8  # used during data collection
ACTION_SCALES = np.tile(np.array([0.2, 0.3, 0.3]), 4)
COMMAND_SCALES = np.array([3.0, 1.0, 3.0])

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

DEVICE = "cuda"


def get_obs(obs_list, data_struct, data_length):
    obs_all = torch.empty(0).to(DEVICE)
    for obs_name in obs_list:
        data_idx = DATA_LIST.index(obs_name)
        obs = torch.tensor(data_struct[data_idx]).to(DEVICE)
        obs = obs.reshape((data_length, 1, -1))  # shape (data_length, 1, n)
        obs_all = torch.cat((obs_all, obs), dim=-1)

    return obs_all.float()


def get_rewards(data_struct, reward_weights, data_length):
    minimalist_cheetah = MinimalistCheetah(device=DEVICE)
    rewards_all = torch.empty(0).to(DEVICE)

    for i in range(data_length):
        minimalist_cheetah.set_states(
            base_height=data_struct[1][:, i],  # shape (1, data_length)
            base_lin_vel=data_struct[2][i],
            base_ang_vel=data_struct[3][i],
            proj_gravity=data_struct[4][i],
            commands=data_struct[5][i] * COMMAND_SCALES,
            dof_pos_obs=data_struct[6][i],
            dof_vel=data_struct[7][i],
            phase_obs=data_struct[8][i],
            grf=data_struct[9][i],
            dof_pos_target=data_struct[10][i],
        )
        total_rewards = 0
        for name, weight in reward_weights.items():
            reward = weight * eval(f"minimalist_cheetah._reward_{name}()")
            total_rewards += reward
        rewards_all = torch.cat((rewards_all, total_rewards), dim=0)
        # Post process mini cheetah
        minimalist_cheetah.post_process()

    return rewards_all.float()


def get_data_dict(train_cfg, name="SMOOTH_RL_CONTROLLER", offpol=False):
    load_run = train_cfg["runner"]["load_run"]
    checkpoint = train_cfg["runner"]["checkpoint"]
    run_dir = os.path.join(ROOT_DIR, load_run)

    if offpol:
        log_files = [file for file in os.listdir(run_dir) if file.endswith(".mat")]
        log_files = sorted(log_files)
    else:
        # Single log file
        log_files = [str(checkpoint) + ".mat"]

    # Initialize data dict
    data = scipy.io.loadmat(os.path.join(run_dir, log_files[0]))
    data_struct = data[name][0][0]
    data_length = data_struct[1].shape[1]  # base_height: shape (1, data_length)
    batch_size = (data_length - 1, len(log_files))  # -1 for next_obs
    data_dict = TensorDict({}, device=DEVICE, batch_size=batch_size)

    # Get all data
    actor_obs_all = torch.empty(0).to(DEVICE)
    critic_obs_all = torch.empty(0).to(DEVICE)
    actions_all = torch.empty(0).to(DEVICE)
    rewards_all = torch.empty(0).to(DEVICE)
    for log in log_files:
        data = scipy.io.loadmat(os.path.join(run_dir, log))
        data_struct = data[name][0][0]

        actor_obs = get_obs(train_cfg["actor"]["obs"], data_struct, data_length)
        critic_obs = get_obs(train_cfg["critic"]["obs"], data_struct, data_length)
        actor_obs_all = torch.cat((actor_obs_all, actor_obs), dim=1)
        critic_obs_all = torch.cat((critic_obs_all, critic_obs), dim=1)

        actions_idx = DATA_LIST.index("dof_pos_target")
        actions = torch.tensor(data_struct[actions_idx]).to(DEVICE).float()
        actions = actions.reshape((data_length, 1, -1))  # shape (data_length, 1, n)
        actions_all = torch.cat((actions_all, actions), dim=1)

        reward_weights = train_cfg["critic"]["reward"]["weights"]
        rewards = get_rewards(data_struct, reward_weights, data_length)
        rewards = rewards.reshape((data_length, 1))  # shape (data_length, 1)
        rewards_all = torch.cat((rewards_all, rewards), dim=1)

    data_dict["actor_obs"] = actor_obs_all[:-1]
    data_dict["next_actor_obs"] = actor_obs_all[1:]
    data_dict["critic_obs"] = critic_obs_all[:-1]
    data_dict["next_critic_obs"] = critic_obs_all[1:]
    data_dict["actions"] = actions_all[:-1]
    data_dict["rewards"] = rewards_all[:-1]

    # No time outs and dones
    data_dict["timed_out"] = torch.zeros(batch_size, device=DEVICE, dtype=bool)
    data_dict["dones"] = torch.zeros(batch_size, device=DEVICE, dtype=bool)

    return data_dict


def setup():
    args = get_args()

    env_cfg, train_cfg = task_registry.create_cfgs(args)
    task_registry.make_gym_and_sim()

    if USE_SIMULATOR:
        env = task_registry.make_env(name=args.task, env_cfg=env_cfg)
    else:
        env = None

    train_cfg = class_to_dict(train_cfg)
    data_onpol = get_data_dict(train_cfg, offpol=False)

    if train_cfg["runner"]["algorithm_class_name"] == "PPO_IPG":
        data_offpol = get_data_dict(train_cfg, offpol=True)
    else:
        data_offpol = None

    print(data_onpol)
    print(data_offpol)

    runner = FineTuneRunner(
        env,
        train_cfg,
        data_onpol,
        data_offpol,
        exploration_scale=EXPLORATION_SCALE,
        device=DEVICE,
    )
    runner._set_up_alg()

    return runner


def finetune(runner):
    load_run = runner.cfg["load_run"]
    checkpoint = runner.cfg["checkpoint"]
    load_path = os.path.join(ROOT_DIR, load_run, "model_" + str(checkpoint) + ".pt")
    print("Loading model from: ", load_path)
    runner.load(load_path)

    # Perform a single update
    runner.learn()

    # Compare old and new actions
    actions_old = runner.data_onpol["actions"]
    action_scales = torch.tensor(ACTION_SCALES).to(DEVICE)
    actions_new = action_scales * runner.alg.actor.act_inference(
        runner.data_onpol["actor_obs"]
    )
    diff = actions_new - actions_old
    print("Mean action diff per actuator: ", diff.mean(dim=(0, 1)))
    print("Overall mean action diff: ", diff.mean())

    save_path = os.path.join(ROOT_DIR, load_run, "model_" + str(checkpoint + 1) + ".pt")
    runner.save(save_path)

    export_path = os.path.join(ROOT_DIR, load_run, "exported_" + str(checkpoint + 1))
    runner.export(export_path)


if __name__ == "__main__":
    runner = setup()
    finetune(runner)
