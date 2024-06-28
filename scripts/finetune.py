from gym.envs import __init__  # noqa: F401
from gym.utils import get_args, task_registry
from gym.utils.helpers import class_to_dict, get_load_path

from learning.algorithms import *  # noqa: F403
from learning.runners.finetune_runner import FineTuneRunner
from gym.envs.mini_cheetah.minimalist_cheetah import MinimalistCheetah

from gym import LEGGED_GYM_ROOT_DIR

import torch
import scipy.io
from tensordict import TensorDict

LOG_DIR = f"{LEGGED_GYM_ROOT_DIR}/logs/Robot-Software/"
LOG_FILE = "sim_data.mat"

# Data struct fields from Robot-Software logs
DATA_LIST = [
    "header",
    "base_height",  # shape (1, batch_size)
    # the following are all shape (batch_size, n)
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "commands",
    "dof_pos_obs",
    "dof_vel",
    "phase_obs",
    "grf",
    "dof_pos_target",
    "footer",
]

# Finetune for these rewards instead of the ones in the cfg
REWARD_WEIGHTS = {
    "tracking_lin_vel": 4.0,
    "tracking_ang_vel": 2.0,
    "orientation": 1.0,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_obs(obs_list, data_struct, batch_size):
    obs_all = torch.empty(0).to(DEVICE)
    for obs_name in obs_list:
        data_idx = DATA_LIST.index(obs_name)
        obs = torch.tensor(data_struct[data_idx]).to(DEVICE)
        obs = obs.reshape((batch_size, 1, -1))  # shape (batch_size, 1, n)
        obs_all = torch.cat((obs_all, obs), dim=-1)

    return obs_all.float()


def get_rewards(data_struct, batch_size):
    rewards_all = torch.empty(0).to(DEVICE)
    for i in range(batch_size):
        mini_cheetah = MinimalistCheetah(device=DEVICE)
        base_height = data_struct[1][:, i]  # shape (1, batch_size)
        base_lin_vel = data_struct[2][i]
        base_ang_vel = data_struct[3][i]
        proj_gravity = data_struct[4][i]
        commands = data_struct[5][i]

        mini_cheetah.set_states(
            base_height, base_lin_vel, base_ang_vel, proj_gravity, commands
        )
        total_rewards = 0
        for name, weight in REWARD_WEIGHTS.items():
            reward = eval(f"mini_cheetah._reward_{name}()")
            total_rewards += weight * reward
        rewards_all = torch.cat((rewards_all, total_rewards), dim=0)

    return rewards_all.float()


def get_data_dict(train_cfg, name="SMOOTH_RL_CONTROLLER"):
    data = scipy.io.loadmat(LOG_DIR + LOG_FILE)
    data_struct = data[name][0][0]
    batch_size = data_struct[1].shape[1]  # base_height: shape (1, batch_size)
    data_dict = TensorDict(
        {},
        device=DEVICE,
        batch_size=(batch_size - 1, 1),  # -1 for next_obs
    )

    actor_obs = get_obs(train_cfg["actor"]["obs"], data_struct, batch_size)
    critic_obs = get_obs(train_cfg["critic"]["obs"], data_struct, batch_size)

    data_dict["actor_obs"] = actor_obs[:-1]
    data_dict["next_actor_obs"] = actor_obs[1:]
    data_dict["critic_obs"] = critic_obs[:-1]
    data_dict["next_critic_obs"] = critic_obs[1:]

    actions_idx = DATA_LIST.index("dof_pos_target")
    actions = torch.tensor(data_struct[actions_idx]).to(DEVICE)
    actions = actions.reshape((batch_size, 1, -1))  # shape (batch_size, 1, n)
    data_dict["actions"] = actions[:-1]

    rewards = get_rewards(data_struct, batch_size)
    rewards = rewards.reshape((batch_size, 1))  # shape (batch_size, 1)
    data_dict["rewards"] = rewards[:-1]

    # Assume no termination
    data_dict["timed_out"] = torch.zeros(batch_size - 1, 1, device=DEVICE, dtype=bool)
    data_dict["terminal"] = torch.zeros(batch_size - 1, 1, device=DEVICE, dtype=bool)
    data_dict["dones"] = torch.zeros(batch_size - 1, 1, device=DEVICE, dtype=bool)

    return data_dict


def setup():
    args = get_args()

    _, train_cfg = task_registry.create_cfgs(args)
    train_cfg = class_to_dict(train_cfg)
    data_dict = get_data_dict(train_cfg)

    runner = FineTuneRunner(train_cfg, data_dict, device=DEVICE)
    runner._set_up_alg()

    return runner


def finetune(runner):
    load_path = get_load_path(
        name=runner.cfg["experiment_name"],
        load_run=runner.cfg["load_run"],
        checkpoint=runner.cfg["checkpoint"],
    )
    runner.load(load_path)

    # Perform a single update
    runner.learn()

    save_path = LOG_DIR + "finetuned_model.pth"
    runner.save(save_path)

    export_path = LOG_DIR + "finetuned_export"
    runner.export(export_path)


if __name__ == "__main__":
    runner = setup()
    finetune(runner)
