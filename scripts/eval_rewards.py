from gym.envs.mini_cheetah.minimalist_cheetah import MinimalistCheetah

from gym import LEGGED_GYM_ROOT_DIR

import os
import torch
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = f"{LEGGED_GYM_ROOT_DIR}/logs/mini_cheetah_ref/"
LOAD_RUN = "Jul13_01-49-59_PPO32_S16"
LOG_FILE = "PPO_701_a001.mat"
REWARDS_FILE = "rewards_a001.csv"

ITERATION = 701  # load this iteration

PLOT = True
PLOT_N = 1000  # number of steps to plot

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


def get_rewards(data_struct):
    data_length = data_struct[1].shape[1]  # base_height: shape (1, data_length)

    mini_cheetah = MinimalistCheetah(device=DEVICE)
    rewards_dict = {name: [] for name in REWARD_WEIGHTS.keys()}  # for plotting
    rewards_all = torch.empty(0).to(DEVICE)

    for i in range(data_length):
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
                "iteration": ITERATION,
                "type": name,
                "mean": mean,
                "std": std,
            },
            ignore_index=True,
        )
        if PLOT:
            plt.plot(rewards[:PLOT_N], label=name)
    rewards_df.to_csv(rewards_path, index=False)
    print(rewards_df)


def setup(name="SMOOTH_RL_CONTROLLER"):
    path = os.path.join(ROOT_DIR, LOAD_RUN, LOG_FILE)
    data = scipy.io.loadmat(path)
    data_struct = data[name][0][0]
    return data_struct


if __name__ == "__main__":
    data_struct = setup()
    get_rewards(data_struct)

    if PLOT:
        plt.title("Rewards")
        plt.legend()
        plt.show()
