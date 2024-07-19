from gym.envs.mini_cheetah.minimalist_cheetah import MinimalistCheetah

from gym import LEGGED_GYM_ROOT_DIR

import os
import torch
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = f"{LEGGED_GYM_ROOT_DIR}/logs/mini_cheetah_ref/"
LOAD_RUN = "Jul12_15-53-57_IPG32_S16"
REWARDS_FILE = "rewards.csv"

DATAFRAME_LOADED = True
PLOT = False
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

REWARD_WEIGHTS = {
    "tracking_lin_vel": 4.0,
    "tracking_ang_vel": 2.0,
    "min_base_height": 1.5,
    "orientation": 1.0,
    "stand_still": 2.0,
    "swing_grf": 3.0,
    "stance_grf": 3.0,
    "action_rate": 0.01,
    "action_rate2": 0.001,
}

DEVICE = "cuda"


def get_rewards(it, data_struct, rewards_path):
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
                "iteration": it,
                "type": name,
                "mean": mean,
                "std": std,
            },
            ignore_index=True,
        )
        if PLOT:
            plt.figure(it)
            plt.plot(rewards[:PLOT_N], label=name)
            plt.title(f"Rewards Iteration {it}")
            plt.legend()
            plt.savefig(f"{ROOT_DIR}/{LOAD_RUN}/rewards_{it}.png")
    rewards_df.to_csv(rewards_path, index=False)


def setup(name="SMOOTH_RL_CONTROLLER"):
    data_dict = {}
    log_files = [
        file
        for file in os.listdir(os.path.join(ROOT_DIR, LOAD_RUN))
        if file.endswith(".mat")
    ]
    log_files = sorted(log_files)
    for file in log_files:
        iteration = int(file.split(".")[0])
        path = os.path.join(ROOT_DIR, LOAD_RUN, file)
        data = scipy.io.loadmat(path)
        data_dict[iteration] = data[name][0][0]
    return data_dict


if __name__ == "__main__":
    data_dict = setup()
    rewards_path = os.path.join(ROOT_DIR, LOAD_RUN, REWARDS_FILE)

    if not DATAFRAME_LOADED:
        # Compute rewards and store in dataframe
        if os.path.exists(rewards_path):
            os.remove(rewards_path)
        for it, data_struct in data_dict.items():
            get_rewards(it, data_struct, rewards_path)

    rewards_df = pd.read_csv(rewards_path)
    print(rewards_df)

    # Plot rewards stats
    num_plots = len(REWARD_WEIGHTS) + 1  # +1 for total rewards
    cols = 5
    rows = np.ceil(num_plots / cols).astype(int)
    fig, axs = plt.subplots(rows, cols, figsize=(15, 6))
    fig.suptitle("IPG Finetuning Rewards")
    axs = axs.flatten()

    for i, key in enumerate(REWARD_WEIGHTS.keys()):
        rewards_mean = rewards_df[rewards_df["type"] == key]["mean"].reset_index(
            drop=True
        )
        rewards_std = rewards_df[rewards_df["type"] == key]["std"].reset_index(
            drop=True
        )
        axs[i].plot(rewards_mean, label=key)
        axs[i].fill_between(
            range(len(rewards_mean)),
            rewards_mean - rewards_std,
            rewards_mean + rewards_std,
            color="b",
            alpha=0.2,
        )
        axs[i].legend()
        axs[i].set_xlabel("Iter")

    i = num_plots - 1
    total_mean = rewards_df[rewards_df["type"] == "total"]["mean"].reset_index(
        drop=True
    )
    total_std = rewards_df[rewards_df["type"] == "total"]["std"].reset_index(drop=True)
    axs[i].plot(total_mean, label="total", color="r")
    axs[i].fill_between(
        range(len(total_mean)),
        total_mean - total_std,
        total_mean + total_std,
        color="r",
        alpha=0.2,
    )
    axs[i].legend()
    axs[i].set_xlabel("Iter")

    for i in range(num_plots, len(axs)):
        axs[i].axis("off")

    plt.tight_layout()
    plt.show()
