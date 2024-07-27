from gym.envs import __init__  # noqa: F401
from gym.utils import get_args, task_registry
from gym.utils.helpers import class_to_dict

from learning.runners.finetune_runner import FineTuneRunner

from gym import LEGGED_GYM_ROOT_DIR

import os
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = f"{LEGGED_GYM_ROOT_DIR}/logs/mini_cheetah_ref/"
# SE_PATH = f"{LEGGED_GYM_ROOT_DIR}/logs/SE/model_1000.pt"  # if None: no SE
SE_PATH = None
LOAD_RUN = "Jul24_22-48-41_nu05_B8"

REWARDS_FILE = (
    "rewards_nu09_nosim2.csv"  # generate this file from logs, if None: just plot
)

PLOT_REWARDS = {
    "Nu=0.5 no sim": "rewards_nu05_nosim.csv",
    "Nu=0.9 no sim": "rewards_nu09_nosim.csv",
    "Nu=0.9 no sim 2": "rewards_nu09_nosim2.csv",
    "Nu=0.95 no sim": "rewards_nu095_nosim.csv",
}

# Data struct fields from Robot-Software logs
DATA_LIST = [
    "header",
    "base_height",  # 1
    "base_lin_vel",  # 2
    "base_ang_vel",  # 3
    "projected_gravity",  # 4
    "commands",  # 5
    "dof_pos_obs",  # 6
    "dof_vel",  # 7
    "phase_obs",  # 8
    "grf",  # 9
    "dof_pos_target",  # 10
    "torques",  # 11
    "exploration_noise",  # 12
    "footer",
]

REWARD_WEIGHTS = {
    "tracking_lin_vel": 4.0,
    "tracking_ang_vel": 2.0,
    "min_base_height": 1.5,
    "orientation": 1.0,
    "stand_still": 1.0,
    "swing_grf": 3.0,
    "stance_grf": 3.0,
    "action_rate": 0.01,
    "action_rate2": 0.001,
}

DEVICE = "cuda"


def update_rewards_df(runner):
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
        runner.data_struct = data[runner.data_name][0][0]
        if runner.SE:
            runner.update_state_estimates()
        data_dict[iteration] = runner.data_struct

    rewards_path = os.path.join(ROOT_DIR, LOAD_RUN, REWARDS_FILE)
    if os.path.exists(rewards_path):
        os.remove(rewards_path)

    # Get rewards from runner
    for it, data_struct in data_dict.items():
        _, rewards_dict = runner.get_data_rewards(data_struct, REWARD_WEIGHTS)

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
        rewards_df.to_csv(rewards_path, index=False)


def plot_rewards(rewards_df, axs, name):
    for i, key in enumerate(REWARD_WEIGHTS.keys()):
        rewards_mean = rewards_df[rewards_df["type"] == key]["mean"].reset_index(
            drop=True
        )
        rewards_std = rewards_df[rewards_df["type"] == key]["std"].reset_index(
            drop=True
        )
        axs[i].plot(rewards_mean, label=name)
        axs[i].fill_between(
            range(len(rewards_mean)),
            rewards_mean - rewards_std,
            rewards_mean + rewards_std,
            alpha=0.2,
        )
        axs[i].set_title(key)
        axs[i].legend()

    i = num_plots - 1
    total_mean = rewards_df[rewards_df["type"] == "total"]["mean"].reset_index(
        drop=True
    )
    total_std = rewards_df[rewards_df["type"] == "total"]["std"].reset_index(drop=True)
    axs[i].plot(total_mean, label=name)
    axs[i].fill_between(
        range(len(total_mean)),
        total_mean - total_std,
        total_mean + total_std,
        alpha=0.2,
    )
    axs[i].set_title("Total Rewards")
    axs[i].legend()


def setup():
    args = get_args()

    env_cfg, train_cfg = task_registry.create_cfgs(args)
    task_registry.make_gym_and_sim()
    env = task_registry.make_env(name=args.task, env_cfg=env_cfg)

    train_cfg = class_to_dict(train_cfg)
    log_dir = os.path.join(ROOT_DIR, train_cfg["runner"]["load_run"])

    runner = FineTuneRunner(
        env,
        train_cfg,
        log_dir,
        data_list=DATA_LIST,
        device=DEVICE,
        se_path=SE_PATH,
    )

    return runner


if __name__ == "__main__":
    if REWARDS_FILE is not None:
        runner = setup()
        update_rewards_df(runner)

    # Plot rewards stats
    num_plots = len(REWARD_WEIGHTS) + 1  # +1 for total rewards
    cols = 5
    rows = np.ceil(num_plots / cols).astype(int)
    fig, axs = plt.subplots(rows, cols, figsize=(20, 8))
    fig.suptitle("IPG Finetuning Rewards")
    axs = axs.flatten()

    for name, file in PLOT_REWARDS.items():
        path = os.path.join(ROOT_DIR, LOAD_RUN, file)
        rewards_df = pd.read_csv(path)
        plot_rewards(rewards_df, axs, name)

    for i in range(num_plots):
        axs[i].set_xlabel("Iter")
    for i in range(num_plots, len(axs)):
        axs[i].axis("off")

    plt.tight_layout()
    plt.savefig(f"{ROOT_DIR}/{LOAD_RUN}/rewards_stats.png")
    plt.show()
