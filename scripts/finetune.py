from gym.envs import __init__  # noqa: F401
from gym.utils import get_args, task_registry
from gym.utils.helpers import class_to_dict

from learning.runners.finetune_runner import FineTuneRunner

from gym import LEGGED_GYM_ROOT_DIR

import os
import torch
import numpy as np
import pandas as pd

ROOT_DIR = f"{LEGGED_GYM_ROOT_DIR}/logs/mini_cheetah_ref/"
# SE_PATH = f"{LEGGED_GYM_ROOT_DIR}/logs/SE/model_1000.pt"  # if None: no SE
SE_PATH = None
OUTPUT_FILE = "output.txt"
LOSSES_FILE = "losses.csv"

USE_SIMULATOR = True

# Load/save off-policy storage, this can contain many runs
OFFPOL_LOAD_FILE = "offpol_data_10.pt"
OFFPOL_SAVE_FILE = None

# Scales
EXPLORATION_SCALE = 0.5  # used during data collection
ACTION_SCALES = np.tile(np.array([0.2, 0.3, 0.3]), 4)

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

DEVICE = "cuda"


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
        se_path=SE_PATH,
        use_simulator=USE_SIMULATOR,
        exploration_scale=EXPLORATION_SCALE,
        device=DEVICE,
    )

    return runner


def finetune(runner):
    # Load model
    load_run = runner.cfg["load_run"]
    checkpoint = runner.cfg["checkpoint"]
    model_path = os.path.join(ROOT_DIR, load_run, "model_" + str(checkpoint) + ".pt")
    runner.load(model_path)

    # Load data
    load_path = (
        os.path.join(ROOT_DIR, load_run, OFFPOL_LOAD_FILE) if OFFPOL_LOAD_FILE else None
    )
    save_path = (
        os.path.join(ROOT_DIR, load_run, OFFPOL_SAVE_FILE) if OFFPOL_SAVE_FILE else None
    )
    runner.load_data(load_path=load_path, save_path=save_path)

    # Get old inference actions
    action_scales = torch.tensor(ACTION_SCALES).to(DEVICE)
    actions_old = action_scales * runner.alg.actor.act_inference(
        runner.data_onpol["actor_obs"]
    )

    # Perform a single update
    runner.learn()

    # Compare old to new actions
    actions_new = action_scales * runner.alg.actor.act_inference(
        runner.data_onpol["actor_obs"]
    )
    diff = actions_new - actions_old

    # Save and export
    save_path = os.path.join(ROOT_DIR, load_run, "model_" + str(checkpoint + 1) + ".pt")
    export_path = os.path.join(ROOT_DIR, load_run, "exported_" + str(checkpoint + 1))
    runner.save(save_path)
    runner.export(export_path)

    # Print to output file
    with open(os.path.join(ROOT_DIR, load_run, OUTPUT_FILE), "a") as f:
        f.write(f"############ Checkpoint: {checkpoint} #######################\n")
        f.write(f"############## Nu={runner.alg.inter_nu} ###################\n")
        f.write("############### DATA ###############\n")
        f.write(f"Data on-policy shape: {runner.data_onpol.shape}\n")
        if runner.data_offpol is not None:
            f.write(f"Data off-policy shape: {runner.data_offpol.shape}\n")
        f.write("############## LOSSES ##############\n")
        f.write(f"Mean Value Loss: {runner.alg.mean_value_loss}\n")
        f.write(f"Mean Surrogate Loss: {runner.alg.mean_surrogate_loss}\n")
        if runner.data_offpol is not None:
            f.write(f"Mean Q Loss: {runner.alg.mean_q_loss}\n")
            f.write(f"Mean Offpol Loss: {runner.alg.mean_offpol_loss}\n")
        f.write("############## ACTIONS #############\n")
        f.write(f"Mean action diff per actuator: {diff.mean(dim=(0, 1))}\n")
        f.write(f"Std action diff per actuator: {diff.std(dim=(0, 1))}\n")
        f.write(f"Overall mean action diff: {diff.mean()}\n")

    # Log losses to csv
    losses_path = os.path.join(ROOT_DIR, load_run, LOSSES_FILE)
    if not os.path.exists(losses_path):
        if runner.data_offpol is None:
            losses_df = pd.DataFrame(
                columns=["checkpoint", "value_loss", "surrogate_loss"]
            )
        else:
            losses_df = pd.DataFrame(
                columns=[
                    "checkpoint",
                    "value_loss",
                    "q_loss",
                    "surrogate_loss",
                    "offpol_loss",
                ]
            )
    else:
        losses_df = pd.read_csv(losses_path)

    append_data = {
        "checkpoint": checkpoint,
        "value_loss": runner.alg.mean_value_loss,
        "surrogate_loss": runner.alg.mean_surrogate_loss,
    }
    if runner.data_offpol is not None:
        append_data["q_loss"] = runner.alg.mean_q_loss
        append_data["offpol_loss"] = runner.alg.mean_offpol_loss

    losses_df = losses_df._append(append_data, ignore_index=True)
    losses_df.to_csv(losses_path, index=False)


if __name__ == "__main__":
    runner = setup()
    finetune(runner)
