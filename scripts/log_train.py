from gym.envs import __init__  # noqa: F401
from gym.utils import get_args, task_registry, randomize_episode_counters
from gym.utils.logging_and_saving import wandb_singleton
from gym.utils.logging_and_saving import local_code_save_helper
from gym import LEGGED_GYM_ROOT_DIR

# torch needs to be imported after isaacgym imports in local source
import torch
import os
import numpy as np

TRAIN_ITERATIONS = 100
ROLLOUT_TIMESTEPS = 32


def create_logging_dict(runner):
    states_to_log = [
        "dof_pos_target",
        "dof_pos_obs",
        "dof_vel",
        "torques",
        "commands",
        # 'base_lin_vel',
        # 'base_ang_vel',
        # 'oscillators',
        # 'grf',
        # 'base_height'
    ]

    states_to_log_dict = {}

    for state in states_to_log:
        array_dim = runner.get_obs_size(
            [
                state,
            ]
        )
        states_to_log_dict[state] = torch.zeros(
            (1, TRAIN_ITERATIONS, ROLLOUT_TIMESTEPS, array_dim),
            device=runner.env.device,
        )
    return states_to_log_dict


def setup():
    args = get_args()
    wandb_helper = wandb_singleton.WandbSingleton()

    env_cfg, train_cfg = task_registry.create_cfgs(args)
    task_registry.make_gym_and_sim()
    wandb_helper.setup_wandb(env_cfg=env_cfg, train_cfg=train_cfg, args=args)
    env = task_registry.make_env(name=args.task, env_cfg=env_cfg)
    randomize_episode_counters(env)

    train_cfg.runner.max_iterations = TRAIN_ITERATIONS
    policy_runner = task_registry.make_alg_runner(env, train_cfg)

    local_code_save_helper.save_local_files_to_logs(train_cfg.log_dir)

    return train_cfg, policy_runner


def train(train_cfg, policy_runner):
    protocol_name = train_cfg.runner.experiment_name

    # * set up logging
    log_file_path = os.path.join(
        LEGGED_GYM_ROOT_DIR, "gym", "logs", "data", "train", protocol_name + ".npz"
    )
    if not os.path.exists(os.path.dirname(log_file_path)):
        os.makedirs(os.path.dirname(log_file_path))

    states_to_log_dict = create_logging_dict(policy_runner)

    # * train
    # wandb_helper = wandb_singleton.WandbSingleton()

    policy_runner.learn(states_to_log_dict)

    # wandb_helper.close_wandb()

    # * save data
    # first convert tensors to cpu
    log_dict_cpu = {k: v.cpu() for k, v in states_to_log_dict.items()}
    np.savez_compressed(log_file_path, **log_dict_cpu)
    print("saved to ", log_file_path)
    return states_to_log_dict


if __name__ == "__main__":
    train_cfg, policy_runner = setup()
    train(train_cfg=train_cfg, policy_runner=policy_runner)
