"""Train a task using the MuJoCo backend (no IsaacGym required).

Usage:
    uv run scripts/train_mujoco.py --task pendulum [--device cpu] [--num_envs 256]
                                   [--max_iterations 200] [--headless]

This script replaces the IsaacGym-specific setup in train.py with a
backend-agnostic path that calls task_registry.make_env_mujoco().
"""

import argparse

from gym.utils.task_registry import task_registry  # , select_backend
from gym.utils.helpers import set_seed
from gym.utils.logging_and_saving import local_code_save_helper


def get_mujoco_args():
    parser = argparse.ArgumentParser(description="Train with MuJoCo backend")
    parser.add_argument(
        "--task", type=str, required=True, help="Task name (e.g. pendulum)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Torch device (cpu, cuda:0)"
    )
    parser.add_argument(
        "--num_envs", type=int, default=None, help="Override num_envs from cfg"
    )
    parser.add_argument(
        "--max_iterations", type=int, default=None, help="Override max_iterations"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Override batch_size from cfg"
    )
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--disable_wandb", action="store_true", default=True)
    return parser.parse_args()


def setup():
    args = get_mujoco_args()

    # Register tasks (imports the task classes)
    import gym.envs  # noqa: F401 — triggers task registration

    env_cfg, train_cfg = task_registry.get_cfgs(args.task)

    # Apply CLI overrides
    if args.num_envs is not None:
        env_cfg.env.num_envs = args.num_envs
    if args.max_iterations is not None:
        train_cfg.runner.max_iterations = args.max_iterations
    if args.batch_size is not None:
        train_cfg.algorithm.batch_size = args.batch_size
    if args.seed is not None:
        train_cfg.seed = args.seed
        env_cfg.seed = args.seed
    else:
        import random

        env_cfg.seed = random.randint(0, 10000)
        train_cfg.seed = env_cfg.seed

    train_cfg.runner.device = args.device

    # Compute sim_dt, decimation, discount rates
    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    task_registry.set_log_dir_name(train_cfg)

    set_seed(env_cfg.seed)

    # Build the environment using the MuJoCo backend
    env = task_registry.make_env_mujoco(
        args.task, env_cfg, device=args.device, headless=args.headless
    )

    from gym.utils import randomize_episode_counters

    randomize_episode_counters(env)

    policy_runner = task_registry.make_alg_runner(env, train_cfg)

    local_code_save_helper.save_local_files_to_logs(train_cfg.log_dir)

    return train_cfg, policy_runner


def train(train_cfg, policy_runner):
    policy_runner.learn()


if __name__ == "__main__":
    train_cfg, policy_runner = setup()
    train(train_cfg=train_cfg, policy_runner=policy_runner)
