"""Train a task using MuJoCo CPU/Warp or optional VSim.

Usage:
    uv run scripts/train_mujoco.py --task pendulum [--device cpu] [--num_envs 256]
                                   [--max_iterations 200] [--headless]

The selected backend is MuJoCo CPU/Warp or optional VSim.
"""

import argparse

from gym.utils.task_registry import task_registry
from gym.utils.helpers import randomize_episode_counters, set_seed
from gym.utils.logging_and_saving import local_code_save_helper
from gym.utils.logging_and_saving import wandb_singleton


def get_mujoco_args():
    parser = argparse.ArgumentParser(description="Train with MuJoCo backend")
    parser.add_argument(
        "--task", type=str, required=True, help="Task name (e.g. pendulum)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Torch device (cpu, cuda:0)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="mujoco",
        choices=["mujoco", "vsim"],
        help="Physics backend (vsim is CUDA-only; start via "
        "`uv run --env-file .env.vsim ...`)",
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
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Override experiment_name (log dir is logs/<experiment_name>/...)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume optimizer and model state from an existing run.",
    )
    parser.add_argument(
        "--load_run",
        type=str,
        default=None,
        help="Run directory under logs/<experiment_name>/ (default: latest).",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=None,
        help="Checkpoint iteration to resume (default: latest).",
    )
    parser.add_argument("--headless", action="store_true", default=False)
    # wandb
    parser.add_argument("--disable_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
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

    if args.experiment_name is not None:
        train_cfg.runner.experiment_name = args.experiment_name
    if args.resume:
        train_cfg.runner.resume = True
    if args.load_run is not None:
        train_cfg.runner.load_run = args.load_run
    if args.checkpoint is not None:
        train_cfg.runner.checkpoint = args.checkpoint

    train_cfg.runner.device = args.device

    # Compute sim_dt, decimation, discount rates
    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    task_registry.set_log_dir_name(train_cfg)

    set_seed(env_cfg.seed)

    # wandb
    wandb_helper = wandb_singleton.WandbSingleton()
    wandb_helper.setup_wandb(env_cfg=env_cfg, train_cfg=train_cfg, args=args)

    # Build the environment using the MuJoCo backend
    env = task_registry.make_env(
        args.task,
        env_cfg,
        device=args.device,
        headless=args.headless,
        backend=args.backend,
    )

    randomize_episode_counters(env)

    policy_runner = task_registry.make_alg_runner(env, train_cfg)

    local_code_save_helper.save_local_files_to_logs(train_cfg.log_dir)

    return train_cfg, policy_runner


def train(train_cfg, policy_runner):
    wandb_helper = wandb_singleton.WandbSingleton()
    policy_runner.learn()
    wandb_helper.close_wandb()


if __name__ == "__main__":
    train_cfg, policy_runner = setup()
    train(train_cfg=train_cfg, policy_runner=policy_runner)
