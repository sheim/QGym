"""Train a task using a supported Q2 physics backend.

Usage:
    uv run scripts/train.py --task pendulum [--device cpu] [--num_envs 256]
                            [--max_iterations 200] [--headless]

The selected backend is MuJoCo CPU/Warp or optional VSim.
"""

import argparse
import subprocess
from dataclasses import asdict

from gym import GYM_ROOT_DIR
from gym.utils.task_registry import task_registry
from gym.utils.helpers import randomize_episode_counters, set_seed
from gym.utils.logging_and_saving import local_code_save_helper
from gym.utils.logging_and_saving import wandb_singleton
from learning.offline.contracts import RecorderConfig


def _source_revision():
    result = subprocess.run(
        ["git", "-C", GYM_ROOT_DIR, "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def make_train_parser():
    parser = argparse.ArgumentParser(description="Train a Q2 task")
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
    parser.add_argument(
        "--save_interval",
        type=int,
        default=None,
        help="Override checkpoint interval in learning iterations",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Override batch_size from cfg"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment folder under logs/ (default: task config value)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run label appended to the timestamp in the log folder and W&B name",
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
    parser.add_argument(
        "--original_cfg",
        action="store_true",
        help="Load environment and runner configs saved with the selected run.",
    )
    parser.add_argument("--headless", action="store_true", default=False)
    # replay recording
    parser.add_argument(
        "--replay_record",
        action="store_true",
        default=False,
        help="Record training transitions into a bounded replay buffer.",
    )
    parser.add_argument(
        "--replay_capacity",
        type=int,
        default=1_000_000,
        help="Maximum transitions retained in the replay ring.",
    )
    parser.add_argument(
        "--replay_snapshot_interval",
        type=int,
        default=0,
        help="Snapshot every N iterations; 0 snapshots only at the end.",
    )
    parser.add_argument(
        "--replay_output_dir",
        type=str,
        default=None,
        help="Replay snapshot root (default: <log_dir>/replay).",
    )
    parser.add_argument(
        "--replay_seed",
        type=int,
        default=0,
        help="Seed for recording-only observation noise.",
    )
    # wandb
    parser.add_argument("--disable_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    return parser


def get_train_args(argv=None):
    return make_train_parser().parse_args(argv)


def setup(args=None, env_cfg=None, train_cfg=None):
    if args is None:
        args = get_train_args()
    if (env_cfg is None) != (train_cfg is None):
        raise ValueError("env_cfg and train_cfg must be supplied together")

    # Register tasks (imports the task classes)
    import gym.envs  # noqa: F401 — triggers task registration

    if env_cfg is None:
        env_cfg, train_cfg = task_registry.get_cfgs(
            args.task,
            original_cfg=args.original_cfg,
            experiment_name=args.experiment_name,
            load_run=args.load_run,
        )
    elif args.original_cfg:
        raise ValueError("original_cfg cannot be combined with supplied configs")

    # Apply CLI overrides
    if args.num_envs is not None:
        env_cfg.env.num_envs = args.num_envs
    if args.max_iterations is not None:
        train_cfg.runner.max_iterations = args.max_iterations
    if args.save_interval is not None:
        if args.save_interval <= 0:
            raise ValueError("--save_interval must be positive")
        train_cfg.runner.save_interval = args.save_interval
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
    if args.run_name is not None:
        train_cfg.runner.run_name = args.run_name
    if args.resume:
        train_cfg.runner.resume = True
    if args.load_run is not None:
        train_cfg.runner.load_run = args.load_run
    if args.checkpoint is not None:
        train_cfg.runner.checkpoint = args.checkpoint

    train_cfg.runner.device = args.device
    train_cfg.runner.replay = asdict(
        RecorderConfig(
            enabled=args.replay_record,
            capacity=args.replay_capacity,
            snapshot_interval=args.replay_snapshot_interval,
            output_dir=args.replay_output_dir or "",
            seed=args.replay_seed,
            task_name=args.task,
            backend=args.backend,
            source_revision=_source_revision(),
        )
    )

    # Compute sim_dt, decimation, discount rates
    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    task_registry.set_log_dir_name(train_cfg)

    set_seed(env_cfg.seed)

    # wandb
    wandb_helper = wandb_singleton.WandbSingleton()
    wandb_helper.setup_wandb(env_cfg=env_cfg, train_cfg=train_cfg, args=args)

    # Build the environment using the selected backend.
    env = task_registry.make_env(
        args.task,
        env_cfg,
        device=args.device,
        headless=args.headless,
        backend=args.backend,
    )

    randomize_episode_counters(env)

    policy_runner = task_registry.make_alg_runner(env, train_cfg)

    local_code_save_helper.save_local_files_to_logs(
        train_cfg.log_dir,
        original_cfg_source_dir=(
            train_cfg._original_cfg_source_dir if args.original_cfg else None
        ),
    )

    return train_cfg, policy_runner


def train(train_cfg, policy_runner):
    wandb_helper = wandb_singleton.WandbSingleton()
    policy_runner.learn()
    wandb_helper.close_wandb()


if __name__ == "__main__":
    train_cfg, policy_runner = setup()
    train(train_cfg=train_cfg, policy_runner=policy_runner)
