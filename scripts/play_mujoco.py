"""Play back a policy trained with the MuJoCo backend.

Usage:
    uv run scripts/play_mujoco.py --task pendulum
    uv run scripts/play_mujoco.py --task mini_cheetah --num_envs 8
    uv run scripts/play_mujoco.py --task mini_cheetah --load_run May08_12-34-56_ \
                                  --checkpoint 1500

Mirrors scripts/play.py but uses task_registry.make_env_mujoco() instead of
the IsaacGym path.  KeyboardInterface and VisualizationRecorder are IsaacGym-
specific and are not used here.
"""

import argparse
import random

import torch

from gym.utils.helpers import set_seed
from gym.utils.task_registry import task_registry


def get_play_args():
    parser = argparse.ArgumentParser(
        description="Play a policy trained with the MuJoCo backend"
    )
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--load_run",
        type=str,
        default=None,
        help="Run directory name under logs/<experiment_name>/ (default: latest)",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=-1,
        help="Model iteration index (default: latest)",
    )
    parser.add_argument("--headless", action="store_true", default=False)
    return parser.parse_args()


def setup():
    args = get_play_args()

    # Register tasks (imports the task classes).
    import gym.envs  # noqa: F401

    env_cfg, train_cfg = task_registry.get_cfgs(args.task)

    # Play-time overrides — small batch, long episodes, no pushes.
    env_cfg.env.num_envs = args.num_envs
    env_cfg.env.episode_length_s = 50
    if hasattr(env_cfg, "commands"):
        env_cfg.commands.resampling_time = 9999
    if hasattr(env_cfg, "push_robots"):
        env_cfg.push_robots.toggle = False
    if hasattr(env_cfg, "init_state") and hasattr(env_cfg.init_state, "reset_mode"):
        env_cfg.init_state.reset_mode = "reset_to_range"

    if args.seed is not None:
        env_cfg.seed = args.seed
        train_cfg.seed = args.seed
    else:
        env_cfg.seed = random.randint(0, 10000)
        train_cfg.seed = env_cfg.seed

    train_cfg.runner.device = args.device
    train_cfg.runner.resume = True
    if args.load_run is not None:
        train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.checkpoint
    train_cfg.logging.enable_local_saving = False

    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    task_registry.set_log_dir_name(train_cfg)
    set_seed(env_cfg.seed)

    env = task_registry.make_env_mujoco(
        args.task, env_cfg, device=args.device, headless=args.headless
    )

    runner = task_registry.make_alg_runner(env, train_cfg)
    runner.switch_to_eval()
    return env, runner


def play(env, runner):
    n_steps = 10 * int(env.max_episode_length)
    for _ in range(n_steps):
        runner.set_actions(
            runner.actor_cfg["actions"],
            runner.get_inference_actions(),
            runner.actor_cfg["disable_actions"],
        )
        env.step()
        env.check_exit()


if __name__ == "__main__":
    with torch.no_grad():
        env, runner = setup()
        play(env, runner)
