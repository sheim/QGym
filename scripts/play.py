"""Play back a policy trained with a supported Q2 physics backend.

Usage:
    uv run scripts/play.py --task pendulum
    uv run scripts/play.py --task mini_cheetah --num_envs 8
    uv run scripts/play.py --task mini_cheetah --load_run May08_12-34-56_ \
                           --checkpoint 1500

Supports MuJoCo CPU/Warp and optional VSim checkpoints.
"""

import argparse
import random

import torch

from gym.utils.helpers import set_seed
from gym.utils.task_registry import task_registry


def get_play_args():
    parser = argparse.ArgumentParser(description="Play a policy trained with Q2")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Checkpoint root under logs/ (default: task config value)",
    )
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
    parser.add_argument(
        "--backend",
        type=str,
        default="mujoco",
        choices=["mujoco", "vsim"],
        help="Physics backend (vsim is CUDA-only; start via "
        "`uv run --env-file .env.vsim ...`)",
    )
    parser.add_argument(
        "--viewer_ui",
        action="store_true",
        default=False,
        help="show the MuJoCo viewer's side panels (hidden by default: their "
        "shortcuts bind most letters to visualisation toggles, which fire "
        "alongside keyboard teleop)",
    )
    parser.add_argument(
        "--keyboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keyboard teleop in the MuJoCo viewer (see "
        "gym/utils/interfaces/MujocoKeyboardInterface.py for key bindings). "
        "Use --no-keyboard to disable.",
    )
    return parser.parse_args()


def setup(args):
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
    env_cfg.init_state.reset_mode = "reset_to_basic"
    if hasattr(env_cfg, "viewer"):
        env_cfg.viewer.show_ui = args.viewer_ui

    if args.seed is not None:
        env_cfg.seed = args.seed
        train_cfg.seed = args.seed
    else:
        env_cfg.seed = random.randint(0, 10000)
        train_cfg.seed = env_cfg.seed

    train_cfg.runner.device = args.device
    train_cfg.runner.resume = True
    if args.experiment_name is not None:
        train_cfg.runner.experiment_name = args.experiment_name
    if args.load_run is not None:
        train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.checkpoint
    train_cfg.logging.enable_local_saving = False

    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    task_registry.set_log_dir_name(train_cfg)
    set_seed(env_cfg.seed)

    env = task_registry.make_env(
        args.task,
        env_cfg,
        device=args.device,
        headless=args.headless,
        backend=args.backend,
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
    args = get_play_args()
    with torch.no_grad():
        env, runner = setup(args)
        # The MuJoCo warp (GPU) backend is headless-only: render() is a no-op and
        # it has no passive viewer, so nothing draws.  Point the user at cpu.
        if (
            args.backend == "mujoco"
            and not args.headless
            and not hasattr(env._backend, "_viewer_overlay_fn")
        ):
            print(
                "WARNING: the MuJoCo warp (GPU) backend has no interactive viewer "
                "— nothing will render. Use --device cpu for playback."
            )
        if args.keyboard and hasattr(env, "commands") and not args.headless:
            # Each viewer has its own input model: MuJoCo dispatches key
            # events via a callback, vlearn polls key state per frame.
            if args.backend == "vsim":
                from gym.utils.interfaces.VsimKeyboardInterface import (
                    VsimKeyboardInterface,
                )

                VsimKeyboardInterface(env)
            else:
                from gym.utils.interfaces.MujocoKeyboardInterface import (
                    MujocoKeyboardInterface,
                )

                MujocoKeyboardInterface(env)
        if hasattr(env, "commands") and not args.headless:
            # Same indicators, different drawing primitives per viewer.
            if args.backend == "vsim":
                from gym.utils.interfaces.VsimCommandVisualizer import (
                    VsimCommandVisualizer,
                )

                VsimCommandVisualizer(env)
            else:
                from gym.utils.interfaces.CommandVisualizer import CommandVisualizer

                CommandVisualizer(env)
        play(env, runner)
