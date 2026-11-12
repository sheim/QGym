"""Intervened single-robot playback with a frozen PPO actor and height Q.

Loads the Q artifact's frozen policy checkpoint, takes eight stochastic policy
draws per control step, and executes either the max-Q draw (``maxq``) or the
first draw with ``Q >= threshold`` falling back to max-Q (``rejection``).

Playback starts and resets at zero commands. Keyboard commands are held until
the next key event; task command resampling is disabled for the process only.
Auto-reset is off by default, so a fall or timeout leaves the robot in place.
MuJoCo CPU with exactly one environment is the supported playback target.
"""

from __future__ import annotations

import argparse
import copy
import math
import random
from pathlib import Path

import torch

from gym import GYM_ROOT_DIR
from gym.utils.helpers import set_seed
from gym.utils.task_registry import task_registry
from learning.offline.evaluation import (
    _policy_observation,
    load_policy_from_artifact,
    policy_spec_from_artifact,
)
from learning.offline.height_q_playback import (
    AUTO_RESET_MODES,
    PLAYBACK_MODES,
    HeightQPlaybackController,
    apply_auto_reset,
    disable_command_resampling,
    restore_command_resampling,
    validate_env_widths,
    validate_playback_contract,
)
from learning.offline.q_artifact import HeightQ


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--artifact",
        required=True,
        help="Height-Q artifact directory containing manifest.json and q.pt",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Frozen policy checkpoint (default: artifact policy_spec.checkpoint)",
    )
    parser.add_argument("--mode", choices=PLAYBACK_MODES, default="maxq")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Required rejection threshold c; Q >= c is feasible",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Playback requires cpu; cuda/Warp playback is unsupported",
    )
    parser.add_argument(
        "--backend",
        choices=("mujoco", "vsim"),
        default="mujoco",
        help="Playback requires mujoco; vsim is unsupported",
    )
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Stop after this many steps; 0 runs until the viewer closes",
    )
    parser.add_argument(
        "--auto_reset",
        choices=AUTO_RESET_MODES,
        default="off",
        help="off (default) keeps the robot in place after fall/timeout",
    )
    parser.add_argument(
        "--keyboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keyboard teleop in the MuJoCo viewer (default: on)",
    )
    parser.add_argument(
        "--viewer_ui",
        action="store_true",
        default=False,
        help="Show the MuJoCo side panels (hidden by default)",
    )
    parser.add_argument(
        "--original_cfg",
        action="store_true",
        help="Load the environment/runner config saved with the selected run",
    )
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--load_run", default=None)
    parser.add_argument(
        "--no-push_robots",
        action="store_true",
        default=False,
        help="Disable the task's configured push disturbances for playback",
    )
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode == "rejection":
        if args.threshold is None:
            parser.error("--threshold is required with --mode rejection")
        if not math.isfinite(args.threshold):
            parser.error("--threshold must be finite")
    elif args.threshold is not None:
        parser.error("--threshold is only valid with --mode rejection")
    if args.device != "cpu":
        parser.error("play_height_q.py supports only --device cpu")
    if args.backend != "mujoco":
        parser.error("play_height_q.py supports only --backend mujoco")
    if args.num_envs != 1:
        parser.error("play_height_q.py supports exactly one environment")
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    if args.headless and args.steps == 0:
        parser.error("--headless requires --steps N for a finite smoke run")
    if (args.experiment_name is not None or args.load_run is not None) and (
        not args.original_cfg
    ):
        parser.error("--experiment_name/--load_run require --original_cfg")
    return args


def resolve_checkpoint(checkpoint: str | None, artifact_default: str) -> Path:
    """Resolve an explicit path from the CWD and an artifact path from the repo."""
    if checkpoint is None:
        path = Path(artifact_default)
        if not path.is_absolute():
            path = Path(GYM_ROOT_DIR) / path
        return path
    return Path(checkpoint)


def _viewer_closed(env) -> bool:
    viewer = env._backend._viewer
    return viewer is not None and not viewer.is_running()


def _close_viewer(env) -> None:
    viewer = env._backend._viewer
    if viewer is not None:
        viewer.close()


def _apply_keyboard_actions(teleop, keyboard) -> None:
    if keyboard is None:
        return
    for action in keyboard.drain():
        teleop.apply(action)


def run(args: argparse.Namespace) -> dict:
    import gym.envs  # noqa: F401

    artifact = HeightQ(args.artifact)
    spec = policy_spec_from_artifact(artifact)
    checkpoint = resolve_checkpoint(args.checkpoint, spec.checkpoint)

    env_cfg, train_cfg = copy.deepcopy(
        task_registry.get_cfgs(
            args.task,
            original_cfg=args.original_cfg,
            experiment_name=args.experiment_name,
            load_run=args.load_run,
        )
    )
    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)

    # Identity checks precede playback-only overrides. The fingerprint therefore
    # covers the task as trained, not num_envs/viewer/push playback settings.
    fingerprint = validate_playback_contract(
        env_cfg,
        train_cfg,
        artifact,
        spec,
        checkpoint,
    )

    env_cfg.env.num_envs = 1
    env_cfg.viewer.show_ui = args.viewer_ui
    if args.no_push_robots:
        if not hasattr(env_cfg, "push_robots"):
            raise ValueError("--no-push_robots requires a task with push_robots")
        env_cfg.push_robots.toggle = False
    seed = args.seed if args.seed is not None else random.randint(0, 10000)
    env_cfg.seed = seed
    train_cfg.seed = seed
    set_seed(seed)

    env = task_registry.make_env(
        args.task,
        env_cfg,
        device="cpu",
        headless=args.headless,
        backend="mujoco",
    )
    try:
        validate_env_widths(env, spec)
        policy, loaded_spec = load_policy_from_artifact(artifact, checkpoint, "cpu")
        if loaded_spec != spec:
            raise RuntimeError("policy spec changed while loading the checkpoint")

        # The task constructor already completed its normal reset. Zero after it
        # and before the first policy action; no direct backend writes are used.
        env.commands.zero_()
        original_resample = disable_command_resampling(env)

        from gym.utils.interfaces.teleop_bindings import TeleopCommands

        zero_commands = (0.0, 0.0, 0.0)
        teleop = TeleopCommands(
            env,
            initial_commands=zero_commands,
            reset_commands=zero_commands,
            preserve_resampling_time=True,
        )
        keyboard = None
        if not args.headless:
            from gym.utils.interfaces.CommandVisualizer import CommandVisualizer

            CommandVisualizer(env)
        if args.keyboard and not args.headless:
            from gym.utils.interfaces.MujocoKeyboardInterface import (
                MujocoKeyboardInterface,
            )

            # Defer native UI-thread key events to the task thread.
            keyboard = MujocoKeyboardInterface(env, commands=teleop, deferred=True)

        observation_generator = torch.Generator(device=env.device).manual_seed(seed + 1)
        action_generator = torch.Generator(device=env.device).manual_seed(seed + 2)
        controller = HeightQPlaybackController(
            policy,
            artifact,
            mode=args.mode,
            threshold=args.threshold,
            generator=action_generator,
        )
        threshold_text = "n/a" if args.threshold is None else f"{args.threshold:g}"
        print(
            "Height-Q playback: "
            f"task={args.task} mode={args.mode} threshold={threshold_text} "
            f"auto_reset={args.auto_reset}"
        )
        pushes = (
            env.cfg.push_robots.toggle if hasattr(env.cfg, "push_robots") else "n/a"
        )
        print(
            f"artifact={args.artifact} fingerprint={fingerprint[:12]} "
            f"seed={seed} pushes={pushes}"
        )
        print(
            f"checkpoint={checkpoint} commands={env.commands[0].tolist()} "
            f"num_envs=1 device=cpu"
        )

        steps = 0
        resets = 0
        fallbacks = 0
        interrupted = False
        try:
            while args.steps == 0 or steps < args.steps:
                _apply_keyboard_actions(teleop, keyboard)
                if env.exit or _viewer_closed(env):
                    break
                observation = _policy_observation(
                    env,
                    spec.obs_names,
                    spec.obs_dims,
                    spec.noise,
                    spec.scales,
                    observation_generator,
                )
                decision = controller.act(observation)
                env.set_states(spec.action_names, decision.action)
                env.step()
                steps += 1
                fallbacks += int(decision.fallback.sum().item())
                resets += apply_auto_reset(env, args.auto_reset)
                if steps % 100 == 0:
                    print(
                        f"step={steps} commands={env.commands[0].tolist()} "
                        f"resets={resets} fallbacks={fallbacks}"
                    )
                if env.exit or _viewer_closed(env):
                    break
            if args.steps and steps < args.steps:
                print(f"stopped early at step={steps}")
        except KeyboardInterrupt:
            interrupted = True
            print("\nInterrupted by user.")
        finally:
            _close_viewer(env)
            restore_command_resampling(env, original_resample)
    finally:
        env._backend.close()

    summary = {
        "steps": steps,
        "resets": resets,
        "fallbacks": fallbacks,
        "mode": args.mode,
        "threshold": args.threshold,
        "final_commands": env.commands[0].tolist(),
        "interrupted": interrupted,
    }
    print(
        "playback done: "
        f"steps={steps} resets={resets} mode={args.mode} "
        f"threshold={threshold_text} fallbacks={fallbacks} "
        f"final_commands={summary['final_commands']}"
    )
    return summary


def main(argv=None) -> int:
    args = parse_args(argv)
    with torch.no_grad():
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
