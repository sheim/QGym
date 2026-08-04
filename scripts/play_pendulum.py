"""Roll out a trained pendulum policy for 10 s and plot diagnostics.

Records (theta, omega, tau) at the control rate, then renders three panels:
phase portrait, torque vs time, and kinetic / potential / total energy.

Usage:
    uv run scripts/play_pendulum.py
    uv run scripts/play_pendulum.py --load_run May08_12-34-56_ --checkpoint 1500
    uv run scripts/play_pendulum.py --duration 20 --start theta=pi
"""

import argparse
import math
import random
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from gym.utils.helpers import set_seed
from gym.utils.task_registry import task_registry


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", type=str, default="pendulum")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--load_run", type=str, default=None)
    p.add_argument("--checkpoint", type=int, default=-1)
    p.add_argument("--duration", type=float, default=10.0, help="rollout length (s)")
    p.add_argument(
        "--start",
        type=str,
        default="pi",
        help="initial theta in rad. Use 'pi' for hanging down, '0' for upright.",
    )
    p.add_argument(
        "--output",
        type=str,
        default="pendulum_rollout.png",
        help="output plot file path",
    )
    p.add_argument(
        "--viewer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="open the MuJoCo viewer and throttle the rollout to real time",
    )
    p.add_argument("--show", action="store_true", help="open the plot in a window")
    return p.parse_args()


def setup(args):
    import gym.envs  # noqa: F401  (registers tasks)

    env_cfg, train_cfg = task_registry.get_cfgs(args.task)

    # Single env, long episode so we don't auto-reset mid-rollout.
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = max(args.duration * 10, 100)
    env_cfg.init_state.reset_mode = "reset_to_range"
    env_cfg.init_state.default_joint_angles = {
        "theta": float(eval(args.start, {"pi": math.pi}))
    }

    seed = args.seed if args.seed is not None else random.randint(0, 10000)
    env_cfg.seed = seed
    train_cfg.seed = seed

    train_cfg.runner.device = args.device
    train_cfg.runner.resume = True
    if args.load_run is not None:
        train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.checkpoint
    train_cfg.logging.enable_local_saving = False

    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    task_registry.set_log_dir_name(train_cfg)
    set_seed(seed)

    env = task_registry.make_env(
        args.task, env_cfg, device=args.device, headless=not args.viewer
    )
    runner = task_registry.make_alg_runner(env, train_cfg)
    runner.switch_to_eval()
    return env, runner


def rollout(env, runner, duration_s, realtime):
    dt = env.cfg.control.ctrl_dt
    n_steps = int(round(duration_s / dt))

    t = np.empty(n_steps)
    theta = np.empty(n_steps)
    omega = np.empty(n_steps)
    tau = np.empty(n_steps)

    wall_start = time.perf_counter()
    for k in range(n_steps):
        runner.set_actions(
            runner.actor_cfg["actions"],
            runner.get_inference_actions(),
            runner.actor_cfg["disable_actions"],
        )
        env.step()

        t[k] = k * dt
        theta[k] = env.dof_pos[0, 0].item()
        omega[k] = env.dof_vel[0, 0].item()
        tau[k] = env.torques[0, 0].item()

        if realtime:
            sleep_for = (k + 1) * dt - (time.perf_counter() - wall_start)
            if sleep_for > 0:
                time.sleep(sleep_for)
    return t, theta, omega, tau


def plot(t, theta, omega, tau, m, L, g, out_path, show):
    ke = 0.5 * m * L**2 * omega**2
    pe = m * g * L * np.cos(theta)
    e_total = ke + pe
    e_target = m * g * L  # full upright at rest

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(theta, omega, lw=1.0)
    ax.plot(theta[0], omega[0], "go", label="start")
    ax.plot(theta[-1], omega[-1], "r*", markersize=10, label="end")
    ax.axvline(0, color="gray", ls=":", lw=0.5)
    ax.axhline(0, color="gray", ls=":", lw=0.5)
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(r"$\omega$ (rad/s)")
    ax.set_title("Phase portrait")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t, tau, lw=1.0)
    ax.set_xlabel("t (s)")
    ax.set_ylabel(r"$\tau$ (N·m)")
    ax.set_title("Torque")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(t, ke, label="KE", lw=1.0)
    ax.plot(t, pe, label="PE", lw=1.0)
    ax.plot(t, e_total, label="Total", lw=1.5)
    ax.axhline(e_target, color="k", ls="--", lw=0.8, label="upright at rest")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Energy (J)")
    ax.set_title("Pendulum energy")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"saved plot to {out_path}")
    if show:
        plt.show()


def main():
    args = get_args()
    with torch.no_grad():
        env, runner = setup(args)
        t, theta, omega, tau = rollout(env, runner, args.duration, realtime=args.viewer)

    m = env.cfg.asset.mass
    L = env.cfg.asset.length
    g = abs(env.cfg.sim.gravity[2])
    plot(t, theta, omega, tau, m, L, g, args.output, args.show)


if __name__ == "__main__":
    main()
