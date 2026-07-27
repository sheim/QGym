"""Evaluate a trained pendulum policy on a chosen backend (Phase 4 RL transfer).

Loads a checkpoint trained on backend A and rolls its *deterministic* policy on
backend B over the deterministic ``reset_to_uniform`` grid, so every (train,
eval) pair sees identical initial conditions.  Reports, per env: the mean
weighted reward (same figure the training curve reports — the time-average of
Σ_term wₜ·rₜ, dt cancels), the per-term breakdown, and whether the pendulum is
held upright at the end.  Dumps an npz per cell; run the full A×B sweep, then
inspect in ``notebooks/pendulum_rl.py``.

    uv run scripts/eval_policy.py --ckpt logs/pendulum_cpu --train_label cpu \
        --eval_backend mujoco --eval_device cpu --out logs/rl_eval/cpu__cpu.npz

    # cross-engine: a cpu-trained policy evaluated under vsim
    uv run --env-file .env.vsim scripts/eval_policy.py --ckpt logs/pendulum_cpu \
        --train_label cpu --eval_backend vsim --out logs/rl_eval/cpu__vsim.npz
"""

import argparse
import math
import os

import numpy as np
import torch

from gym.utils.helpers import set_seed
from gym.utils.task_registry import task_registry


def resolve_ckpt(path):
    """Accept a run dir (pick the highest-iteration model) or a .pt file."""
    if os.path.isfile(path):
        return path
    models = [
        f for f in os.listdir(path) if f.startswith("model_") and f.endswith(".pt")
    ]
    if not models:
        # maybe a logs/<exp> dir with timestamped run subdirs — take the latest run
        runs = sorted(
            (os.path.join(path, d) for d in os.listdir(path)),
            key=os.path.getmtime,
        )
        if not runs:
            raise FileNotFoundError(f"no checkpoints under {path}")
        return resolve_ckpt(runs[-1])
    latest = max(models, key=lambda f: int(f[len("model_") : -len(".pt")]))
    return os.path.join(path, latest)


def build(task, eval_backend, eval_device, num_envs, t_end, ckpt, reset_mode):
    import gym.envs  # noqa: F401 — registers tasks

    # reset_to_uniform lays ICs on a sqrt(N)xsqrt(N) grid (pendulum) — needs a
    # perfect square.  Legged tasks use reset_to_range (distributional eval).
    if reset_mode == "reset_to_uniform":
        root = int(math.isqrt(num_envs))
        if root * root != num_envs:
            raise ValueError(f"num_envs must be a perfect square; got {num_envs}")

    env_cfg, train_cfg = task_registry.get_cfgs(task)
    env_cfg.env.num_envs = num_envs
    env_cfg.init_state.reset_mode = reset_mode
    # Keep the eval controlled: no pushes, fixed commands for the whole episode.
    if hasattr(env_cfg, "push_robots"):
        env_cfg.push_robots.toggle = False
    if hasattr(env_cfg, "commands"):
        env_cfg.commands.resampling_time = t_end + 10.0
    # reset_to_uniform runs one long episode; range-reset legged eval keeps the
    # task's own episode length so survival-to-timeout is meaningful.
    if reset_mode == "reset_to_uniform":
        env_cfg.env.episode_length_s = t_end + 10.0
    env_cfg.seed = 0
    train_cfg.seed = 0
    train_cfg.runner.device = eval_device
    train_cfg.runner.resume = False  # we load the checkpoint explicitly below
    if hasattr(train_cfg, "logging"):
        train_cfg.logging.enable_local_saving = False
    set_seed(0)

    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    task_registry.set_log_dir_name(train_cfg, log_root=None)  # no run dir on disk

    env = task_registry.make_env_mujoco(
        task, env_cfg, device=eval_device, headless=True, backend=eval_backend
    )
    runner = task_registry.make_alg_runner(env, train_cfg)
    runner.load(resolve_ckpt(ckpt), load_optimizer=False)
    runner.switch_to_eval()
    return env, runner


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="pendulum")
    p.add_argument("--ckpt", required=True, help="run dir or model_*.pt")
    p.add_argument("--train_label", required=True, help="e.g. cpu / warp / vsim")
    p.add_argument("--eval_backend", choices=["mujoco", "vsim"], default="mujoco")
    p.add_argument("--eval_device", default="cpu")
    p.add_argument("--eval_label", default=None, help="defaults to backend/device")
    p.add_argument("--num_envs", type=int, default=1024)
    p.add_argument("--t_end", type=float, default=10.0)
    p.add_argument(
        "--reset_mode",
        default="reset_to_uniform",
        help="reset_to_uniform (pendulum grid) or reset_to_range (legged)",
    )
    p.add_argument(
        "--record_dof",
        action="store_true",
        help="record the full dof_pos trajectory [T,N,ndof] (for cross-backend "
        "DOF-RMS comparison; pair with --reset_mode reset_to_basic so ICs match)",
    )
    p.add_argument("--out", required=True)
    args = p.parse_args()

    eval_label = args.eval_label or (
        args.eval_backend
        if args.eval_backend == "vsim"
        else f"mujoco-{args.eval_device}"
    )

    env, runner = build(
        args.task,
        args.eval_backend,
        args.eval_device,
        args.num_envs,
        args.t_end,
        args.ckpt,
        args.reset_mode,
    )
    weights = runner.critic_cfg["reward"]["weights"]  # {term: weight}, zeros removed
    terms = list(weights)
    n_steps = int(args.t_end * float(env.cfg.control.ctrl_frequency))
    N, dev = args.num_envs, env.device
    is_pendulum = args.task == "pendulum"

    per_term_sum = {t: torch.zeros(N, device=dev) for t in terms}
    base_z = np.empty((n_steps, N), dtype=np.float32)
    terminated = np.zeros((n_steps, N), dtype=np.bool_)
    ever_term = torch.zeros(N, dtype=torch.bool, device=dev)
    first_term = torch.full((N,), n_steps, dtype=torch.long, device=dev)
    theta = np.empty((n_steps, N), dtype=np.float32) if is_pendulum else None
    omega = np.empty((n_steps, N), dtype=np.float32) if is_pendulum else None
    # projected_gravity z: -1 upright, 0 on its side (legged uprightness).
    upright = None if is_pendulum else np.empty((n_steps, N), dtype=np.float32)
    dof_traj = (
        np.empty((n_steps, N, env.num_dof), dtype=np.float32)
        if args.record_dof
        else None
    )

    with torch.no_grad():
        for k in range(n_steps):
            actions = runner.get_inference_actions()
            runner.set_actions(
                runner.actor_cfg["actions"],
                actions,
                runner.actor_cfg["disable_actions"],
            )
            base_z[k] = env.root_states[:, 2].detach().cpu().numpy()
            if dof_traj is not None:
                dof_traj[k] = env.dof_pos.detach().cpu().numpy()
            if is_pendulum:
                theta[k] = env.dof_pos[:, 0].detach().cpu().numpy()
                omega[k] = env.dof_vel[:, 0].detach().cpu().numpy()
            else:
                upright[k] = env.projected_gravity[:, 2].detach().cpu().numpy()
            env.step()
            for term, w in weights.items():
                per_term_sum[term] += w * runner.reward_functions[term]().to(dev)
            term = env.terminated
            terminated[k] = term.detach().cpu().numpy()
            newly = term & ~ever_term
            first_term[newly] = k
            ever_term |= term

    env._backend.close()

    # Mean over steps => same scale as the training total_rewards curve.
    per_term_mean = {t: (per_term_sum[t] / n_steps).cpu().numpy() for t in terms}
    mean_reward = np.sum([per_term_mean[t] for t in terms], axis=0)
    survived = (~ever_term).detach().cpu().numpy()
    ep_len = (first_term.float() / float(env.cfg.control.ctrl_frequency)).cpu().numpy()

    extra = {}
    if is_pendulum:
        tw = np.arctan2(np.sin(theta), np.cos(theta))
        hold = int(round(float(env.cfg.control.ctrl_frequency)))
        success = (np.abs(tw[-hold:]) < 0.14).all(0) & (
            np.abs(omega[-hold:]) < 0.5
        ).all(0)
        extra = {"theta": theta, "omega": omega, "success": success}
        headline = f"upright {100 * success.mean():.1f}%"
    else:
        extra = {"upright": upright}
        headline = f"survival {100 * survived.mean():.1f}%"
    if dof_traj is not None:
        extra["dof_traj"] = dof_traj
        extra["dof_names"] = np.array(env.dof_names)

    print(
        f"[{args.train_label} -> {eval_label}] mean reward "
        f"{mean_reward.mean():+.3f}  |  {headline}"
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        train_label=args.train_label,
        eval_label=eval_label,
        task=args.task,
        mean_reward=mean_reward.astype(np.float32),
        survived=survived,
        ep_len=ep_len.astype(np.float32),
        base_z=base_z,
        terminated=terminated,
        terms=np.array(terms),
        ctrl_hz=float(env.cfg.control.ctrl_frequency),
        **{f"term_{t}": per_term_mean[t].astype(np.float32) for t in terms},
        **extra,
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
