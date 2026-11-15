"""Record foot ground-contact forces and gait-cycle state from a frozen go2trot policy.

The study helper, not product code. It mirrors ``scripts/play.py`` setup but
records the per-control-step tensors needed to characterise the foot contact
force over the locomotion cycle:

* ``env.contact_forces`` for the canonical ``feet`` body group (Newtons, world
  frame), plus the largest contact norm on any non-foot body;
* phase / phase frequency, so force can be binned by gait phase;
* the policy-facing actor observation (the eventual Q input);
* joint positions, velocities, targets, torques and base state for the
  observability probe.

Run from the product repo root:

    uv run --frozen python \
      docs/plans/supplements/foot_contact_study/collect_foot_contacts.py \
      --device cuda:0 --num_envs 128 --steps 600 --tag gpu
"""

import argparse
import hashlib
import json
import pathlib
import random
import time

import numpy as np
import torch

from gym.utils.helpers import set_seed
from gym.utils.task_registry import task_registry

STUDY_DIR = pathlib.Path(__file__).resolve().parent

SCALAR_STEP_FIELDS = (
    "phase",
    "phase_frequency",
    "base_height",
    "episode_length_buf",
    "terminated",
    "timed_out",
)
VECTOR_STEP_FIELDS = (
    ("commands", ("commands",)),
    ("dof_pos", ("dof_pos",)),
    ("dof_vel", ("dof_vel",)),
    ("dof_pos_target", ("dof_pos_target",)),
    ("torques", ("torques",)),
    ("root_states", ("root_states",)),
    ("base_lin_vel", ("base_lin_vel",)),
    ("base_ang_vel", ("base_ang_vel",)),
    ("projected_gravity", ("projected_gravity",)),
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="go2trot")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", default="mujoco", choices=["mujoco", "vsim"])
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--load_run", default="Nov10_10-36-22_gpu-final-600-trot-hist-20261110"
    )
    parser.add_argument("--checkpoint", type=int, default=600)
    parser.add_argument(
        "--original_cfg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the environment/runner configs saved with the checkpoint run",
    )
    parser.add_argument(
        "--push",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep the training-time push perturbation (off: nominal gait)",
    )
    parser.add_argument("--episode_length_s", type=float, default=50.0)
    parser.add_argument("--command_resampling_s", type=float, default=3.0)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions from the policy distribution (training behaviour) "
        "instead of the deterministic mean (playback behaviour)",
    )
    parser.add_argument("--tag", default="gpu")
    return parser.parse_args(argv)


def setup(args):
    import gym.envs  # noqa: F401

    env_cfg, train_cfg = task_registry.get_cfgs(
        args.task,
        original_cfg=args.original_cfg,
        load_run=args.load_run,
    )
    env_cfg.env.num_envs = args.num_envs
    env_cfg.env.episode_length_s = args.episode_length_s
    env_cfg.commands.resampling_time = args.command_resampling_s
    env_cfg.push_robots.toggle = args.push
    env_cfg.init_state.reset_mode = "reset_to_basic"
    env_cfg.seed = args.seed
    train_cfg.seed = args.seed

    train_cfg.runner.device = args.device
    train_cfg.runner.resume = True
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
        headless=True,
        backend=args.backend,
    )
    runner = task_registry.make_alg_runner(env, train_cfg)
    runner.switch_to_eval()
    return env_cfg, train_cfg, env, runner


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    env_cfg, train_cfg, env, runner = setup(args)
    if env.dt * 1.0 != env_cfg.sim_dt or env.cfg.control.decimation != 1:
        raise RuntimeError(
            "this recorder assumes one control step per physics step; "
            f"dt={env.dt} sim_dt={env_cfg.sim_dt} "
            f"decimation={env.cfg.control.decimation}"
        )

    obs_names = list(runner.actor_cfg["obs"])
    obs_dims = [int(env.get_state(name).shape[1]) for name in obs_names]
    feet = env.feet_indices.to(env.device)
    foot_names = [env.robot_layout.body_names[int(i)] for i in feet]
    body_names = list(env.robot_layout.body_names)
    non_foot = torch.ones(len(body_names), dtype=torch.bool, device=env.device)
    non_foot[feet] = False

    steps = args.steps
    num_envs = args.num_envs
    rec = {
        name: np.zeros((steps, num_envs), dtype=np.float32)
        for name in SCALAR_STEP_FIELDS
    }
    rec.update(
        {
            name: np.zeros(
                (steps, num_envs, int(env.get_state(attrs[0]).shape[1])),
                dtype=np.float32,
            )
            for name, attrs in VECTOR_STEP_FIELDS
        }
    )
    rec["foot_forces"] = np.zeros(
        (steps, num_envs, len(foot_names), 3), dtype=np.float32
    )
    rec["actor_obs"] = np.zeros((steps, num_envs, int(sum(obs_dims))), dtype=np.float32)
    rec["nonfoot_contact_norm"] = np.zeros((steps, num_envs), dtype=np.float32)

    total_mass = env._backend.link_mass.sum(dim=1).clone()
    print(
        f"env: {num_envs} worlds, {steps} steps, dt={env.dt}s, "
        f"feet={foot_names}, total mass {total_mass.mean().item():.3f} kg"
    )

    reset_count = 0
    started = time.perf_counter()
    with torch.no_grad():
        for step in range(steps):
            if args.stochastic:
                actions = runner.alg.actor.act(runner.get_obs(obs_names))
            else:
                actions = runner.get_inference_actions()
            runner.set_actions(
                runner.actor_cfg["actions"],
                actions,
                runner.actor_cfg["disable_actions"],
            )
            env.step()

            forces = env.contact_forces
            rec["foot_forces"][step] = (
                forces.index_select(1, feet).detach().float().cpu().numpy()
            )
            rec["nonfoot_contact_norm"][step] = (
                forces.masked_select(non_foot.view(1, -1, 1))
                .reshape(num_envs, -1)
                .norm(dim=1)
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            for name in SCALAR_STEP_FIELDS:
                value = getattr(env, name)
                rec[name][step] = value.detach().float().cpu().numpy().reshape(num_envs)
            for name, attrs in VECTOR_STEP_FIELDS:
                # Raw simulator units, not observation-scaled: get_states()
                # divides by env.scales (commands by 3, velocities by 0.3/0.4).
                value = getattr(env, attrs[0])
                rec[name][step] = value.detach().float().cpu().numpy()
            rec["actor_obs"][step] = (
                runner.get_obs(obs_names).detach().float().cpu().numpy()
            )

            reset_mask = torch.logical_or(env.terminated, env.timed_out)
            reset_count += int(reset_mask.sum().item())
            if bool(reset_mask.any()):
                env._reset_idx(reset_mask)
        runner_end = time.perf_counter()

    elapsed = runner_end - started
    out_dir = STUDY_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{args.tag}.npz"
    np.savez_compressed(
        npz_path,
        **rec,
        total_mass=total_mass.detach().float().cpu().numpy(),
        feet_indices=feet.detach().cpu().numpy(),
        dt=np.float64(env.dt),
        steps=np.int64(steps),
        num_envs=np.int64(num_envs),
    )

    runner_dir = pathlib.Path(train_cfg.log_dir).parent / train_cfg.runner.load_run
    ckpt = runner_dir / f"model_{args.checkpoint}.pt"
    meta = {
        "tag": args.tag,
        "task": args.task,
        "backend": args.backend,
        "device": args.device,
        "seed": args.seed,
        "num_envs": num_envs,
        "steps": steps,
        "dt": float(env.dt),
        "control_frequency": float(1.0 / env.dt),
        "decimation": int(env.cfg.control.decimation),
        "episode_length_s": float(env.cfg.env.episode_length_s),
        "command_resampling_s": float(env.cfg.commands.resampling_time),
        "push_robots": bool(env.cfg.push_robots.toggle),
        "stochastic_actions": bool(args.stochastic),
        "load_run": args.load_run,
        "checkpoint": args.checkpoint,
        "checkpoint_path": str(ckpt),
        "checkpoint_sha256": sha256_file(ckpt) if ckpt.exists() else None,
        "obs_names": obs_names,
        "obs_dims": obs_dims,
        "obs_dim": int(sum(obs_dims)),
        "feet_names": foot_names,
        "body_names": body_names,
        "total_mass_kg_mean": float(total_mass.mean().item()),
        "total_mass_kg_std": float(total_mass.std().item()),
        "resets": reset_count,
        "wall_seconds": elapsed,
        "env_steps_per_s": num_envs * steps / elapsed,
        "contact_force_tensor": "backend contact_forces[:, body, :] (world frame, N)",
    }
    (out_dir / f"{args.tag}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    print(f"wrote {npz_path} ({npz_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
