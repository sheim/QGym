"""Legged cross-engine fidelity probes for mini_cheetah_ref (Phase 4 parity).

Two policy-free, deterministic probes driven identically on every backend, which
SEPARATE the two physics regimes that a quadruped exercises:

  drop  — free base, PD-hold the default pose, released from a height onto the
          plane.  The CONTACT / impact-model comparison (feet <-> ground).
  step  — fix_base_link=True, spawned clear of the ground: step the desired
          joint angles and record the joint response.  Pure PD + limb dynamics,
          NO contact — isolates actuator / integration differences.

`dof_pos_target` is a residual on the default pose (torque =
p_gains*(dof_pos_target + default - dof_pos)), so 0 holds default and a nonzero
entry steps that joint.  reset_to_basic gives a deterministic IC identical on
every backend; contact termination is disabled so trajectories run full length.

Let PF="scripts/mini_cheetah_fidelity.py", F=logs/mc_fid (for vsim, prefix
`uv run --env-file .env.vsim` and pass `--backend vsim`):

    uv run $PF run --probe drop --backend mujoco --device cpu    --out $F/drop_cpu.npz
    uv run $PF run --probe step --backend mujoco --device cuda:0 --out $F/step_warp.npz
    uv run $PF compare $F/drop_*.npz

Expectation: cpu ~= warp tight (same MuJoCo solver); vsim diverges on the drop
(different contact solver) more than on the contact-free step.
"""

import argparse
import os

import numpy as np
import torch

from gym.utils.helpers import set_seed
from gym.utils.task_registry import task_registry

TASK = "mini_cheetah_ref"


def reset_probe_state(env):
    """Restore the configured deterministic IC after task construction.

    TaskSkeleton.reset() performs one implicit control step during construction.
    That step is useful for normal task initialization but would make a fidelity
    probe start from an already-evolved, backend-dependent state.
    """
    env_ids = torch.arange(env.num_envs, device=env.device)
    env.dof_pos_target.zero_()
    env._reset_system(env_ids)
    env.dof_pos_target.zero_()


def build_env(backend, device, num_envs, fixed_base, base_z, t_end):
    import gym.envs  # noqa: F401 — registers tasks

    env_cfg, train_cfg = task_registry.get_cfgs(TASK)
    env_cfg.env.num_envs = num_envs
    env_cfg.asset.fix_base_link = fixed_base
    # Full-length trajectories: no timeout- or contact-driven resets mid-probe.
    env_cfg.asset.terminate_after_contacts_on = []
    env_cfg.env.episode_length_s = t_end + 10.0
    env_cfg.init_state.reset_mode = "reset_to_basic"
    env_cfg.init_state.pos = [0.0, 0.0, base_z]
    env_cfg.seed = 0
    train_cfg.seed = 0
    set_seed(0)

    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    return task_registry.make_env_mujoco(
        TASK, env_cfg, device=device, headless=True, backend=backend
    )


def run_drop(env, n_steps):
    """Free base falls from height, PD holds the default pose."""
    nfeet = len(env.feet_indices)
    base_pos = np.empty((n_steps, env.num_envs, 3), dtype=np.float32)
    base_quat = np.empty((n_steps, env.num_envs, 4), dtype=np.float32)
    base_lin_vel = np.empty((n_steps, env.num_envs, 3), dtype=np.float32)
    base_ang_vel = np.empty((n_steps, env.num_envs, 3), dtype=np.float32)
    dof_pos = np.empty((n_steps, env.num_envs, env.num_dof), dtype=np.float32)
    grf = np.empty((n_steps, env.num_envs, nfeet), dtype=np.float32)
    with torch.no_grad():
        for k in range(n_steps):
            env.dof_pos_target[:] = 0.0  # hold default pose
            env.step()
            base_pos[k] = env.root_states[:, :3].detach().cpu().numpy()
            base_quat[k] = env.root_states[:, 3:7].detach().cpu().numpy()
            base_lin_vel[k] = env.root_states[:, 7:10].detach().cpu().numpy()
            base_ang_vel[k] = env.root_states[:, 10:13].detach().cpu().numpy()
            dof_pos[k] = env.dof_pos.detach().cpu().numpy()
            f = torch.norm(env.contact_forces[:, env.feet_indices, :], dim=-1)
            grf[k] = f.detach().cpu().numpy()
    feet = env.feet_indices.detach().cpu().tolist()
    foot_names = [env._backend.body_names[i] for i in feet]
    # vsim sensor definitions retain XML insertion order, which need not match
    # the engine's link order. Record both so the probe can expose (and avoid
    # hiding) a contact-tensor/body-name contract mismatch.
    grf_names = foot_names
    art = getattr(env._backend, "_art_instance", None)
    if art is not None:
        art_def = art.get_articulation_def()
        sensor_links = [
            art_def.get_force_sensor_def(i).link_name
            for i in range(art_def.get_num_force_sensor_defs())
        ]
        grf_names = [sensor_links[i] for i in feet]
    return {
        "base_pos": base_pos,
        "base_z": base_pos[..., 2],
        "base_quat": base_quat,
        "base_lin_vel": base_lin_vel,
        "base_ang_vel": base_ang_vel,
        "dof_pos": dof_pos,
        "dof_names": np.asarray(env.dof_names),
        "grf": grf,
        "foot_names": np.asarray(foot_names),
        "grf_names": np.asarray(grf_names),
    }


def run_step(env, n_steps, settle, deltas):
    """Fixed base: hold default, then step every joint by a per-env delta."""
    nd = env.num_dof
    dof_pos = np.empty((n_steps, env.num_envs, nd), dtype=np.float32)
    d = torch.tensor(deltas, dtype=torch.float, device=env.device)
    with torch.no_grad():
        for k in range(n_steps):
            if k < settle:
                env.dof_pos_target[:] = 0.0
            else:
                env.dof_pos_target[:] = d[:, None]
            env.step()
            dof_pos[k] = env.dof_pos.detach().cpu().numpy()
    return {
        "dof_pos": dof_pos,
        "deltas": np.asarray(deltas, dtype=np.float32),
        "settle": settle,
        "default_dof_pos": env.default_dof_pos.detach().cpu().numpy().ravel(),
        "dof_names": np.array(env.dof_names),
    }


def run(args):
    hz_key = "ctrl_frequency"
    if args.probe == "drop":
        env = build_env(
            args.backend,
            args.device,
            args.num_envs,
            fixed_base=False,
            base_z=0.5,
            t_end=args.t_end,
        )
        n_steps = int(args.t_end * float(getattr(env.cfg.control, hz_key)))
        reset_probe_state(env)
        data = run_drop(env, n_steps)
    else:
        env = build_env(
            args.backend,
            args.device,
            args.num_envs,
            fixed_base=True,
            base_z=1.0,
            t_end=args.t_end,
        )
        n_steps = int(args.t_end * float(getattr(env.cfg.control, hz_key)))
        settle = int(0.5 * float(getattr(env.cfg.control, hz_key)))
        deltas = np.linspace(-0.3, 0.3, args.num_envs)
        reset_probe_state(env)
        data = run_step(env, n_steps, settle, deltas)

    label = args.backend if args.backend == "vsim" else f"mujoco-{args.device}"
    env._backend.close()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        probe=args.probe,
        label=label,
        ctrl_hz=float(getattr(env.cfg.control, hz_key)),
        **data,
    )
    if args.probe == "drop":
        print(
            f"[{label}] drop: final base_z mean {data['base_z'][-1].mean():.4f} "
            f"| peak grf {data['grf'].max():.1f} N | wrote {args.out}"
        )
    else:
        settled = data["dof_pos"][-1]  # [N, nd]
        print(
            f"[{label}] step: final joint spread "
            f"{settled.std(0).mean():.4f} rad | wrote {args.out}"
        )


def compare(args):
    data = {}
    for path in args.files:
        d = np.load(path, allow_pickle=True)
        data[str(d["label"])] = d
    ref_label = next((k for k in data if k.startswith("mujoco-cpu")), list(data)[0])
    ref = data[ref_label]
    probe = str(ref["probe"])

    def aligned_grf(d):
        ref_names = [str(name) for name in ref.get("grf_names", ref["foot_names"])]
        names = [str(name) for name in d.get("grf_names", d["foot_names"])]
        return d["grf"][..., [names.index(name) for name in ref_names]]

    print(f"\nprobe: {probe}  ·  reference: {ref_label}")
    if probe == "drop":
        print(f"{'engine':<16}{'settle-z RMS':>14}{'quat RMS':>12}{'grf-time RMS':>14}")
        for label, d in data.items():
            if label == ref_label:
                print(f"{label:<16}{'—':>14}{'—':>12}{'—':>14}")
                continue
            zr = np.sqrt(np.mean((d["base_z"] - ref["base_z"]) ** 2))
            qr = np.sqrt(np.mean((d["base_quat"] - ref["base_quat"]) ** 2))
            gr = np.sqrt(np.mean((aligned_grf(d) - aligned_grf(ref)) ** 2))
            print(f"{label:<16}{zr:>14.2e}{qr:>12.2e}{gr:>14.2e}")
    else:
        print(f"{'engine':<16}{'joint-traj RMS':>16}{'final-pos RMS':>15}")
        for label, d in data.items():
            if label == ref_label:
                print(f"{label:<16}{'—':>16}{'—':>15}")
                continue
            tr = np.sqrt(np.mean((d["dof_pos"] - ref["dof_pos"]) ** 2))
            fr = np.sqrt(np.mean((d["dof_pos"][-1] - ref["dof_pos"][-1]) ** 2))
            print(f"{label:<16}{tr:>16.2e}{fr:>15.2e}")
    print(
        "\ndrop RMS is contact-model divergence (expect vsim >> warp); step RMS is"
        " contact-free PD/limb divergence (expect all small if actuation matches)."
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--probe", choices=["drop", "step"], required=True)
    r.add_argument("--backend", choices=["mujoco", "vsim"], default="mujoco")
    r.add_argument("--device", default="cpu")
    r.add_argument("--num_envs", type=int, default=32)
    r.add_argument("--t_end", type=float, default=3.0)
    r.add_argument("--out", required=True)
    r.set_defaults(func=run)
    c = sub.add_parser("compare")
    c.add_argument("files", nargs="+")
    c.set_defaults(func=compare)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
