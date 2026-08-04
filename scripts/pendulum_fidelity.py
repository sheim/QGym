"""Cross-engine physics-fidelity probe for the pendulum (MIGRATION_PLAN Phase 4).

An analytic energy-pump + LQR swing-up controller drives the pendulum, reading
state and emitting torque identically on every backend.  Because the control
law is deterministic and the initial-condition grid (``reset_to_uniform``) is
identical across backends, any divergence between engines is *pure physics*
(integration / solver), with no learned-policy confound.

The controller is also a self-check: if the same law swings up and balances on
MuJoCo-CPU, MuJoCo-warp AND vsim, the engines agree on the pendulum's dynamics;
if it catches on one but not another, that localises a real inertia / damping /
torque-mapping discrepancy.

Usage (one backend per invocation — vsim needs its env file, and is a process
singleton).  Let PF="scripts/pendulum_fidelity.py" and F=logs/fidelity:

    uv run $PF run --backend mujoco --device cpu    --out $F/cpu.npz
    uv run $PF run --backend mujoco --device cuda:0 --out $F/warp.npz
    uv run --env-file .env.vsim $PF run --backend vsim --out $F/vsim.npz

    uv run $PF compare $F/cpu.npz $F/warp.npz $F/vsim.npz

The ``compare`` step matches envs by index (identical grid ICs), takes CPU as
the deterministic reference, and reports per-engine catch rate, mean
time-to-catch, and the angular-divergence RMS vs the reference over a short
(pre-chaos) window and the full episode.
"""

import argparse
import math
import os

import numpy as np
import torch

# ── Physical params (pendulum.urdf, parallel-axis about the joint; theta=0 UP) ──
I_JOINT = 1.0267  # kg m^2   I_com,yy (0.0267) + m d^2 (d = 1 m)
MGL = 9.81  # N m      m g l_com  (l_com = 1 m)
DAMPING = 0.1  # N m s    cfg.asset.joint_damping (all backends apply it)
TAU_MAX = 5.0  # N m      URDF effort limit

# Controller gains (validated on the ideal ODE: 100% catch over the 32x32 grid).
K_ENERGY = 1.0
J_SWITCH = 5.0  # cost-to-go xᵀPx below which LQR engages
J_EXIT = 15.0  # hysteresis: fall back to pumping if J climbs past this

SHORT_HORIZON_S = 1.5  # pre-chaos window for the tight divergence check


def solve_care(A, B, Q, R):
    """Continuous-time algebraic Riccati via Hamiltonian eigen-decomposition.

    Avoids a scipy dependency; exact for this 2-state system.
    """
    n = A.shape[0]
    Rinv = np.linalg.inv(R)
    H = np.block([[A, -B @ Rinv @ B.T], [-Q, -A.T]])
    w, v = np.linalg.eig(H)
    stab = np.argsort(w.real)[:n]  # n most-stable eigenvectors
    U = v[:, stab]
    P = (U[n:, :] @ np.linalg.inv(U[:n, :])).real
    return 0.5 * (P + P.T)


class SwingUpLQR:
    """Batched energy-pump + LQR controller (theta measured from upright).

    Reads torch tensors on the env's device, so it is backend-agnostic.  Keeps a
    per-env pump/LQR mode with hysteresis on the LQR cost-to-go J = xᵀPx.
    """

    def __init__(self, device):
        A = np.array([[0.0, 1.0], [MGL / I_JOINT, -DAMPING / I_JOINT]])
        B = np.array([[0.0], [1.0 / I_JOINT]])
        Q = np.diag([10.0, 1.0])
        R = np.array([[1.0]])
        P = solve_care(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P

        self.P = torch.tensor(P, dtype=torch.float, device=device)
        self.k_theta = float(K[0, 0])
        self.k_omega = float(K[0, 1])
        self._mode_lqr = None  # bool [N]; lazily sized

    @staticmethod
    def wrap(theta):
        return torch.atan2(torch.sin(theta), torch.cos(theta))

    def __call__(self, theta, omega):
        if self._mode_lqr is None or self._mode_lqr.shape != theta.shape:
            self._mode_lqr = torch.zeros_like(theta, dtype=torch.bool)

        tw = self.wrap(theta)
        # J = [tw, omega] P [tw, omega]^T, per env
        j = (
            self.P[0, 0] * tw * tw
            + 2.0 * self.P[0, 1] * tw * omega
            + self.P[1, 1] * omega * omega
        )
        self._mode_lqr = (self._mode_lqr & (j <= J_EXIT)) | (j < J_SWITCH)

        tau_lqr = -(self.k_theta * tw + self.k_omega * omega)
        energy = 0.5 * I_JOINT * omega * omega + MGL * torch.cos(theta)
        tau_pump = K_ENERGY * (MGL - energy) * omega
        tau = torch.where(self._mode_lqr, tau_lqr, tau_pump)
        return torch.clamp(tau, -TAU_MAX, TAU_MAX)


def build_env(backend, device, num_envs, t_end):
    import gym.envs  # noqa: F401 — registers tasks
    from gym.utils.helpers import set_seed
    from gym.utils.task_registry import task_registry

    root = int(math.isqrt(num_envs))
    if root * root != num_envs:
        raise ValueError(
            f"num_envs must be a perfect square for the reset_to_uniform grid; "
            f"got {num_envs}"
        )

    env_cfg, train_cfg = task_registry.get_cfgs("pendulum")
    env_cfg.env.num_envs = num_envs
    env_cfg.init_state.reset_mode = "reset_to_uniform"
    # Long episode so the deterministic grid is never re-drawn mid-run.
    env_cfg.env.episode_length_s = t_end + 10.0
    env_cfg.seed = 0
    train_cfg.seed = 0
    set_seed(0)

    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    return task_registry.make_env(
        "pendulum", env_cfg, device=device, headless=True, backend=backend
    )


def run(args):
    env = build_env(args.backend, args.device, args.num_envs, args.t_end)
    ctrl = SwingUpLQR(env.device)

    n_steps = int(args.t_end * float(env.cfg.control.ctrl_frequency))
    theta = np.empty((n_steps, args.num_envs), dtype=np.float32)
    omega = np.empty((n_steps, args.num_envs), dtype=np.float32)
    tau_log = np.empty((n_steps, args.num_envs), dtype=np.float32)
    mode = np.empty((n_steps, args.num_envs), dtype=np.bool_)

    with torch.no_grad():
        for t in range(n_steps):
            th = env.dof_pos[:, 0]
            om = env.dof_vel[:, 0]
            tau = ctrl(th, om)
            env.tau_ff[:, 0] = tau
            theta[t] = th.detach().cpu().numpy()
            omega[t] = om.detach().cpu().numpy()
            tau_log[t] = tau.detach().cpu().numpy()
            mode[t] = ctrl._mode_lqr.detach().cpu().numpy()
            env.step()

    env._backend.close()

    label = args.backend if args.backend == "vsim" else f"mujoco-{args.device}"
    caught, catch_rate, t_catch = _catch_stats(theta, omega, mode, env, args)
    print(
        f"[{label}] caught & held upright: {caught}/{args.num_envs} "
        f"({100 * catch_rate:.1f}%)"
    )
    if caught:
        print(f"[{label}] time-to-catch: mean {np.mean(t_catch):.2f}s")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        theta=theta,
        omega=omega,
        tau=tau_log,
        mode=mode,
        label=label,
        ctrl_hz=float(env.cfg.control.ctrl_frequency),
    )
    print(f"[{label}] wrote {args.out}")


def _catch_stats(theta, omega, mode, env, args):
    hz = float(env.cfg.control.ctrl_frequency)
    hold = int(round(hz))  # last ~1 s
    tw = np.arctan2(np.sin(theta), np.cos(theta))
    final_up = (np.abs(tw[-hold:]) < 0.14).all(0) & (np.abs(omega[-hold:]) < 0.5).all(0)
    caught = int(final_up.sum())
    t_catch = []
    for e in np.where(final_up)[0]:
        first = np.argmax(mode[:, e])  # first LQR engagement
        t_catch.append(first / hz)
    return caught, caught / args.num_envs, np.array(t_catch)


def _angdiff(a, b):
    d = a - b
    return np.arctan2(np.sin(d), np.cos(d))


def compare(args):
    data = {}
    for path in args.files:
        d = np.load(path, allow_pickle=True)
        data[str(d["label"])] = d

    ref_label = next((k for k in data if k.startswith("mujoco-cpu")), None)
    if ref_label is None:
        ref_label = list(data)[0]
    ref = data[ref_label]
    hz = float(ref["ctrl_hz"])
    short = int(SHORT_HORIZON_S * hz)

    # Sanity: identical grid ICs across engines (first-step state).
    for label, d in data.items():
        if label == ref_label:
            continue
        ic_gap = np.max(np.abs(_angdiff(d["theta"][0], ref["theta"][0])))
        if ic_gap > 1e-4:
            print(
                f"WARNING: {label} initial grid differs from {ref_label} "
                f"(max {ic_gap:.2e} rad) — envs may not be index-matched"
            )

    print(f"\nreference engine: {ref_label}")
    print(
        f"{'engine':<16}{'catch%':>8}{'RMSθ short':>13}{'RMSθ full':>12}"
        f"{'maxθ short':>12}"
    )
    for label, d in data.items():
        tw = np.arctan2(np.sin(d["theta"]), np.cos(d["theta"]))
        hold = int(round(hz))
        up = (np.abs(tw[-hold:]) < 0.14).all(0) & (
            np.abs(d["omega"][-hold:]) < 0.5
        ).all(0)
        catch = 100 * up.mean()
        if label == ref_label:
            print(f"{label:<16}{catch:>7.1f}%{'—':>13}{'—':>12}{'—':>12}")
            continue
        diff = _angdiff(d["theta"], ref["theta"])  # [T, N]
        rms_short = np.sqrt(np.mean(diff[:short] ** 2))
        rms_full = np.sqrt(np.mean(diff**2))
        max_short = np.max(np.abs(diff[:short]))
        print(
            f"{label:<16}{catch:>7.1f}%{rms_short:>13.2e}{rms_full:>12.2e}"
            f"{max_short:>12.2e}"
        )

    print(
        "\nShort-horizon RMS is the physics-agreement number (pre-chaos); full-"
        "episode RMS grows with the expected chaotic divergence near the top and"
        " is diagnostic only.  Catch% agreement is the distributional check."
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser(
        "run", help="run the controller in one backend, dump trajectories"
    )
    r.add_argument("--backend", choices=["mujoco", "vsim"], default="mujoco")
    r.add_argument("--device", default="cpu")
    r.add_argument("--num_envs", type=int, default=1024, help="perfect square")
    r.add_argument("--t_end", type=float, default=10.0)
    r.add_argument("--out", required=True)
    r.set_defaults(func=run)

    c = sub.add_parser("compare", help="compare dumped trajectories across engines")
    c.add_argument("files", nargs="+")
    c.set_defaults(func=compare)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
