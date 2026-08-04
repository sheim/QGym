"""Cross-ENGINE parity: VSimBackend vs analytic solution and MuJoCo CPU.

Per-step lockstep comparison is impossible across engines (different
integrators/solvers), so parity is established with physics metrics whose
bounds are stated here and never silently widened:

1. Small-oscillation pendulum period vs the analytic value
   T = 2*pi*sqrt(I/(m*g*L)) — within 5% (integrator-agnostic to O(dt)).
2. Damped-pendulum energy envelope vs MuJocoCPUBackend from IDENTICAL
   initial conditions — relative deviation < 20% while the reference
   envelope is above 0.05 J (the late tail is dominated by tiny residuals).

Opt-in: runs only under scripts/run_vsim_tests.sh (license + CUDA).
"""

import math
import types
import os

import pytest
import torch

from gym import LEGGED_GYM_ROOT_DIR
from tests.unit_tests.conftest import vsim_guard

pytestmark = pytest.mark.vsim

PENDULUM_URDF = os.path.join(
    LEGGED_GYM_ROOT_DIR, "resources", "robots", "pendulum", "urdf", "pendulum.urdf"
)

MASS, LENGTH, GRAVITY = 1.0, 1.0, 9.81
SIM_DT = 0.005


def _make_cfg(damping: float) -> types.SimpleNamespace:
    asset = types.SimpleNamespace(
        file=PENDULUM_URDF,
        joint_damping=damping,
        rotor_inertia=0.0,
        disable_gravity=False,
        penalize_contacts_on=[],
        terminate_after_contacts_on=[],
    )
    sim = types.SimpleNamespace(gravity=[0.0, 0.0, -GRAVITY])
    return types.SimpleNamespace(asset=asset, sim=sim, sim_dt=SIM_DT)


def _energy(dof_pos, dof_vel):
    q, qd = dof_pos[:, 0], dof_vel[:, 0]
    inertia = MASS * LENGTH**2
    return 0.5 * inertia * qd**2 + MASS * GRAVITY * LENGTH * (1.0 + q.cos())


def test_small_oscillation_period_matches_analytic():
    """Undamped pendulum, 0.1 rad amplitude around the stable equilibrium."""
    vsim_guard()
    from gym.envs.base.vsim_backend import VSimBackend

    b = VSimBackend()
    b.setup(_make_cfg(damping=0.0), num_envs=4, device="cuda:0", task=None)
    try:
        b.dof_pos[:] = math.pi + 0.1
        b.dof_vel[:] = 0.0
        b.reset_dof_state(torch.arange(4, device=b.device))

        torques = torch.zeros(4, 1, device=b.device)
        crossings = []
        prev = b.dof_pos[0, 0].item() - math.pi
        for step in range(1, 4000):
            b.step(torques)
            cur = b.dof_pos[0, 0].item() - math.pi
            if prev > 0.0 >= cur:  # downward zero crossing = once per period
                crossings.append(step * SIM_DT)
                if len(crossings) == 4:
                    break
            prev = cur
        assert len(crossings) >= 4, "pendulum did not oscillate"
        periods = [b - a for a, b in zip(crossings, crossings[1:])]
        t_measured = sum(periods) / len(periods)
        t_analytic = (
            2 * math.pi * math.sqrt((MASS * LENGTH**2) / (MASS * GRAVITY * LENGTH))
        )
        rel = abs(t_measured - t_analytic) / t_analytic
        assert rel < 0.05, (
            f"period {t_measured:.4f}s vs analytic {t_analytic:.4f}s "
            f"({rel:.1%} off, bound 5%)"
        )
    finally:
        b.close()


def test_damped_envelope_matches_mujoco_cpu():
    """Same seeded ICs on both engines; energy envelopes must agree ±20%."""
    vsim_guard()
    pytest.importorskip("mujoco")
    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
    from gym.envs.base.vsim_backend import VSimBackend

    n_envs, n_steps, sample = 8, 3000, 200
    torch.manual_seed(42)
    q0 = math.pi + (torch.rand(n_envs, 1) - 0.5) * math.pi
    qd0 = (torch.rand(n_envs, 1) - 0.5) * 4.0

    def run(backend, device):
        backend.setup(_make_cfg(damping=1.0), n_envs, device, task=None)
        backend.dof_pos[:] = q0.to(device)
        backend.dof_vel[:] = qd0.to(device)
        backend.reset_dof_state(torch.arange(n_envs, device=device))
        torques = torch.zeros(n_envs, 1, device=device)
        env_curve = []
        for step in range(1, n_steps + 1):
            backend.step(torques)
            if step % sample == 0:
                env_curve.append(_energy(backend.dof_pos, backend.dof_vel).max().item())
        return env_curve

    mj_curve = run(MuJocoCPUBackend(), "cpu")
    vs = VSimBackend()
    try:
        vs_curve = run(vs, "cuda:0")
    finally:
        vs.close()

    for i, (em, ev) in enumerate(zip(mj_curve, vs_curve)):
        if em < 0.05:
            break  # tail regime — covered by the convergence test
        rel = abs(ev - em) / em
        assert rel < 0.20, (
            f"energy envelope diverged at sample {i}: mujoco {em:.4f} J vs "
            f"vsim {ev:.4f} J ({rel:.1%}, bound 20%)"
        )
