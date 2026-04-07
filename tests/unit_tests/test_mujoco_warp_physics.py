"""Physics sanity test for MuJocoWarpBackend — damped pendulum convergence.

Identical physics assertions to test_mujoco_cpu_physics.py; runs on the
MuJocoWarpBackend.  Skipped if mujoco_warp is not installed or CUDA is absent.

See test_mujoco_cpu_physics.py for the full coordinate-convention explanation.
"""

import math
import types
import pytest
import torch

from gym import LEGGED_GYM_ROOT_DIR
import os

PENDULUM_URDF = os.path.join(
    LEGGED_GYM_ROOT_DIR, "resources", "robots", "pendulum", "urdf", "pendulum.urdf"
)

MASS = 1.0
LENGTH = 1.0
GRAVITY = 9.81

SIM_DT = 0.005
DAMPING = 1.0
N_ENVS = 32
N_STEPS = 4000


def _make_cfg(
    damping: float = DAMPING, sim_dt: float = SIM_DT
) -> types.SimpleNamespace:
    asset = types.SimpleNamespace(
        file=PENDULUM_URDF,
        joint_damping=damping,
        rotor_inertia=0.0,
        disable_gravity=False,
        penalize_contacts_on=[],
        terminate_after_contacts_on=[],
    )
    sim = types.SimpleNamespace(gravity=[0.0, 0.0, -GRAVITY])
    return types.SimpleNamespace(asset=asset, sim=sim, sim_dt=sim_dt)


def _stable_energy(dof_pos: torch.Tensor, dof_vel: torch.Tensor) -> torch.Tensor:
    INERTIA = MASS * LENGTH**2
    q = dof_pos[:, 0]
    qd = dof_vel[:, 0]
    KE = 0.5 * INERTIA * qd**2
    PE = MASS * GRAVITY * LENGTH * (1.0 + q.cos())
    return KE + PE


@pytest.fixture
def damped_warp_backend():
    pytest.importorskip("mujoco_warp")
    from gym.envs.base.mujoco_warp_backend import MuJocoWarpBackend

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for MuJocoWarpBackend physics test")

    b = MuJocoWarpBackend()
    b.setup(_make_cfg(), num_envs=N_ENVS, device="cuda:0", task=None)
    return b


def _set_lower_half_ics(b, seed: int = 42):
    torch.manual_seed(seed)
    offsets = (torch.rand(N_ENVS, 1) - 0.5) * math.pi
    b.dof_pos[:] = math.pi + offsets
    b.dof_vel[:] = (torch.rand(N_ENVS, 1) - 0.5) * 4.0
    b.reset_dof_state(torch.arange(N_ENVS))


def test_energy_monotonically_decreases(damped_warp_backend):
    """Total mechanical energy must never increase under damped free motion."""
    b = damped_warp_backend
    _set_lower_half_ics(b)

    torques = torch.zeros(N_ENVS, 1, device=b.device)
    prev_energy = _stable_energy(b.dof_pos.cpu(), b.dof_vel.cpu())

    for step in range(N_STEPS):
        b.step(torques)
        energy = _stable_energy(b.dof_pos.cpu(), b.dof_vel.cpu())
        increase = (energy - prev_energy).clamp(min=0.0).max().item()
        assert increase < 1e-4, (
            f"Energy increased by {increase:.6f} J at step {step} "
            f"(max env energy: {energy.max():.4f} J)"
        )
        prev_energy = energy


def test_all_envs_converge_to_bottom(damped_warp_backend):
    """All environments must reach near-zero energy (stable downward equilibrium)."""
    b = damped_warp_backend
    _set_lower_half_ics(b)

    torques = torch.zeros(N_ENVS, 1, device=b.device)
    for _ in range(N_STEPS):
        b.step(torques)

    final_energy = _stable_energy(b.dof_pos.cpu(), b.dof_vel.cpu())
    max_residual = final_energy.max().item()

    assert max_residual < 0.01, (
        f"Worst-case residual energy after {N_STEPS} steps: {max_residual:.4f} J "
        f"(expected < 0.01 J). "
        f"Energy per env: {[f'{e:.4f}' for e in final_energy.tolist()]}"
    )
