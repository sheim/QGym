"""Physics sanity for VSimBackend — damped pendulum convergence.

Fixture-swapped copy of test_mujoco_cpu_physics.py (same physical constants,
same 1e-4 J/step and 0.01 J bounds; see that file's docstring for the
energy convention).  Opt-in: runs only under scripts/run_vsim_tests.sh.
"""

import math
import types
import os

import pytest
import torch

from gym import LEGGED_GYM_ROOT_DIR
from tests.unit_tests.conftest import vsim_guard

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


def _make_cfg() -> types.SimpleNamespace:
    asset = types.SimpleNamespace(
        file=PENDULUM_URDF,
        joint_damping=DAMPING,
        rotor_inertia=0.0,
        disable_gravity=False,
        penalize_contacts_on=[],
        terminate_after_contacts_on=[],
    )
    sim = types.SimpleNamespace(gravity=[0.0, 0.0, -GRAVITY])
    return types.SimpleNamespace(asset=asset, sim=sim, sim_dt=SIM_DT)


def _stable_energy(dof_pos: torch.Tensor, dof_vel: torch.Tensor) -> torch.Tensor:
    INERTIA = MASS * LENGTH**2
    q = dof_pos[:, 0]
    qd = dof_vel[:, 0]
    return 0.5 * INERTIA * qd**2 + MASS * GRAVITY * LENGTH * (1.0 + q.cos())


@pytest.fixture
def damped_vsim_backend():
    vsim_guard()
    from gym.envs.base.vsim_backend import VSimBackend

    b = VSimBackend()
    b.setup(_make_cfg(), num_envs=N_ENVS, device="cuda:0", task=None)
    yield b
    b.close()


def _set_lower_half_ics(b, seed: int = 42):
    torch.manual_seed(seed)
    offsets = (torch.rand(N_ENVS, 1, device=b.device) - 0.5) * math.pi
    b.dof_pos[:] = math.pi + offsets
    b.dof_vel[:] = (torch.rand(N_ENVS, 1, device=b.device) - 0.5) * 4.0
    b.reset_dof_state(torch.arange(N_ENVS, device=b.device))


def test_energy_monotonically_decreases(damped_vsim_backend):
    b = damped_vsim_backend
    _set_lower_half_ics(b)
    torques = torch.zeros(N_ENVS, 1, device=b.device)
    prev_energy = _stable_energy(b.dof_pos, b.dof_vel)
    for step in range(N_STEPS):
        b.step(torques)
        energy = _stable_energy(b.dof_pos, b.dof_vel)
        increase = (energy - prev_energy).clamp(min=0.0).max().item()
        assert increase < 1e-4, (
            f"Energy increased by {increase:.6f} J at step {step} "
            f"(max env energy: {energy.max():.4f} J)"
        )
        prev_energy = energy


def test_all_envs_converge_to_bottom(damped_vsim_backend):
    b = damped_vsim_backend
    _set_lower_half_ics(b)
    torques = torch.zeros(N_ENVS, 1, device=b.device)
    for _ in range(N_STEPS):
        b.step(torques)
    final_energy = _stable_energy(b.dof_pos, b.dof_vel)
    max_residual = final_energy.max().item()
    assert max_residual < 0.01, (
        f"Worst-case residual energy after {N_STEPS} steps: {max_residual:.4f} J "
        f"(expected < 0.01 J). "
        f"Energy per env: {[f'{e:.4f}' for e in final_energy.tolist()]}"
    )
