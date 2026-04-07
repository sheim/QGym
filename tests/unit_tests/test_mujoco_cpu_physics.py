"""Physics sanity test for MuJocoCPUBackend — damped pendulum convergence.

Pendulum URDF coordinate convention
------------------------------------
Joint axis is Y, CoM is at z=+1 in the body frame (above the joint).
With gravity in -Z:
  - theta = 0   → pole pointing UP   → UNSTABLE equilibrium
  - theta = pi  → pole pointing DOWN → STABLE equilibrium

Energy is measured relative to the stable equilibrium (theta = pi):

  E = 0.5 * INERTIA * qdot^2 + m*g*L * (1 + cos(theta))

  E = 0  at theta=pi, qdot=0   (minimum, stable)
  E = 2*m*g*L at theta=0, qdot=0  (maximum, unstable)

With nonzero damping and zero applied torque, starting from any initial
condition that is not exactly on the unstable equilibrium, the system must:

  1. Dissipate energy monotonically (E never increases).
  2. Converge to the stable downward equilibrium (E → 0).

To avoid degenerate cases where a pendulum starts exactly at or near the
unstable equilibrium, initial positions are drawn from [pi/2, 3*pi/2]
(the "lower half"), which are all gravitationally attracted toward theta=pi.

Skipped if `mujoco` is not installed.
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

# Physical parameters (must match the URDF inertial block)
MASS = 1.0  # kg
LENGTH = 1.0  # m  (CoM at z=1 in body frame per URDF <origin xyz="0 0 1.0"/>)
GRAVITY = 9.81  # m/s²

# Simulation parameters
SIM_DT = 0.005  # s  (200 Hz, matching PendulumCfg)
DAMPING = 1.0  # joint damping — sufficient for clear convergence in 20 s
N_ENVS = 32  # number of parallel environments (random ICs)
N_STEPS = 4000  # 20 s of simulation at 200 Hz


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
    """Mechanical energy per environment measured from the stable equilibrium.

    E = 0.5 * INERTIA * qdot^2 + m*g*L * (1 + cos(theta))

    E = 0  when theta=pi (downward, stable), qdot=0.
    E > 0  for all other states.
    """
    INERTIA = MASS * LENGTH**2
    q = dof_pos[:, 0]
    qd = dof_vel[:, 0]
    KE = 0.5 * INERTIA * qd**2
    PE = MASS * GRAVITY * LENGTH * (1.0 + q.cos())
    return KE + PE


@pytest.fixture
def damped_backend():
    pytest.importorskip("mujoco")
    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend

    b = MuJocoCPUBackend()
    b.setup(_make_cfg(), num_envs=N_ENVS, device="cpu", task=None)
    return b


def _set_lower_half_ics(b, seed: int = 42):
    """Set ICs in theta ∈ [pi/2, 3*pi/2], i.e. the lower half of the circle.

    All positions are gravitationally attracted toward theta=pi (downward),
    so none of the environments are near the unstable equilibrium at theta=0.
    """
    torch.manual_seed(seed)
    # theta = pi + U(-pi/2, pi/2)  → [pi/2, 3pi/2]
    offsets = (torch.rand(N_ENVS, 1) - 0.5) * math.pi
    b.dof_pos[:] = math.pi + offsets
    b.dof_vel[:] = (torch.rand(N_ENVS, 1) - 0.5) * 4.0  # qdot ∈ [-2, 2]
    b.reset_dof_state(torch.arange(N_ENVS))


def test_energy_monotonically_decreases(damped_backend):
    """Total mechanical energy must never increase under damped free motion."""
    b = damped_backend
    _set_lower_half_ics(b)

    torques = torch.zeros(N_ENVS, 1)
    prev_energy = _stable_energy(b.dof_pos, b.dof_vel)

    for step in range(N_STEPS):
        b.step(torques)
        energy = _stable_energy(b.dof_pos, b.dof_vel)
        # Numerical integration can add a tiny amount per step; allow 1e-4 J tolerance
        increase = (energy - prev_energy).clamp(min=0.0).max().item()
        assert increase < 1e-4, (
            f"Energy increased by {increase:.6f} J at step {step} "
            f"(max env energy: {energy.max():.4f} J)"
        )
        prev_energy = energy


def test_all_envs_converge_to_bottom(damped_backend):
    """All environments must reach near-zero energy (stable downward equilibrium)."""
    b = damped_backend
    _set_lower_half_ics(b)

    torques = torch.zeros(N_ENVS, 1)
    for _ in range(N_STEPS):
        b.step(torques)

    final_energy = _stable_energy(b.dof_pos, b.dof_vel)
    max_residual = final_energy.max().item()

    # After 20 s with damping=1.0 all envs should be within 0.01 J of equilibrium
    assert max_residual < 0.01, (
        f"Worst-case residual energy after {N_STEPS} steps: {max_residual:.4f} J "
        f"(expected < 0.01 J). "
        f"Energy per env: {[f'{e:.4f}' for e in final_energy.tolist()]}"
    )
