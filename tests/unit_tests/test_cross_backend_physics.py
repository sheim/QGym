"""Cross-backend physics comparison — CPU vs Warp on the same pendulum.

Both backends are initialised from identical initial conditions (same seed)
and stepped in lockstep with zero torque.  The test asserts that dof_pos and
dof_vel trajectories agree to within a tight tolerance at every step.

Run explicitly with ``pytest -m warp`` on a CUDA machine.
"""

import math
import types
import pytest
import torch

from gym import LEGGED_GYM_ROOT_DIR
import os

pytestmark = pytest.mark.warp

PENDULUM_URDF = os.path.join(
    LEGGED_GYM_ROOT_DIR, "resources", "robots", "pendulum", "urdf", "pendulum.urdf"
)

SIM_DT = 0.005
DAMPING = 1.0
N_ENVS = 32
N_STEPS = 2000  # 10 s of simulation


def _make_cfg():
    asset = types.SimpleNamespace(
        file=PENDULUM_URDF,
        joint_damping=DAMPING,
        rotor_inertia=0.0,
        disable_gravity=False,
        penalize_contacts_on=[],
        terminate_after_contacts_on=[],
    )
    sim = types.SimpleNamespace(gravity=[0.0, 0.0, -9.81])
    return types.SimpleNamespace(asset=asset, sim=sim, sim_dt=SIM_DT)


def _random_ics(seed=42):
    torch.manual_seed(seed)
    pos = math.pi + (torch.rand(N_ENVS, 1) - 0.5) * math.pi
    vel = (torch.rand(N_ENVS, 1) - 0.5) * 4.0
    return pos, vel


@pytest.fixture
def cpu_and_warp_backends():
    if not torch.cuda.is_available():
        pytest.fail("Warp tests requested but CUDA is not available", pytrace=False)

    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
    from gym.envs.base.mujoco_warp_backend import MuJocoWarpBackend

    cfg = _make_cfg()

    cpu = MuJocoCPUBackend()
    cpu.setup(cfg, num_envs=N_ENVS, device="cpu", task=None)

    warp = MuJocoWarpBackend()
    warp.setup(cfg, num_envs=N_ENVS, device="cuda:0", task=None)

    yield cpu, warp
    cpu.close()
    warp.close()


def test_trajectories_match(cpu_and_warp_backends):
    """CPU and Warp backends must produce near-identical trajectories."""
    cpu, warp = cpu_and_warp_backends

    pos_ic, vel_ic = _random_ics()

    # Set identical ICs on both backends
    cpu.dof_pos[:] = pos_ic
    cpu.dof_vel[:] = vel_ic
    cpu.reset_dof_state(torch.arange(N_ENVS))

    warp.dof_pos[:] = pos_ic.to(warp.device)
    warp.dof_vel[:] = vel_ic.to(warp.device)
    warp.reset_dof_state(torch.arange(N_ENVS))

    cpu_torques = torch.zeros(N_ENVS, 1)
    warp_torques = torch.zeros(N_ENVS, 1, device=warp.device)

    max_pos_err = 0.0
    max_vel_err = 0.0

    for step in range(N_STEPS):
        cpu.step(cpu_torques)
        warp.step(warp_torques)

        pos_err = (cpu.dof_pos - warp.dof_pos.cpu()).abs().max().item()
        vel_err = (cpu.dof_vel - warp.dof_vel.cpu()).abs().max().item()

        max_pos_err = max(max_pos_err, pos_err)
        max_vel_err = max(max_vel_err, vel_err)

        # Fail early if divergence is large
        assert pos_err < 1e-3, (
            f"Position diverged at step {step}: max |delta| = {pos_err:.6e}"
        )
        assert vel_err < 1e-3, (
            f"Velocity diverged at step {step}: max |delta| = {vel_err:.6e}"
        )

    print(
        f"Cross-backend max errors over {N_STEPS} steps: "
        f"pos={max_pos_err:.2e}, vel={max_vel_err:.2e}"
    )
