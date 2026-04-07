"""Shared fixtures for unit tests.

No IsaacGym dependency — these tests must run on any machine, including Mac
CI without a GPU.  MuJoCo fixtures are skipped if mujoco is not installed.
"""

import types
import pytest
import torch

from tests.unit_tests.mock_backend import MockBackend

# Path to the pendulum URDF used by MuJoCo backend fixtures
import os
from gym import LEGGED_GYM_ROOT_DIR

PENDULUM_URDF = os.path.join(
    LEGGED_GYM_ROOT_DIR, "resources", "robots", "pendulum", "urdf", "pendulum.urdf"
)


def _make_pendulum_cfg(sim_dt: float = 0.005) -> types.SimpleNamespace:
    """Minimal cfg-like object for the pendulum, no task registry needed."""
    asset = types.SimpleNamespace(
        file=PENDULUM_URDF,
        joint_damping=0.1,
        rotor_inertia=0.0,
        disable_gravity=False,
        penalize_contacts_on=[],
        terminate_after_contacts_on=[],
    )
    sim = types.SimpleNamespace(gravity=[0.0, 0.0, -9.81])
    return types.SimpleNamespace(asset=asset, sim=sim, sim_dt=sim_dt)


# ── Device parametrisation ───────────────────────────────────────────────────


def _available_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda:0")
    return devices


@pytest.fixture(params=_available_devices())
def device(request):
    return request.param


# ── MockBackend fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def backend(device):
    """4-env pendulum backend on the parametrised device."""
    return MockBackend(num_envs=4, device=device)


@pytest.fixture
def backend_16(device):
    """16-env pendulum backend — used for independence / reset tests."""
    return MockBackend(num_envs=16, device=device)


# ── MuJocoCPUBackend fixtures ────────────────────────────────────────────────


@pytest.fixture
def mujoco_cpu_backend(device):
    """4-env pendulum on MuJocoCPUBackend (CPU only; skipped if mujoco absent)."""
    mujoco = pytest.importorskip("mujoco")  # noqa: F841
    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend

    if device != "cpu":
        pytest.skip("MuJocoCPUBackend only supports CPU")

    b = MuJocoCPUBackend()
    b.setup(_make_pendulum_cfg(), num_envs=4, device=device, task=None)
    return b


@pytest.fixture
def mujoco_cpu_backend_16(device):
    """16-env pendulum on MuJocoCPUBackend."""
    pytest.importorskip("mujoco")
    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend

    if device != "cpu":
        pytest.skip("MuJocoCPUBackend only supports CPU")

    b = MuJocoCPUBackend()
    b.setup(_make_pendulum_cfg(), num_envs=16, device=device, task=None)
    return b


# ── MuJocoWarpBackend fixtures ───────────────────────────────────────────────


@pytest.fixture
def mujoco_warp_backend(device):
    """4-env pendulum on MuJocoWarpBackend (skipped if mujoco_warp absent)."""
    pytest.importorskip("mujoco_warp")
    from gym.envs.base.mujoco_warp_backend import MuJocoWarpBackend

    if not torch.cuda.is_available() and device == "cuda:0":
        pytest.skip("CUDA not available")

    b = MuJocoWarpBackend()
    # Warp defaults to cuda:0 when available; fall back to cpu
    warp_device = device if torch.cuda.is_available() else "cpu"
    b.setup(_make_pendulum_cfg(), num_envs=4, device=warp_device, task=None)
    return b


@pytest.fixture
def mujoco_warp_backend_16(device):
    """16-env pendulum on MuJocoWarpBackend."""
    pytest.importorskip("mujoco_warp")
    from gym.envs.base.mujoco_warp_backend import MuJocoWarpBackend

    if not torch.cuda.is_available() and device == "cuda:0":
        pytest.skip("CUDA not available")

    b = MuJocoWarpBackend()
    warp_device = device if torch.cuda.is_available() else "cpu"
    b.setup(_make_pendulum_cfg(), num_envs=16, device=warp_device, task=None)
    return b
