"""Shared fixtures for the portable and explicit backend test groups."""

import os
import types

import pytest
import torch

from tests.unit_tests.mock_backend import MockBackend

# Path to the pendulum URDF used by MuJoCo backend fixtures
from gym import GYM_ROOT_DIR

PENDULUM_URDF = os.path.join(
    GYM_ROOT_DIR, "resources", "robots", "pendulum", "urdf", "pendulum.urdf"
)
MINI_CHEETAH_URDF = os.path.join(
    GYM_ROOT_DIR,
    "resources",
    "robots",
    "mini_cheetah",
    "urdf",
    "mini_cheetah_simple.urdf",
)


def vsim_guard():
    """Fail clearly when the explicitly requested VSim group cannot run.

    Run via scripts/run_vsim_tests.sh (sets Q2_VSIM_TESTS, LD_LIBRARY_PATH,
    VL_WORKING_DIRECTORY).
    """
    if os.environ.get("Q2_VSIM_TESTS") != "1":
        pytest.fail(
            "VSim tests must be launched with scripts/run_vsim_tests.sh",
            pytrace=False,
        )
    try:
        __import__("vlearn")
    except ImportError:
        pytest.fail("VSim tests requested but vlearn is not installed", pytrace=False)
    if not torch.cuda.is_available():
        pytest.fail("VSim tests requested but CUDA is not available", pytrace=False)


def _make_vsim_backend(cfg, num_envs: int):
    from gym.envs.base.vsim_backend import VSimBackend

    b = VSimBackend()
    b.setup(cfg, num_envs=num_envs, device="cuda:0", task=None)
    return b


@pytest.fixture
def vsim_backend():
    """4-env pendulum on VSimBackend (opt-in; see vsim_guard)."""
    vsim_guard()
    b = _make_vsim_backend(_make_pendulum_cfg(), 4)
    yield b
    b.close()


@pytest.fixture
def vsim_backend_16():
    vsim_guard()
    b = _make_vsim_backend(_make_pendulum_cfg(), 16)
    yield b
    b.close()


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


# ── MockBackend fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def mock_backend():
    return MockBackend(num_envs=4, device="cpu")


# ── MuJocoCPUBackend fixtures ────────────────────────────────────────────────


@pytest.fixture
def mujoco_cpu_backend():
    """Four-environment pendulum on the mandatory MuJoCo CPU backend."""
    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend

    b = MuJocoCPUBackend()
    b.setup(_make_pendulum_cfg(), num_envs=4, device="cpu", task=None)
    yield b
    b.close()


@pytest.fixture
def mujoco_cpu_backend_16():
    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend

    b = MuJocoCPUBackend()
    b.setup(_make_pendulum_cfg(), num_envs=16, device="cpu", task=None)
    yield b
    b.close()


# ── MuJocoWarpBackend fixtures ───────────────────────────────────────────────


@pytest.fixture
def mujoco_warp_backend():
    """Four-environment pendulum on the explicitly requested CUDA backend."""
    if not torch.cuda.is_available():
        pytest.fail("Warp tests requested but CUDA is not available", pytrace=False)
    try:
        __import__("mujoco_warp")
    except ImportError:
        pytest.fail(
            "Warp tests requested but mujoco-warp is not installed", pytrace=False
        )
    from gym.envs.base.mujoco_warp_backend import MuJocoWarpBackend

    b = MuJocoWarpBackend()
    b.setup(_make_pendulum_cfg(), num_envs=4, device="cuda:0", task=None)
    yield b
    b.close()


@pytest.fixture
def mujoco_warp_backend_16():
    if not torch.cuda.is_available():
        pytest.fail("Warp tests requested but CUDA is not available", pytrace=False)
    try:
        __import__("mujoco_warp")
    except ImportError:
        pytest.fail(
            "Warp tests requested but mujoco-warp is not installed", pytrace=False
        )
    from gym.envs.base.mujoco_warp_backend import MuJocoWarpBackend

    b = MuJocoWarpBackend()
    b.setup(_make_pendulum_cfg(), num_envs=16, device="cuda:0", task=None)
    yield b
    b.close()


PENDULUM_BACKENDS = [
    pytest.param("mujoco_cpu_backend", id="mujoco-cpu"),
    pytest.param(
        "mujoco_warp_backend",
        id="mujoco-warp",
        marks=pytest.mark.warp,
    ),
    pytest.param("vsim_backend", id="vsim", marks=pytest.mark.vsim),
]

PENDULUM_BACKENDS_16 = [
    pytest.param("mujoco_cpu_backend_16", id="mujoco-cpu"),
    pytest.param(
        "mujoco_warp_backend_16",
        id="mujoco-warp",
        marks=pytest.mark.warp,
    ),
    pytest.param("vsim_backend_16", id="vsim", marks=pytest.mark.vsim),
]


@pytest.fixture(params=PENDULUM_BACKENDS)
def pendulum_backend(request):
    """A real backend selected by the test execution group."""
    return request.getfixturevalue(request.param)


@pytest.fixture(params=PENDULUM_BACKENDS_16)
def pendulum_backend_16(request):
    return request.getfixturevalue(request.param)


# ── Floating-base (mini_cheetah) fixtures ───────────────────────────────────


def _make_mini_cheetah_cfg(sim_dt: float = 0.002) -> types.SimpleNamespace:
    """Minimal cfg for mini_cheetah (floating-base, 12 DOFs)."""
    asset = types.SimpleNamespace(
        file=MINI_CHEETAH_URDF,
        joint_damping=0.01,
        rotor_inertia=0.0,
        disable_gravity=False,
        fix_base_link=False,
        penalize_contacts_on=["thigh"],
        terminate_after_contacts_on=["base"],
        foot_name="foot",
    )
    terrain = types.SimpleNamespace(
        mesh_type="plane",
        static_friction=1.0,
        dynamic_friction=1.0,
    )
    sim = types.SimpleNamespace(gravity=[0.0, 0.0, -9.81])
    return types.SimpleNamespace(asset=asset, sim=sim, terrain=terrain, sim_dt=sim_dt)


@pytest.fixture
def legged_vsim_backend():
    """4-env mini_cheetah on VSimBackend (opt-in; see vsim_guard)."""
    vsim_guard()
    from gym.envs.base.vsim_backend import VSimBackend

    cfg = _make_mini_cheetah_cfg()
    # Spawn ABOVE leg length: at qpos=0 the legs are fully extended, and a
    # penetrating spawn gets a maxDepenetrationVelocity kick (10 m/s) that
    # launches the robot ballistically.  Tasks reset to crouched poses before
    # stepping; this raw fixture has no task, so it must spawn clear.
    cfg.init_state = types.SimpleNamespace(pos=[0.0, 0.0, 0.5], rot=[0, 0, 0, 1])
    b = VSimBackend()
    b.setup(cfg, num_envs=4, device="cuda:0", task=None)
    yield b
    b.close()


@pytest.fixture
def legged_cpu_backend():
    """4-env mini_cheetah on MuJocoCPUBackend (floating-base)."""
    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend

    b = MuJocoCPUBackend()
    b.setup(_make_mini_cheetah_cfg(), num_envs=4, device="cpu", task=None)
    yield b
    b.close()


@pytest.fixture
def legged_warp_backend():
    """4-env mini_cheetah on MuJocoWarpBackend (floating-base)."""
    if not torch.cuda.is_available():
        pytest.fail("Warp tests requested but CUDA is not available", pytrace=False)
    try:
        __import__("mujoco_warp")
    except ImportError:
        pytest.fail(
            "Warp tests requested but mujoco-warp is not installed", pytrace=False
        )
    from gym.envs.base.mujoco_warp_backend import MuJocoWarpBackend

    b = MuJocoWarpBackend()
    b.setup(_make_mini_cheetah_cfg(), num_envs=4, device="cuda:0", task=None)
    yield b
    b.close()
