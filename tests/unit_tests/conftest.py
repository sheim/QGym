"""Shared fixtures for unit tests.

No IsaacGym / MuJoCo / Warp dependency — these tests must run on any
machine, including Mac CI without a GPU.
"""

import pytest
import torch

from tests.unit_tests.mock_backend import MockBackend


# ── Device parametrisation ──────────────────────────────────────────────────


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
