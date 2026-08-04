"""Shared executable contract for every supported physics backend.

The test body is intentionally backend-neutral. Pytest parameter marks select
MuJoCo CPU in the default gate, MuJoCo Warp with ``-m warp``, and licensed
VSim with ``-m vsim``.
"""

import pytest
import torch


def _env_ids(backend, count: int) -> torch.Tensor:
    return torch.arange(count, device=backend.device)


def test_public_state_schema_and_metadata(pendulum_backend):
    backend = pendulum_backend

    assert backend.num_dof == 1
    assert backend.num_bodies >= 1
    assert backend.dof_pos.shape == (4, 1)
    assert backend.dof_vel.shape == (4, 1)
    assert backend.dof_state.shape == (4, 2)
    assert backend.root_states.shape == (4, 13)
    assert backend.contact_forces.shape == (4, backend.num_bodies, 3)
    assert backend.dof_pos.device == backend.dof_vel.device
    assert backend.dof_pos.device.type == backend.device.split(":")[0]
    assert tuple(backend.dof_names) == backend.robot_layout.dof_names
    assert tuple(backend.body_names) == backend.robot_layout.body_names
    assert len(backend.dof_names) == backend.num_dof
    assert len(backend.body_names) == backend.num_bodies

    pole_index = backend.find_body_index("pole")
    assert 0 <= pole_index < backend.num_bodies
    with pytest.raises((ValueError, KeyError, IndexError)):
        backend.find_body_index("nonexistent_body_xyz")


def test_step_preserves_public_tensor_identity(pendulum_backend):
    backend = pendulum_backend
    state_tensors = (
        backend.dof_state,
        backend.dof_pos,
        backend.dof_vel,
        backend.root_states,
        backend.contact_forces,
    )
    pointers = tuple(tensor.data_ptr() for tensor in state_tensors)

    backend.step(torch.zeros(4, 1, device=backend.device))

    refreshed = (
        backend.dof_state,
        backend.dof_pos,
        backend.dof_vel,
        backend.root_states,
        backend.contact_forces,
    )
    assert tuple(tensor.data_ptr() for tensor in refreshed) == pointers


def test_gravity_and_applied_torque_have_expected_signs(pendulum_backend):
    backend = pendulum_backend
    env_ids = _env_ids(backend, 4)
    zero_torques = torch.zeros(4, 1, device=backend.device)

    backend.dof_pos[:] = torch.pi / 2
    backend.dof_vel.zero_()
    backend.reset_dof_state(env_ids)
    backend.step(zero_torques)
    assert backend.dof_vel.abs().mean() > 1e-6

    backend.dof_pos.zero_()
    backend.dof_vel.zero_()
    backend.reset_dof_state(env_ids)
    backend.step(torch.full((4, 1), 2.0, device=backend.device))
    assert backend.dof_vel[:, 0].mean() > 0


def test_environments_evolve_independently(pendulum_backend_16):
    backend = pendulum_backend_16
    backend.dof_pos[:8] = 0.0
    backend.dof_pos[8:] = torch.pi / 2
    backend.dof_vel.zero_()
    backend.reset_dof_state(_env_ids(backend, 16))

    torques = torch.zeros(16, 1, device=backend.device)
    for _ in range(20):
        backend.step(torques)

    separation = (backend.dof_pos[:8, 0].mean() - backend.dof_pos[8:, 0].mean()).abs()
    assert separation > 0.05


def test_full_and_partial_dof_resets_round_trip(pendulum_backend):
    backend = pendulum_backend
    env_ids = _env_ids(backend, 4)

    backend.dof_pos[:] = 1.23
    backend.dof_vel[:] = 4.56
    backend.reset_dof_state(env_ids)
    torch.testing.assert_close(
        backend.dof_pos,
        torch.full_like(backend.dof_pos, 1.23),
        atol=1e-5,
        rtol=0,
    )
    torch.testing.assert_close(
        backend.dof_vel,
        torch.full_like(backend.dof_vel, 4.56),
        atol=1e-5,
        rtol=0,
    )

    backend.step(torch.zeros(4, 1, device=backend.device))
    untouched = backend.dof_pos[2:].clone()
    backend.dof_pos[:2] = 0.25
    backend.dof_vel[:2] = 0.0
    backend.reset_dof_state(torch.tensor([0, 1], device=backend.device))

    torch.testing.assert_close(
        backend.dof_pos[:2],
        torch.full_like(backend.dof_pos[:2], 0.25),
        atol=1e-5,
        rtol=0,
    )
    torch.testing.assert_close(backend.dof_pos[2:], untouched, atol=1e-5, rtol=0)


def test_dof_state_is_synchronized_after_reset(pendulum_backend):
    backend = pendulum_backend
    backend.dof_pos[:] = 2.71
    backend.dof_vel[:] = -0.5
    backend.reset_dof_state(_env_ids(backend, 4))

    state = backend.dof_state.view(4, 1, 2)
    torch.testing.assert_close(state[..., 0], backend.dof_pos)
    torch.testing.assert_close(state[..., 1], backend.dof_vel)
