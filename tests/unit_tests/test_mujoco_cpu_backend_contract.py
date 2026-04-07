"""Run the SimBackend contract suite against MuJocoCPUBackend.

All tests are identical to test_backend_contract.py; the only difference is
the fixture (`mujoco_cpu_backend` instead of `backend`).  This keeps the two
suites in sync while making failures easy to attribute to a specific backend.

Skipped automatically if `mujoco` is not installed.
"""

import torch
import pytest


# ══════════════════════════════════════════════════════════════════════════════
# Tensor shape contract
# ══════════════════════════════════════════════════════════════════════════════


class TestTensorShapes:
    def test_dof_pos_shape(self, mujoco_cpu_backend):
        assert mujoco_cpu_backend.dof_pos.shape == (4, 1)

    def test_dof_vel_shape(self, mujoco_cpu_backend):
        assert mujoco_cpu_backend.dof_vel.shape == (4, 1)

    def test_dof_state_shape(self, mujoco_cpu_backend):
        assert mujoco_cpu_backend.dof_state.shape == (4, 2)

    def test_root_states_shape(self, mujoco_cpu_backend):
        assert mujoco_cpu_backend.root_states.shape == (4, 13)

    def test_contact_forces_shape(self, mujoco_cpu_backend):
        cf = mujoco_cpu_backend.contact_forces
        assert cf.ndim == 3
        assert cf.shape[0] == 4
        assert cf.shape[2] == 3

    def test_shapes_unchanged_after_step(self, mujoco_cpu_backend):
        b = mujoco_cpu_backend
        torques = torch.zeros(4, 1, device=b.device)
        b.step(torques)
        assert b.dof_pos.shape == (4, 1)
        assert b.dof_vel.shape == (4, 1)

    def test_device_is_cpu(self, mujoco_cpu_backend):
        b = mujoco_cpu_backend
        assert b.dof_pos.device.type == "cpu"
        assert b.dof_vel.device.type == "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# Metadata contract
# ══════════════════════════════════════════════════════════════════════════════


class TestMetadata:
    def test_num_dof(self, mujoco_cpu_backend):
        assert mujoco_cpu_backend.num_dof == 1

    def test_num_bodies(self, mujoco_cpu_backend):
        assert mujoco_cpu_backend.num_bodies >= 1

    def test_dof_names_length(self, mujoco_cpu_backend):
        b = mujoco_cpu_backend
        assert len(b.dof_names) == b.num_dof

    def test_body_names_length(self, mujoco_cpu_backend):
        b = mujoco_cpu_backend
        assert len(b.body_names) == b.num_bodies

    def test_find_body_index_known(self, mujoco_cpu_backend):
        idx = mujoco_cpu_backend.find_body_index("pole")
        assert 0 <= idx < mujoco_cpu_backend.num_bodies

    def test_find_body_index_unknown_raises(self, mujoco_cpu_backend):
        with pytest.raises((ValueError, KeyError, IndexError)):
            mujoco_cpu_backend.find_body_index("nonexistent_body_xyz")


# ══════════════════════════════════════════════════════════════════════════════
# Physics / step contract
# ══════════════════════════════════════════════════════════════════════════════


class TestPhysics:
    def test_zero_torque_gravity_accelerates(self, mujoco_cpu_backend):
        b = mujoco_cpu_backend
        b.dof_pos[:] = torch.pi / 2
        b.dof_vel[:] = 0.0
        b.reset_dof_state(torch.arange(4))
        torques = torch.zeros(4, 1, device=b.device)
        b.step(torques)
        assert b.dof_vel.abs().mean() > 1e-6

    def test_upright_equilibrium_near_zero(self, mujoco_cpu_backend):
        b = mujoco_cpu_backend
        b.dof_pos[:] = 0.0
        b.dof_vel[:] = 0.0
        b.reset_dof_state(torch.arange(4))
        torques = torch.zeros(4, 1, device=b.device)
        b.step(torques)
        assert b.dof_vel.abs().mean() < 1e-4

    def test_positive_torque_increases_velocity(self, mujoco_cpu_backend):
        b = mujoco_cpu_backend
        b.dof_pos[:] = 0.0
        b.dof_vel[:] = 0.0
        b.reset_dof_state(torch.arange(4))
        torques = torch.ones(4, 1, device=b.device) * 2.0
        b.step(torques)
        assert b.dof_vel[:, 0].mean() > 0

    def test_environments_evolve_independently(self, mujoco_cpu_backend_16):
        b = mujoco_cpu_backend_16
        b.dof_pos[:8] = 0.0
        b.dof_pos[8:] = torch.pi / 2
        b.dof_vel[:] = 0.0
        b.reset_dof_state(torch.arange(16))
        torques = torch.zeros(16, 1, device=b.device)
        for _ in range(20):
            b.step(torques)
        mean_a = b.dof_pos[:8, 0].mean()
        mean_b = b.dof_pos[8:, 0].mean()
        assert (mean_a - mean_b).abs() > 0.05


# ══════════════════════════════════════════════════════════════════════════════
# Reset contract
# ══════════════════════════════════════════════════════════════════════════════


class TestReset:
    def test_reset_writes_pos(self, mujoco_cpu_backend):
        b = mujoco_cpu_backend
        target = 1.23
        b.dof_pos[:] = target
        b.reset_dof_state(torch.arange(4))
        assert torch.allclose(
            b.dof_pos, torch.full((4, 1), target, device=b.device), atol=1e-5
        )

    def test_reset_writes_vel(self, mujoco_cpu_backend):
        b = mujoco_cpu_backend
        b.dof_vel[:] = 4.56
        b.reset_dof_state(torch.arange(4))
        assert torch.allclose(
            b.dof_vel, torch.full((4, 1), 4.56, device=b.device), atol=1e-5
        )

    def test_partial_reset(self, mujoco_cpu_backend):
        b = mujoco_cpu_backend
        b.dof_pos[:] = 0.0
        b.dof_vel[:] = 0.0
        b.reset_dof_state(torch.arange(4))

        torques = torch.zeros(4, 1, device=b.device)
        for _ in range(10):
            b.step(torques)
        pos_after = b.dof_pos.clone()

        b.dof_pos[:2] = 99.0
        b.dof_vel[:2] = 0.0
        b.reset_dof_state(torch.tensor([0, 1]))

        assert torch.allclose(b.dof_pos[:2], torch.full((2, 1), 99.0), atol=1e-5)
        assert torch.allclose(b.dof_pos[2:], pos_after[2:], atol=1e-5)

    def test_dof_state_view_consistent_with_dof_pos(self, mujoco_cpu_backend):
        b = mujoco_cpu_backend
        b.dof_pos[:] = 2.71
        assert torch.allclose(
            b.dof_state[:, 0],
            torch.full((4,), 2.71, device=b.device),
            atol=1e-5,
        )
