"""Run the SimBackend contract suite against VSimBackend.

Fixture-swapped copy of test_mujoco_cpu_backend_contract.py (the house
pattern: identical assertions per backend, failures attribute cleanly).
Differences from the CPU file: device is cuda, env_ids tensors are created
on the backend device, and num_bodies has no `world` entry (vsim counts
links only — pendulum: base + pole = 2).

Opt-in: runs only under scripts/run_vsim_tests.sh (license + CUDA).
"""

import torch
import pytest


def _ids(b, n):
    return torch.arange(n, device=b.device)


class TestTensorShapes:
    def test_dof_pos_shape(self, vsim_backend):
        assert vsim_backend.dof_pos.shape == (4, 1)

    def test_dof_vel_shape(self, vsim_backend):
        assert vsim_backend.dof_vel.shape == (4, 1)

    def test_dof_state_shape(self, vsim_backend):
        assert vsim_backend.dof_state.shape == (4, 2)

    def test_root_states_shape(self, vsim_backend):
        assert vsim_backend.root_states.shape == (4, 13)

    def test_contact_forces_shape(self, vsim_backend):
        cf = vsim_backend.contact_forces
        assert cf.ndim == 3
        assert cf.shape[0] == 4
        assert cf.shape[2] == 3

    def test_rigid_body_states_shape(self, vsim_backend):
        b = vsim_backend
        assert b.rigid_body_states.shape == (4 * b.num_bodies, 13)

    def test_shapes_unchanged_after_step(self, vsim_backend):
        b = vsim_backend
        b.step(torch.zeros(4, 1, device=b.device))
        assert b.dof_pos.shape == (4, 1)
        assert b.dof_vel.shape == (4, 1)

    def test_device_is_cuda(self, vsim_backend):
        b = vsim_backend
        assert b.dof_pos.device.type == "cuda"
        assert b.root_states.device.type == "cuda"


class TestMetadata:
    def test_num_dof(self, vsim_backend):
        assert vsim_backend.num_dof == 1

    def test_num_bodies(self, vsim_backend):
        assert vsim_backend.num_bodies == 2  # base + pole (no `world` in vsim)

    def test_dof_names_length(self, vsim_backend):
        b = vsim_backend
        assert len(b.dof_names) == b.num_dof

    def test_body_names_length(self, vsim_backend):
        b = vsim_backend
        assert len(b.body_names) == b.num_bodies

    def test_find_body_index_known(self, vsim_backend):
        idx = vsim_backend.find_body_index("pole")
        assert 0 <= idx < vsim_backend.num_bodies

    def test_find_body_index_unknown_raises(self, vsim_backend):
        with pytest.raises((ValueError, KeyError, IndexError)):
            vsim_backend.find_body_index("nonexistent_body_xyz")


class TestPhysics:
    def test_zero_torque_gravity_accelerates(self, vsim_backend):
        b = vsim_backend
        b.dof_pos[:] = torch.pi / 2
        b.dof_vel[:] = 0.0
        b.reset_dof_state(_ids(b, 4))
        b.step(torch.zeros(4, 1, device=b.device))
        assert b.dof_vel.abs().mean() > 1e-6

    def test_upright_equilibrium_near_zero(self, vsim_backend):
        b = vsim_backend
        b.dof_pos[:] = 0.0
        b.dof_vel[:] = 0.0
        b.reset_dof_state(_ids(b, 4))
        b.step(torch.zeros(4, 1, device=b.device))
        assert b.dof_vel.abs().mean() < 1e-4

    def test_positive_torque_increases_velocity(self, vsim_backend):
        b = vsim_backend
        b.dof_pos[:] = 0.0
        b.dof_vel[:] = 0.0
        b.reset_dof_state(_ids(b, 4))
        b.step(torch.ones(4, 1, device=b.device) * 2.0)
        assert b.dof_vel[:, 0].mean() > 0

    def test_environments_evolve_independently(self, vsim_backend_16):
        b = vsim_backend_16
        b.dof_pos[:8] = 0.0
        b.dof_pos[8:] = torch.pi / 2
        b.dof_vel[:] = 0.0
        b.reset_dof_state(_ids(b, 16))
        torques = torch.zeros(16, 1, device=b.device)
        for _ in range(20):
            b.step(torques)
        mean_a = b.dof_pos[:8, 0].mean()
        mean_b = b.dof_pos[8:, 0].mean()
        assert (mean_a - mean_b).abs() > 0.05


class TestReset:
    def test_reset_writes_pos(self, vsim_backend):
        b = vsim_backend
        b.dof_pos[:] = 1.23
        b.reset_dof_state(_ids(b, 4))
        assert torch.allclose(
            b.dof_pos, torch.full((4, 1), 1.23, device=b.device), atol=1e-5
        )

    def test_reset_writes_vel(self, vsim_backend):
        b = vsim_backend
        b.dof_vel[:] = 4.56
        b.reset_dof_state(_ids(b, 4))
        assert torch.allclose(
            b.dof_vel, torch.full((4, 1), 4.56, device=b.device), atol=1e-5
        )

    def test_partial_reset(self, vsim_backend):
        b = vsim_backend
        b.dof_pos[:] = 0.0
        b.dof_vel[:] = 0.0
        b.reset_dof_state(_ids(b, 4))

        torques = torch.zeros(4, 1, device=b.device)
        for _ in range(10):
            b.step(torques)
        pos_after = b.dof_pos.clone()

        b.dof_pos[:2] = 9.9
        b.dof_vel[:2] = 0.0
        b.reset_dof_state(torch.tensor([0, 1], device=b.device))

        assert torch.allclose(
            b.dof_pos[:2], torch.full((2, 1), 9.9, device=b.device), atol=1e-5
        )
        assert torch.allclose(b.dof_pos[2:], pos_after[2:], atol=1e-5)

    def test_dof_state_view_consistent_with_dof_pos(self, vsim_backend):
        b = vsim_backend
        b.dof_pos[:] = 2.71
        assert torch.allclose(
            b.dof_state[:, 0],
            torch.full((4,), 2.71, device=b.device),
            atol=1e-5,
        )
