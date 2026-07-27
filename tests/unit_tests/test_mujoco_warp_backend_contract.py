"""Run the SimBackend contract suite against MuJocoWarpBackend.

All tests mirror test_mujoco_cpu_backend_contract.py; the only difference is
the fixture (mujoco_warp_backend instead of mujoco_cpu_backend).

Skipped automatically if `mujoco_warp` is not installed or CUDA is unavailable.
"""

import torch
import pytest


# ══════════════════════════════════════════════════════════════════════════════
# Tensor shape contract
# ══════════════════════════════════════════════════════════════════════════════


class TestTensorShapes:
    def test_dof_pos_shape(self, mujoco_warp_backend):
        assert mujoco_warp_backend.dof_pos.shape == (4, 1)

    def test_dof_vel_shape(self, mujoco_warp_backend):
        assert mujoco_warp_backend.dof_vel.shape == (4, 1)

    def test_dof_state_shape(self, mujoco_warp_backend):
        assert mujoco_warp_backend.dof_state.shape == (4, 2)

    def test_root_states_shape(self, mujoco_warp_backend):
        assert mujoco_warp_backend.root_states.shape == (4, 13)

    def test_contact_forces_shape(self, mujoco_warp_backend):
        cf = mujoco_warp_backend.contact_forces
        assert cf.ndim == 3
        assert cf.shape[0] == 4
        assert cf.shape[2] == 3

    def test_shapes_unchanged_after_step(self, mujoco_warp_backend):
        b = mujoco_warp_backend
        torques = torch.zeros(4, 1, device=b.device)
        b.step(torques)
        assert b.dof_pos.shape == (4, 1)
        assert b.dof_vel.shape == (4, 1)

    def test_device_matches_setup(self, mujoco_warp_backend):
        b = mujoco_warp_backend
        assert b.dof_pos.device.type == b.device.split(":")[0]
        assert b.dof_vel.device.type == b.device.split(":")[0]


# ══════════════════════════════════════════════════════════════════════════════
# Metadata contract
# ══════════════════════════════════════════════════════════════════════════════


class TestMetadata:
    def test_num_dof(self, mujoco_warp_backend):
        assert mujoco_warp_backend.num_dof == 1

    def test_num_bodies(self, mujoco_warp_backend):
        assert mujoco_warp_backend.num_bodies >= 1

    def test_dof_names_length(self, mujoco_warp_backend):
        b = mujoco_warp_backend
        assert len(b.dof_names) == b.num_dof

    def test_body_names_length(self, mujoco_warp_backend):
        b = mujoco_warp_backend
        assert len(b.body_names) == b.num_bodies

    def test_find_body_index_known(self, mujoco_warp_backend):
        idx = mujoco_warp_backend.find_body_index("pole")
        assert 0 <= idx < mujoco_warp_backend.num_bodies

    def test_find_body_index_unknown_raises(self, mujoco_warp_backend):
        with pytest.raises((ValueError, KeyError, IndexError)):
            mujoco_warp_backend.find_body_index("nonexistent_body_xyz")


# ══════════════════════════════════════════════════════════════════════════════
# Physics / step contract
# ══════════════════════════════════════════════════════════════════════════════


class TestPhysics:
    def test_zero_torque_gravity_accelerates(self, mujoco_warp_backend):
        b = mujoco_warp_backend
        b.dof_pos[:] = torch.pi / 2
        b.dof_vel[:] = 0.0
        b.reset_dof_state(torch.arange(4))
        torques = torch.zeros(4, 1, device=b.device)
        b.step(torques)
        assert b.dof_vel.abs().mean() > 1e-6

    def test_upright_equilibrium_near_zero(self, mujoco_warp_backend):
        b = mujoco_warp_backend
        b.dof_pos[:] = 0.0
        b.dof_vel[:] = 0.0
        b.reset_dof_state(torch.arange(4))
        torques = torch.zeros(4, 1, device=b.device)
        b.step(torques)
        assert b.dof_vel.abs().mean() < 1e-4

    def test_positive_torque_increases_velocity(self, mujoco_warp_backend):
        b = mujoco_warp_backend
        b.dof_pos[:] = 0.0
        b.dof_vel[:] = 0.0
        b.reset_dof_state(torch.arange(4))
        torques = torch.ones(4, 1, device=b.device) * 2.0
        b.step(torques)
        assert b.dof_vel[:, 0].mean() > 0

    def test_environments_evolve_independently(self, mujoco_warp_backend_16):
        b = mujoco_warp_backend_16
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
    def test_reset_writes_pos(self, mujoco_warp_backend):
        b = mujoco_warp_backend
        target = 1.23
        b.dof_pos[:] = target
        b.reset_dof_state(torch.arange(4))
        assert torch.allclose(
            b.dof_pos, torch.full((4, 1), target, device=b.device), atol=1e-5
        )

    def test_reset_writes_vel(self, mujoco_warp_backend):
        b = mujoco_warp_backend
        b.dof_vel[:] = 4.56
        b.reset_dof_state(torch.arange(4))
        assert torch.allclose(
            b.dof_vel, torch.full((4, 1), 4.56, device=b.device), atol=1e-5
        )

    def test_partial_reset(self, mujoco_warp_backend):
        b = mujoco_warp_backend
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

        assert torch.allclose(
            b.dof_pos[:2], torch.full((2, 1), 99.0, device=b.device), atol=1e-5
        )
        assert torch.allclose(b.dof_pos[2:], pos_after[2:], atol=1e-5)

    def test_dof_state_consistent_after_sync(self, mujoco_warp_backend):
        # On warp, qpos/qvel live in SEPARATE Warp arrays, so dof_state cannot
        # share storage with dof_pos (as it does on cpu/vsim); it is an
        # assembled buffer refreshed at sync points (step / reset).  The task
        # caches dof_state once and reads it after steps, so the contract is
        # "dof_state reflects dof_pos/dof_vel after a sync", verified here via
        # reset_dof_state (which syncs without advancing physics).  A per-call
        # torch.stack copy passed a naive read-consistency check but left the
        # task's cached reference frozen at the init zeros — the pendulum
        # _reward_equilibrium staleness bug (2026-07-24, see
        # test_task_state_liveness).  Refreshing in the getter is fenced off
        # (the root_states-scar anti-pattern): getters stay plain returns.
        b = mujoco_warp_backend
        b.dof_pos[:] = 2.71
        b.dof_vel[:] = 0.0
        b.reset_dof_state(torch.arange(b._num_envs, device=b.device))
        assert torch.allclose(
            b.dof_state[:, 0], torch.full_like(b.dof_state[:, 0], 2.71), atol=1e-5
        )
        assert torch.allclose(b.dof_state[:, 1], torch.zeros_like(b.dof_state[:, 1]))
