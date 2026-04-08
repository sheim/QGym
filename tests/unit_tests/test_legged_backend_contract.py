"""Contract tests for floating-base (legged) robot backends.

Uses the mini_cheetah URDF (12 actuated DOFs + free joint).
Tests run against both MuJocoCPUBackend and MuJocoWarpBackend.
"""

import pytest
import torch


# ── Shapes and metadata ────────────────────────────────────────────────────────


class TestLeggedShapes:
    def test_num_dof(self, legged_cpu_backend):
        assert legged_cpu_backend.num_dof == 12

    def test_dof_pos_shape(self, legged_cpu_backend):
        assert legged_cpu_backend.dof_pos.shape == (4, 12)

    def test_dof_vel_shape(self, legged_cpu_backend):
        assert legged_cpu_backend.dof_vel.shape == (4, 12)

    def test_dof_state_shape(self, legged_cpu_backend):
        assert legged_cpu_backend.dof_state.shape == (4 * 12, 2)

    def test_root_states_shape(self, legged_cpu_backend):
        assert legged_cpu_backend.root_states.shape == (4, 13)

    def test_rigid_body_states_shape(self, legged_cpu_backend):
        b = legged_cpu_backend
        assert b.rigid_body_states.shape == (4 * b.num_bodies, 13)

    def test_contact_forces_shape(self, legged_cpu_backend):
        b = legged_cpu_backend
        assert b.contact_forces.shape == (4, b.num_bodies, 3)

    def test_dof_names_count(self, legged_cpu_backend):
        assert len(legged_cpu_backend.dof_names) == 12

    def test_body_names_nonempty(self, legged_cpu_backend):
        assert len(legged_cpu_backend.body_names) > 0

    def test_contact_indices_nonempty(self, legged_cpu_backend):
        # cfg has penalize_contacts_on=["thigh"], terminate_after_contacts_on=["base"]
        assert len(legged_cpu_backend.penalised_contact_indices) > 0
        assert len(legged_cpu_backend.termination_contact_indices) > 0


# ── Quaternion convention ──────────────────────────────────────────────────────


class TestLeggedQuaternion:
    def test_identity_quat_scalar_last(self, legged_cpu_backend):
        """Initial root quaternion should be identity [0,0,0,1] (scalar-last)."""
        quat = legged_cpu_backend.root_states[0, 3:7]
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0])
        assert torch.allclose(quat, expected, atol=1e-4)

    def test_quat_norm_after_step(self, legged_cpu_backend):
        b = legged_cpu_backend
        torques = torch.zeros(4, b.num_dof)
        for _ in range(50):
            b.step(torques)
        quat = b.root_states[:, 3:7]
        norms = quat.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-4)


# ── Physics sanity ─────────────────────────────────────────────────────────────


class TestLeggedPhysics:
    def test_robot_above_ground(self, legged_cpu_backend):
        """Robot shouldn't fall through the ground plane."""
        b = legged_cpu_backend
        # Set initial height
        b.root_states[:, 2] = 0.35
        b.reset_root_state(torch.arange(4))
        torques = torch.zeros(4, b.num_dof)
        for _ in range(500):
            b.step(torques)
        z = b.root_states[:, 2]
        assert (z > -0.1).all(), f"Robot fell through ground: z={z.tolist()}"

    def test_gravity_affects_root(self, legged_cpu_backend):
        """With no ground, gravity should pull the robot down."""
        b = legged_cpu_backend
        z_init = b.root_states[0, 2].item()
        # Disable contacts by zeroing contype (simulate free fall)
        b._mjm.geom_contype[:] = 0
        b._mjm.geom_conaffinity[:] = 0
        torques = torch.zeros(4, b.num_dof)
        for _ in range(100):
            b.step(torques)
        z_after = b.root_states[0, 2].item()
        assert z_after < z_init, "Gravity should pull robot down"


# ── Reset ──────────────────────────────────────────────────────────────────────


class TestLeggedReset:
    def test_reset_root_state_persists(self, legged_cpu_backend):
        b = legged_cpu_backend
        b.root_states[0, 2] = 1.0  # set z=1
        b.root_states[0, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        b.reset_root_state(torch.tensor([0]))
        # One step should keep it roughly near z=1
        b.step(torch.zeros(4, b.num_dof))
        assert b.root_states[0, 2].item() > 0.9

    def test_reset_dof_state_persists(self, legged_cpu_backend):
        b = legged_cpu_backend
        b.dof_pos[0, 0] = 0.5
        b.dof_vel[0, :] = 0.0
        b.reset_dof_state(torch.tensor([0]))
        b.step(torch.zeros(4, b.num_dof))
        # Should be close to 0.5 after one step
        assert abs(b.dof_pos[0, 0].item() - 0.5) < 0.1


# ── Warp backend (same tests) ─────────────────────────────────────────────────


class TestLeggedWarpShapes:
    def test_num_dof(self, legged_warp_backend):
        assert legged_warp_backend.num_dof == 12

    def test_dof_pos_shape(self, legged_warp_backend):
        assert legged_warp_backend.dof_pos.shape == (4, 12)

    def test_root_states_shape(self, legged_warp_backend):
        assert legged_warp_backend.root_states.shape == (4, 13)

    def test_identity_quat_scalar_last(self, legged_warp_backend):
        quat = legged_warp_backend.root_states[0, 3:7].cpu()
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0])
        assert torch.allclose(quat, expected, atol=1e-4)

    def test_rigid_body_states_shape(self, legged_warp_backend):
        b = legged_warp_backend
        assert b.rigid_body_states.shape == (4 * b.num_bodies, 13)


# ── Cross-backend comparison ──────────────────────────────────────────────────


class TestLeggedCrossBackend:
    @pytest.fixture
    def cpu_and_warp(self):
        pytest.importorskip("mujoco")
        pytest.importorskip("mujoco_warp")
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for cross-backend comparison")

        from tests.unit_tests.conftest import _make_mini_cheetah_cfg
        from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
        from gym.envs.base.mujoco_warp_backend import MuJocoWarpBackend

        cfg = _make_mini_cheetah_cfg()
        cpu = MuJocoCPUBackend()
        cpu.setup(cfg, num_envs=4, device="cpu", task=None)
        warp = MuJocoWarpBackend()
        warp.setup(cfg, num_envs=4, device="cuda:0", task=None)
        return cpu, warp

    def test_trajectories_match(self, cpu_and_warp):
        """CPU and Warp backends should produce near-identical states."""
        cpu, warp = cpu_and_warp
        N = 4

        # Set identical initial height
        cpu.root_states[:, 2] = 0.35
        cpu.reset_root_state(torch.arange(N))
        warp.root_states[:, 2] = 0.35
        warp.reset_root_state(torch.arange(N, device="cuda:0"))

        cpu_torques = torch.zeros(N, 12)
        warp_torques = torch.zeros(N, 12, device="cuda:0")

        for step in range(200):
            cpu.step(cpu_torques)
            warp.step(warp_torques)

            pos_err = (cpu.dof_pos - warp.dof_pos.cpu()).abs().max().item()
            root_err = (cpu.root_states - warp.root_states.cpu()).abs().max().item()

            # Floating-base with contacts accumulates float diffs faster
            # than fixed-base pendulum; use looser tolerance
            assert pos_err < 0.01, f"DOF pos diverged at step {step}: {pos_err:.2e}"
            assert root_err < 0.01, (
                f"Root states diverged at step {step}: {root_err:.2e}"
            )
