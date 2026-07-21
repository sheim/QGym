"""Floating-base (mini_cheetah) contract on VSimBackend.

Adapted from test_legged_backend_contract.py; the CPU↔warp lockstep test is
replaced by physics invariants (cross-ENGINE lockstep is meaningless), plus
a static-weight check that validates contact-force frame and sign: the total
vertical contact force on a settled robot must carry its weight
(MINI_CHEETAH mass ≈ 8.292 kg → ≈ 81.3 N).

Opt-in: runs only under scripts/run_vsim_tests.sh (license + CUDA).
"""

import torch

MC_WEIGHT_N = 8.292 * 9.81


class TestLeggedShapes:
    def test_num_dof(self, legged_vsim_backend):
        assert legged_vsim_backend.num_dof == 12

    def test_num_bodies_includes_feet(self, legged_vsim_backend):
        b = legged_vsim_backend
        # 17 links: base + 4 × (hip, thigh, shank, foot); no `world` in vsim
        assert b.num_bodies == 17
        assert sum("foot" in n for n in b.body_names) == 4

    def test_dof_state_shape(self, legged_vsim_backend):
        assert legged_vsim_backend.dof_state.shape == (4 * 12, 2)

    def test_root_states_shape(self, legged_vsim_backend):
        assert legged_vsim_backend.root_states.shape == (4, 13)

    def test_rigid_body_states_shape(self, legged_vsim_backend):
        b = legged_vsim_backend
        assert b.rigid_body_states.shape == (4 * b.num_bodies, 13)

    def test_contact_indices_nonempty(self, legged_vsim_backend):
        b = legged_vsim_backend
        assert len(b.penalised_contact_indices) > 0
        assert len(b.termination_contact_indices) > 0


class TestQuaternionConvention:
    def test_identity_quat_scalar_last(self, legged_vsim_backend):
        """Upright spawn → quat ≈ [0,0,0,1] in scalar-last convention."""
        b = legged_vsim_backend
        quat = b.root_states[:, 3:7]
        assert (quat[:, 3].abs() > 0.99).all(), f"w component wrong: {quat[0]}"

    def test_quat_stays_normalized(self, legged_vsim_backend):
        b = legged_vsim_backend
        torques = torch.zeros(4, 12, device=b.device)
        for _ in range(50):
            b.step(torques)
        norms = b.root_states[:, 3:7].norm(dim=-1)
        assert torch.allclose(norms, torch.ones(4, device=b.device), atol=1e-4)


class TestLeggedPhysics:
    def test_does_not_fall_through_plane(self, legged_vsim_backend):
        b = legged_vsim_backend
        torques = torch.zeros(4, 12, device=b.device)
        for _ in range(500):
            b.step(torques)
        z = b.root_states[:, 2]
        assert (z > -0.05).all(), f"fell through plane: z={z.tolist()}"
        assert (z < 0.4).all(), f"floating unnaturally: z={z.tolist()}"

    def test_contact_forces_carry_weight(self, legged_vsim_backend):
        """Frame/sign validation: settled robot's +z contact sum ≈ m·g.

        Averaged over the last 50 steps: instantaneous totals oscillate
        through zero during collapse rebounds (measured 2026-07-12), but the
        settled mean must carry the weight.
        """
        b = legged_vsim_backend
        torques = torch.zeros(4, 12, device=b.device)
        for _ in range(800):  # 1.6 s — collapse and settle
            b.step(torques)
        total_up = torch.zeros(4, device=b.device)
        for _ in range(50):
            b.step(torques)
            total_up += b.contact_forces[..., 2].sum(dim=-1)
        rel = total_up / 50 / MC_WEIGHT_N
        assert ((rel > 0.6) & (rel < 1.4)).all(), (
            f"mean vertical contact force {(total_up / 50).tolist()} N vs "
            f"weight {MC_WEIGHT_N:.1f} N — frame or sign error?"
        )


class TestLeggedReset:
    def test_root_reset_persists(self, legged_vsim_backend):
        b = legged_vsim_backend
        b.root_states[:, :3] = torch.tensor([0.0, 0.0, 1.0], device=b.device)
        b.root_states[:, 3:7] = torch.tensor([0, 0, 0, 1.0], device=b.device)
        b.root_states[:, 7:13] = 0.0
        b.reset_root_state(torch.arange(4, device=b.device))
        b.step(torch.zeros(4, 12, device=b.device))
        assert (b.root_states[:, 2] > 0.9).all()

    def test_dof_reset_persists(self, legged_vsim_backend):
        b = legged_vsim_backend
        b.dof_pos[:] = 0.5
        b.dof_vel[:] = 0.0
        b.reset_dof_state(torch.arange(4, device=b.device))
        assert (b.dof_pos - 0.5).abs().max() < 0.1

    def test_partial_root_reset(self, legged_vsim_backend):
        b = legged_vsim_backend
        before = b.root_states.clone()
        b.root_states[:2, 2] = 2.0
        b.reset_root_state(torch.tensor([0, 1], device=b.device))
        assert (b.root_states[:2, 2] - 2.0).abs().max() < 1e-4
        assert torch.allclose(b.root_states[2:], before[2:], atol=1e-5)
