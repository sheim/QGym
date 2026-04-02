"""Tests for the SimBackend contract.

These run against MockBackend and document the exact invariants that any
future backend (MuJocoWarpBackend, MuJocoCPUBackend) must satisfy.  When
implementing a new backend, run this suite against it to verify compliance.
"""

import torch
import pytest


# ══════════════════════════════════════════════════════════════════════════════
# Tensor shape contract
# ══════════════════════════════════════════════════════════════════════════════


class TestTensorShapes:
    def test_dof_pos_shape(self, backend):
        assert backend.dof_pos.shape == (4, 1), "dof_pos must be [num_envs, num_dof]"

    def test_dof_vel_shape(self, backend):
        assert backend.dof_vel.shape == (4, 1)

    def test_dof_state_shape(self, backend):
        # [num_envs * num_dof, 2]
        assert backend.dof_state.shape == (4, 2)

    def test_root_states_shape(self, backend):
        assert backend.root_states.shape == (4, 13), (
            "root_states must be [num_envs, 13]: pos(3) quat(4) linvel(3) angvel(3)"
        )

    def test_contact_forces_shape(self, backend):
        cf = backend.contact_forces
        assert cf.ndim == 3, "contact_forces must be 3-dimensional"
        assert cf.shape[0] == 4, "first dim must be num_envs"
        assert cf.shape[2] == 3, "last dim must be xyz force"

    def test_shapes_unchanged_after_step(self, backend):
        torques = torch.zeros(4, 1, device=backend.device)
        backend.step(torques)
        assert backend.dof_pos.shape == (4, 1)
        assert backend.dof_vel.shape == (4, 1)

    def test_device_matches(self, backend, device):
        assert str(backend.dof_pos.device) == device or (
            backend.dof_pos.device.type == device.split(":")[0]
        ), "dof_pos must live on the declared device"
        assert backend.dof_vel.device == backend.dof_pos.device


# ══════════════════════════════════════════════════════════════════════════════
# Metadata contract
# ══════════════════════════════════════════════════════════════════════════════


class TestMetadata:
    def test_num_dof(self, backend):
        assert backend.num_dof == 1

    def test_num_bodies(self, backend):
        assert backend.num_bodies >= 1

    def test_dof_names_length(self, backend):
        assert len(backend.dof_names) == backend.num_dof

    def test_body_names_length(self, backend):
        assert len(backend.body_names) == backend.num_bodies

    def test_find_body_index_known(self, backend):
        idx = backend.find_body_index("pole")
        assert 0 <= idx < backend.num_bodies

    def test_find_body_index_unknown_raises(self, backend):
        with pytest.raises((ValueError, KeyError, IndexError)):
            backend.find_body_index("nonexistent_body_xyz")


# ══════════════════════════════════════════════════════════════════════════════
# Physics / step contract
# ══════════════════════════════════════════════════════════════════════════════


class TestPhysics:
    def test_zero_torque_gravity_accelerates(self, backend):
        """Pendulum at 90° with no torque must gain angular velocity."""
        backend.dof_pos[:] = torch.pi / 2
        backend.dof_vel[:] = 0.0
        backend.reset_dof_state(torch.arange(4))

        torques = torch.zeros(4, 1, device=backend.device)
        backend.step(torques)

        assert backend.dof_vel.abs().mean() > 1e-6, (
            "Gravity should produce nonzero velocity from 90° position"
        )

    def test_upright_equilibrium_is_unstable(self, backend):
        """Pendulum at θ=0 (upright) should stay near zero with zero torque
        (linearised — not a stability test, just a sign/symmetry check)."""
        backend.dof_pos[:] = 0.0
        backend.dof_vel[:] = 0.0
        backend.reset_dof_state(torch.arange(4))

        torques = torch.zeros(4, 1, device=backend.device)
        backend.step(torques)
        # At θ=0 the restoring torque is zero; velocity should remain ~0.
        assert backend.dof_vel.abs().mean() < 1e-4

    def test_energy_approximately_conserved(self, backend):
        """Free pendulum from 45°: mechanical energy should be roughly
        conserved over 200 steps (< 5% relative drift for a reasonable dt)."""
        backend.dof_pos[:] = torch.pi / 4
        backend.dof_vel[:] = 0.0
        backend.reset_dof_state(torch.arange(4))

        mass = backend._mass
        length = backend._length
        g = backend._gravity

        pe0 = mass * g * length * (1.0 - backend.dof_pos[:, 0].cos()).mean()
        ke0 = torch.tensor(0.0, device=backend.device)
        E0 = pe0 + ke0

        torques = torch.zeros(4, 1, device=backend.device)
        for _ in range(200):
            backend.step(torques)

        ke = 0.5 * mass * length**2 * backend.dof_vel[:, 0].pow(2).mean()
        pe = mass * g * length * (1.0 - backend.dof_pos[:, 0].cos()).mean()
        E1 = ke + pe

        rel_drift = ((E1 - E0) / (E0.abs() + 1e-8)).abs().item()
        assert rel_drift < 0.05, (
            f"Energy drifted {rel_drift:.1%} over 200 steps — "
            "check timestep or integration scheme"
        )

    def test_positive_torque_increases_velocity(self, backend):
        """Positive torque at θ=0 should accelerate in the positive direction."""
        backend.dof_pos[:] = 0.0
        backend.dof_vel[:] = 0.0
        backend.reset_dof_state(torch.arange(4))

        torques = torch.ones(4, 1, device=backend.device) * 2.0
        backend.step(torques)

        assert backend.dof_vel[:, 0].mean() > 0, (
            "Positive torque should produce positive angular velocity"
        )

    def test_environments_evolve_independently(self, backend_16):
        """Two groups of environments with different initial conditions must
        diverge — proving they are truly parallel and independent."""
        b = backend_16
        b.dof_pos[:8] = 0.0
        b.dof_pos[8:] = torch.pi / 2
        b.dof_vel[:] = 0.0
        b.reset_dof_state(torch.arange(16))

        torques = torch.zeros(16, 1, device=b.device)
        for _ in range(20):
            b.step(torques)

        mean_a = b.dof_pos[:8, 0].mean()
        mean_b = b.dof_pos[8:, 0].mean()
        assert (mean_a - mean_b).abs() > 0.05, (
            "Environments with different ICs should produce different states"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Reset contract
# ══════════════════════════════════════════════════════════════════════════════


class TestReset:
    def test_reset_writes_pos(self, backend):
        """After writing dof_pos and calling reset_dof_state, the value
        persists (i.e. was not overwritten by the backend)."""
        target = 1.23
        backend.dof_pos[:] = target
        backend.reset_dof_state(torch.arange(4))
        assert torch.allclose(
            backend.dof_pos,
            torch.full((4, 1), target, device=backend.device),
            atol=1e-5,
        )

    def test_reset_writes_vel(self, backend):
        backend.dof_vel[:] = 4.56
        backend.reset_dof_state(torch.arange(4))
        assert torch.allclose(
            backend.dof_vel,
            torch.full((4, 1), 4.56, device=backend.device),
            atol=1e-5,
        )

    def test_partial_reset(self, backend):
        """Resetting a subset of envs must not change the others."""
        backend.dof_pos[:] = 0.0
        backend.dof_vel[:] = 0.0
        backend.reset_dof_state(torch.arange(4))

        # Evolve all envs for a few steps
        torques = torch.zeros(4, 1, device=backend.device)
        for _ in range(10):
            backend.step(torques)
        pos_after = backend.dof_pos.clone()

        # Reset only envs 0 and 1 to a known value
        backend.dof_pos[:2] = 99.0
        backend.dof_vel[:2] = 0.0
        backend.reset_dof_state(torch.tensor([0, 1]))

        # Envs 0, 1 should now be at 99
        assert torch.allclose(
            backend.dof_pos[:2],
            torch.full((2, 1), 99.0, device=backend.device),
            atol=1e-5,
        )
        # Envs 2, 3 should be unchanged
        assert torch.allclose(backend.dof_pos[2:], pos_after[2:], atol=1e-5)

    def test_dof_state_view_consistent_with_dof_pos_vel(self, backend):
        """dof_state must be a view: writing dof_pos must update dof_state."""
        backend.dof_pos[:] = 2.71
        # dof_state[:, 0] should reflect the same value (it's a view)
        assert torch.allclose(
            backend.dof_state[:, 0],
            torch.full((4,), 2.71, device=backend.device),
            atol=1e-5,
        ), "dof_state[..., 0] must be the same storage as dof_pos"
