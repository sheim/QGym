"""Contract tests for floating-base (legged) robot backends.

Uses the mini_cheetah URDF (12 actuated DOFs + free joint).
Tests run against both MuJocoCPUBackend and MuJocoWarpBackend.
"""

import numpy as np
import pytest
import torch


def _reset_mask(backend, selected=None):
    mask = torch.zeros(
        backend.root_states.shape[0],
        dtype=torch.bool,
        device=backend.device,
    )
    if selected is None:
        mask.fill_(True)
    else:
        mask[selected] = True
    return mask


def _assert_limited_dof_reset_clamps(backend):
    props = backend._make_dof_props(backend._mjm)
    lower = torch.tensor(props["lower"], dtype=torch.float, device=backend.device)
    upper = torch.tensor(props["upper"], dtype=torch.float, device=backend.device)
    requested = torch.where(
        torch.arange(backend.num_dof, device=backend.device) % 2 == 0,
        lower - 1.0,
        upper + 1.0,
    )
    untouched = backend.dof_pos[1:].clone()

    backend.dof_pos[0] = requested
    backend.reset_state(_reset_mask(backend, [0]))

    expected = torch.where(
        torch.arange(backend.num_dof, device=backend.device) % 2 == 0,
        lower,
        upper,
    )
    torch.testing.assert_close(backend.dof_pos[0], expected)
    torch.testing.assert_close(backend.dof_pos[1:], untouched)


def _assert_rigid_body_state_is_current_after_step(backend):
    device = backend.device
    num_envs = backend.root_states.shape[0]
    reset_mask = _reset_mask(backend)
    torques = torch.zeros(num_envs, backend.num_dof, device=device)

    backend.root_states[:, :3] = torch.tensor([0.0, 0.0, 5.0], device=device)
    backend.root_states[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    backend.root_states[:, 7:13] = 0.0
    backend.dof_vel.zero_()
    backend.reset_state(reset_mask)

    # The first step puts the assembled public body state on a known native
    # state. The second step must expose that step's result, not the first
    # step's pre-integration kinematics.
    backend.step(torques)
    root_position = backend.root_states[:, :3].clone()
    body_state = backend.rigid_body_states.view(num_envs, backend.num_bodies, 13)
    body_position = body_state[:, :, :3].clone()

    backend.step(torques)
    root_translation = backend.root_states[:, :3] - root_position
    body_state = backend.rigid_body_states.view(num_envs, backend.num_bodies, 13)
    body_translation = body_state[:, :, :3] - body_position

    torch.testing.assert_close(
        body_translation,
        root_translation[:, None, :].expand_as(body_translation),
        atol=2e-6,
        rtol=1e-4,
    )
    torch.testing.assert_close(
        body_state[:, :, 7:10],
        backend.root_states[:, None, 7:10].expand_as(body_state[:, :, 7:10]),
        atol=2e-6,
        rtol=1e-4,
    )


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
    @pytest.mark.parametrize(
        "task_name,njmax,ccd_iterations,disable_multiccd,solref",
        [
            ("go2", 256, 50, True, [0.005, 1.0]),
            ("go2trot", 256, 50, True, [0.005, 1.0]),
            ("mini_cheetah", 200, 50, False, [0.02, 1.0]),
            ("humanoid", -1, 35, False, [0.02, 1.0]),
            ("humanoid_running", -1, 35, False, [0.02, 1.0]),
            ("pendulum", -1, 35, False, [0.02, 1.0]),
        ],
    )
    def test_task_solver_settings_survive_config_inheritance(
        self, task_name, njmax, ccd_iterations, disable_multiccd, solref
    ):
        """Sharing Go2 defaults must not retune other robots' contact physics."""
        import mujoco

        from gym.envs import task_registry
        from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend

        cfg = type(task_registry.env_cfgs[task_name])()
        model = MuJocoCPUBackend()._load_model(cfg, discard_visual=True)

        assert model.njmax == njmax
        assert model.opt.ccd_iterations == ccd_iterations
        expected_flags = (
            int(mujoco.mjtDisableBit.mjDSBL_MULTICCD) if disable_multiccd else 0
        )
        assert model.opt.disableflags == expected_flags
        np.testing.assert_allclose(
            model.geom_solref, np.broadcast_to(solref, model.geom_solref.shape)
        )

    def test_solver_option_override_retains_inherited_collision_settings(self):
        import mujoco

        from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
        from gym.envs.go2.go2trot_config import Go2TrotCfg

        class TunedGo2TrotCfg(Go2TrotCfg):
            class mujoco(Go2TrotCfg.mujoco):
                ccd_iterations = 75
                njmax = 300
                solref = [0.01, 1.5]

        model = MuJocoCPUBackend()._load_model(TunedGo2TrotCfg(), discard_visual=True)

        assert model.opt.ccd_iterations == 75
        assert model.njmax == 300
        np.testing.assert_allclose(
            model.geom_solref,
            np.broadcast_to([0.01, 1.5], model.geom_solref.shape),
        )
        assert model.opt.disableflags == int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
        assert Go2TrotCfg().mujoco.ccd_iterations == 50

    def test_ground_friction_uses_mujoco_slot_semantics(self, legged_cpu_backend):
        """Ground friction is [sliding, torsional, rolling], not static/dynamic."""
        import mujoco

        model = legged_cpu_backend._mjm
        plane_ids = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
        assert len(plane_ids) == 1
        np.testing.assert_allclose(
            model.geom_friction[plane_ids[0]],
            [1.0, 0.005, 0.0001],
        )

    def test_configured_friction_is_shared_by_robot_and_ground(self):
        """Robot defaults must not override terrain coefficients below one."""
        pytest.importorskip("mujoco")
        from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
        from tests.unit_tests.conftest import _make_mini_cheetah_cfg

        cfg = _make_mini_cheetah_cfg()
        cfg.terrain.static_friction = 0.4
        cfg.terrain.dynamic_friction = 0.35
        backend = MuJocoCPUBackend()
        backend.setup(cfg, num_envs=1, device="cpu", task=None)
        try:
            np.testing.assert_allclose(
                backend._mjm.geom_friction,
                np.broadcast_to(
                    [0.35, 0.005, 0.0001], backend._mjm.geom_friction.shape
                ),
            )
        finally:
            backend.close()

    def test_configured_geom_attributes_are_applied(self):
        """Compiled MuJoCo geoms receive the config's solver parameters."""
        pytest.importorskip("mujoco")
        from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
        from tests.unit_tests.conftest import _make_mini_cheetah_cfg

        cfg = _make_mini_cheetah_cfg()
        cfg.mujoco.solref = [0.005, 1.0]
        backend = MuJocoCPUBackend()
        backend.setup(cfg, num_envs=1, device="cpu", task=None)
        try:
            np.testing.assert_allclose(
                backend._mjm.geom_solref,
                np.broadcast_to([0.005, 1.0], backend._mjm.geom_solref.shape),
            )
        finally:
            backend.close()

    def test_robot_above_ground(self, legged_cpu_backend):
        """Robot shouldn't fall through the ground plane."""
        b = legged_cpu_backend
        # Set initial height
        b.root_states[:, 2] = 0.35
        b.reset_state(_reset_mask(b))
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

    def test_rigid_body_state_is_current_after_step(self, legged_cpu_backend):
        _assert_rigid_body_state_is_current_after_step(legged_cpu_backend)

    def test_rigid_body_velocity_is_at_published_body_origin(self, legged_cpu_backend):
        import mujoco

        backend = legged_cpu_backend
        backend.root_states[:, 2] = 2.0
        backend.root_states[:, 7:10] = torch.tensor([0.2, -0.1, 0.3])
        backend.root_states[:, 10:13] = torch.tensor([0.4, -0.2, 0.1])
        backend.dof_vel[:] = torch.linspace(-0.5, 0.5, backend.num_dof)
        reset_mask = _reset_mask(backend)
        backend.reset_state(reset_mask)
        backend.step(torch.zeros(4, backend.num_dof))

        public_state = backend.rigid_body_states.view(4, backend.num_bodies, 13)[0]
        data = backend._datas[0]
        model = backend._model_for_env(0)
        for canonical_id, native_id in enumerate(backend._canonical_to_native_body_np):
            linear_jacobian = np.empty((3, model.nv))
            angular_jacobian = np.empty((3, model.nv))
            mujoco.mj_jac(
                model,
                data,
                linear_jacobian,
                angular_jacobian,
                data.xpos[native_id],
                int(native_id),
            )
            linear_velocity = linear_jacobian @ data.qvel
            angular_velocity = angular_jacobian @ data.qvel
            np.testing.assert_allclose(
                public_state[canonical_id, 7:13].numpy(),
                np.concatenate((linear_velocity, angular_velocity)),
                atol=2e-6,
                rtol=1e-5,
            )


# ── Reset ──────────────────────────────────────────────────────────────────────


class TestLeggedReset:
    def test_limited_dof_reset_clamps_to_asset_range(self, legged_cpu_backend):
        _assert_limited_dof_reset_clamps(legged_cpu_backend)

    def test_root_state_persists_after_reset(self, legged_cpu_backend):
        b = legged_cpu_backend
        b.root_states[0, 2] = 1.0  # set z=1
        b.root_states[0, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        b.reset_state(_reset_mask(b, [0]))
        # One step should keep it roughly near z=1
        b.step(torch.zeros(4, b.num_dof))
        assert b.root_states[0, 2].item() > 0.9

    def test_dof_state_persists_after_reset(self, legged_cpu_backend):
        b = legged_cpu_backend
        b.dof_pos[0, 0] = 0.5
        b.dof_vel[0, :] = 0.0
        b.reset_state(_reset_mask(b, [0]))
        b.step(torch.zeros(4, b.num_dof))
        # Should be close to 0.5 after one step
        assert abs(b.dof_pos[0, 0].item() - 0.5) < 0.1


# ── Warp backend (same tests) ─────────────────────────────────────────────────


@pytest.mark.warp
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

    def test_limited_dof_reset_clamps_to_asset_range(self, legged_warp_backend):
        _assert_limited_dof_reset_clamps(legged_warp_backend)

    def test_rigid_body_state_is_current_after_step(self, legged_warp_backend):
        _assert_rigid_body_state_is_current_after_step(legged_warp_backend)


# ── Cross-backend comparison ──────────────────────────────────────────────────


@pytest.mark.warp
class TestLeggedCrossBackend:
    @pytest.fixture
    def cpu_and_warp(self):
        if not torch.cuda.is_available():
            pytest.fail("Warp tests requested but CUDA is not available", pytrace=False)

        from tests.unit_tests.conftest import _make_mini_cheetah_cfg
        from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
        from gym.envs.base.mujoco_warp_backend import MuJocoWarpBackend

        cfg = _make_mini_cheetah_cfg()
        cpu = MuJocoCPUBackend()
        cpu.setup(cfg, num_envs=4, device="cpu", task=None)
        warp = MuJocoWarpBackend()
        warp.setup(cfg, num_envs=4, device="cuda:0", task=None)
        yield cpu, warp
        cpu.close()
        warp.close()

    def test_trajectories_match(self, cpu_and_warp):
        """CPU and Warp backends should produce near-identical states.

        Contact-rich floating-base rollouts are chaotic: float-level
        implementation differences grow ~10x per 25 steps once contacts
        engage (measured 2026-07-11: ~1e-6 through step 100, ~3e-2 by step
        200).  A single flat tolerance either misses systematic modeling
        bugs (too loose early) or trips on chaos (too tight late), so the
        check is split:

        - steps 1-100 (pre-chaos): 1e-4 — any real mismatch (wrong mass,
          inertia, quaternion swizzle, missing contact) exceeds this
          immediately; measured margin ~10x.
        - steps 101-200: 0.2 — blow-up detector only; chaos alone reaches
          ~5e-2 at step 200.
        """
        cpu, warp = cpu_and_warp
        N = 4

        # Set identical initial height
        cpu.root_states[:, 2] = 0.35
        cpu.reset_state(_reset_mask(cpu))
        warp.root_states[:, 2] = 0.35
        warp.reset_state(_reset_mask(warp))

        cpu_torques = torch.zeros(N, 12)
        warp_torques = torch.zeros(N, 12, device="cuda:0")

        for step in range(200):
            cpu.step(cpu_torques)
            warp.step(warp_torques)

            pos_err = (cpu.dof_pos - warp.dof_pos.cpu()).abs().max().item()
            root_err = (cpu.root_states - warp.root_states.cpu()).abs().max().item()
            body_err = (
                (cpu.rigid_body_states - warp.rigid_body_states.cpu())
                .abs()
                .max()
                .item()
            )

            tol = 1e-4 if step < 100 else 0.2
            assert pos_err < tol, f"DOF pos diverged at step {step}: {pos_err:.2e}"
            assert root_err < tol, (
                f"Root states diverged at step {step}: {root_err:.2e}"
            )
            if step < 100:
                assert body_err < 2e-3, (
                    f"Rigid-body states diverged at step {step}: {body_err:.2e}"
                )
