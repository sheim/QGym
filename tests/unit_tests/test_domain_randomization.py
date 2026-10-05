import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch

from gym import GYM_ROOT_DIR
from gym.envs.base.mujoco_config import MuJoCoCfg
from tests.unit_tests.conftest import vsim_guard


def _reset_mask(num_envs, device, selected=None):
    mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
    if selected is None:
        mask.fill_(True)
    else:
        mask[selected] = True
    return mask


def _domain_randomization_cfg(
    *, contact_friction=None, stiffness=None, damping=None, link_mass=None
):
    return SimpleNamespace(
        startup=SimpleNamespace(
            contact_friction_range=contact_friction,
            link_mass_scale_range=link_mass,
        ),
        episode=SimpleNamespace(
            scale_ranges={
                name: values
                for name, values in (
                    ("p_gains", stiffness),
                    ("d_gains", damping),
                )
                if values is not None
            }
        ),
    )


def _friction_cfg():
    return SimpleNamespace(
        seed=11,
        asset=SimpleNamespace(
            file=(
                f"{GYM_ROOT_DIR}/resources/robots/friction_sled/urdf/friction_sled.urdf"
            ),
            vsim_visual_mesh_dir=None,
            joint_damping=1.0,
            rotor_inertia=0.0,
            disable_gravity=False,
            fix_base_link=False,
            penalize_contacts_on=[],
            terminate_after_contacts_on=[],
        ),
        init_state=SimpleNamespace(
            pos=[0.0, 0.0, 0.105],
            rot=[0.0, 0.0, 0.0, 1.0],
        ),
        terrain=SimpleNamespace(
            mesh_type="plane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        domain_randomization=_domain_randomization_cfg(contact_friction=[0.2, 1.0]),
        mujoco=MuJoCoCfg(),
        sim=SimpleNamespace(gravity=[5.0, 0.0, -9.81]),
        sim_dt=0.002,
    )


def _link_mass_cfg():
    return SimpleNamespace(
        seed=11,
        asset=SimpleNamespace(
            file=f"{GYM_ROOT_DIR}/resources/robots/pendulum/urdf/pendulum.urdf",
            vsim_visual_mesh_dir=None,
            joint_damping=0.0,
            rotor_inertia=0.0,
            disable_gravity=False,
            fix_base_link=True,
            penalize_contacts_on=[],
            terminate_after_contacts_on=[],
        ),
        init_state=SimpleNamespace(
            pos=[0.0, 0.0, 0.0],
            rot=[0.0, 0.0, 0.0, 1.0],
        ),
        domain_randomization=_domain_randomization_cfg(
            link_mass=[0.5, 2.0],
        ),
        mujoco=MuJoCoCfg(),
        sim=SimpleNamespace(gravity=[0.0, 0.0, -9.81]),
        sim_dt=0.005,
    )


def _set_explicit_friction(backend):
    cached = backend.contact_friction
    backend.set_contact_friction(
        torch.tensor([0, 1], device=backend.device),
        torch.tensor([0.2, 0.8], device=backend.device),
    )
    assert backend.contact_friction is cached
    torch.testing.assert_close(backend.contact_friction.cpu(), torch.tensor([0.2, 0.8]))

    backend.set_contact_friction(
        torch.tensor([1], device=backend.device),
        torch.tensor([0.7], device=backend.device),
    )
    torch.testing.assert_close(backend.contact_friction.cpu(), torch.tensor([0.2, 0.7]))


def _assert_friction_changes_motion(backend):
    backend.set_contact_friction(
        torch.tensor([0, 1], device=backend.device),
        torch.tensor([0.2, 0.8], device=backend.device),
    )
    reset_mask = _reset_mask(2, backend.device)
    backend.dof_pos.zero_()
    backend.dof_vel.zero_()
    backend.root_states.zero_()
    backend.root_states[:, 2] = 0.105
    backend.root_states[:, 6] = 1.0
    backend.reset_state(reset_mask)

    torques = torch.zeros(2, backend.num_dof, device=backend.device)
    for _ in range(500):
        backend.step(torques)

    displacement = backend.root_states[:, 0].cpu()
    assert displacement[0] > 0.5, displacement
    assert displacement[1].abs() < 0.15, displacement
    assert displacement[0] - displacement[1] > 0.5, displacement


def _assert_link_mass_changes_acceleration(backend):
    pole = backend.body_names.index("pole")
    nominal_mass = backend.link_mass.clone()
    nominal_inertia = backend.link_inertia.clone()
    scales = torch.ones(2, backend.num_bodies, device=backend.device)
    scales[1, pole] = 2.0
    backend.set_link_mass_scale(torch.tensor([0, 1], device=backend.device), scales)

    torch.testing.assert_close(backend.link_mass[0], nominal_mass[0])
    torch.testing.assert_close(backend.link_mass[1, pole], 2.0 * nominal_mass[1, pole])
    torch.testing.assert_close(
        backend.link_inertia[1, pole], 2.0 * nominal_inertia[1, pole]
    )

    backend.dof_pos.zero_()
    backend.dof_vel.zero_()
    backend.reset_state(_reset_mask(2, backend.device))
    backend.step(torch.full((2, backend.num_dof), 0.1, device=backend.device))
    speed = backend.dof_vel[:, 0].abs().cpu()
    assert speed[0] / speed[1] == pytest.approx(2.0, rel=0.05)


def _build_randomized_task(
    device,
    backend_name="mujoco",
    *,
    contact_friction=(0.5, 1.0),
    stiffness=None,
    damping=None,
    link_mass=None,
    reset_mode=None,
):
    from gym.envs.mini_cheetah.mini_cheetah import MiniCheetah
    from gym.envs.mini_cheetah.mini_cheetah_config import (
        MiniCheetahCfg,
        MiniCheetahRunnerCfg,
    )
    from gym.utils.task_registry import select_backend, task_registry

    cfg = MiniCheetahCfg()
    runner_cfg = MiniCheetahRunnerCfg()
    cfg.env.num_envs = 4
    cfg.seed = 17
    cfg.push_robots.toggle = False
    if reset_mode is not None:
        cfg.init_state.reset_mode = reset_mode
    cfg.domain_randomization.startup.contact_friction_range = contact_friction
    cfg.domain_randomization.startup.link_mass_scale_range = link_mass
    cfg.domain_randomization.episode.scale_ranges = {
        name: values
        for name, values in (
            ("p_gains", stiffness),
            ("d_gains", damping),
        )
        if values is not None
    }
    task_registry.convert_frequencies_to_params(cfg, runner_cfg)
    backend = select_backend(cfg, device, backend_name)
    return MiniCheetah(cfg, device, True, backend)


def _assert_task_startup_friction_randomization(device, backend_name="mujoco"):
    env = _build_randomized_task(device, backend_name)
    try:
        generator = torch.Generator(device=device).manual_seed(17)
        expected_initial = 0.5 + 0.5 * torch.rand(4, generator=generator, device=device)
        assert env.domain_randomizer._generators["contact_friction"].device == (
            torch.device(device)
        )
        torch.testing.assert_close(
            env.domain_randomizer.contact_friction, expected_initial
        )

        env._reset_idx(_reset_mask(4, device, [1, 3]))
        torch.testing.assert_close(
            env.domain_randomizer.contact_friction, expected_initial
        )
    finally:
        env._backend.close()


def _assert_task_pd_randomization(device, backend_name="mujoco"):
    env = _build_randomized_task(
        device,
        backend_name,
        contact_friction=None,
        stiffness=(0.8, 1.2),
        damping=(0.7, 1.3),
    )
    try:
        p_scale = env.domain_randomizer.episode_scale("p_gains")
        d_scale = env.domain_randomizer.episode_scale("d_gains")
        nominal_p = env.p_gains / p_scale
        nominal_d = env.d_gains / d_scale
        torch.testing.assert_close(
            env.p_gains,
            nominal_p * p_scale,
        )
        torch.testing.assert_close(
            env.d_gains,
            nominal_d * d_scale,
        )
        before_p = env.p_gains.clone()
        before_d = env.d_gains.clone()
        env._reset_idx(_reset_mask(4, device, [1, 3]))
        torch.testing.assert_close(
            env.p_gains,
            nominal_p * env.domain_randomizer.episode_scale("p_gains"),
        )
        torch.testing.assert_close(
            env.d_gains,
            nominal_d * env.domain_randomizer.episode_scale("d_gains"),
        )
        torch.testing.assert_close(env.p_gains[[0, 2]], before_p[[0, 2]])
        torch.testing.assert_close(env.d_gains[[0, 2]], before_d[[0, 2]])
        assert not torch.equal(env.p_gains[[1, 3]], before_p[[1, 3]])
        assert not torch.equal(env.d_gains[[1, 3]], before_d[[1, 3]])
    finally:
        env._backend.close()


def _assert_task_link_mass_randomization(device, backend_name="mujoco"):
    env = _build_randomized_task(
        device,
        backend_name,
        contact_friction=None,
        link_mass=(0.8, 1.2),
    )
    try:
        scales = env.domain_randomizer.link_mass_scale
        torch.testing.assert_close(
            env._backend.link_mass,
            env._backend._nominal_link_mass * scales,
        )
        torch.testing.assert_close(
            env._backend.link_inertia,
            env._backend._nominal_link_inertia * scales.unsqueeze(-1),
        )
        before_mass = env._backend.link_mass.clone()
        before_inertia = env._backend.link_inertia.clone()
        env._reset_idx(_reset_mask(4, device, [1, 3]))
        torch.testing.assert_close(env._backend.link_mass, before_mass)
        torch.testing.assert_close(env._backend.link_inertia, before_inertia)
    finally:
        env._backend.close()


def test_mujoco_cpu_applies_friction_per_environment(monkeypatch):
    import mujoco

    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend

    backend = MuJocoCPUBackend()
    backend.setup(_friction_cfg(), num_envs=2, device="cpu", task=None)
    try:
        assert all(data.model is backend._mjm for data in backend._datas)

        _set_explicit_friction(backend)
        active = {"step": [], "forward": [], "rne": []}
        original_step = mujoco.mj_step
        original_forward = mujoco.mj_forward
        original_rne = mujoco.mj_rnePostConstraint

        def record_step(model, data):
            active["step"].append(float(model.geom_friction[0, 0]))
            original_step(model, data)

        def record_forward(model, data):
            active["forward"].append(float(model.geom_friction[0, 0]))
            original_forward(model, data)

        def record_rne(model, data):
            active["rne"].append(float(model.geom_friction[0, 0]))
            original_rne(model, data)

        monkeypatch.setattr(mujoco, "mj_step", record_step)
        monkeypatch.setattr(mujoco, "mj_forward", record_forward)
        monkeypatch.setattr(mujoco, "mj_rnePostConstraint", record_rne)
        reset_mask = _reset_mask(2, "cpu")
        backend.reset_state(reset_mask)
        assert active["forward"] == pytest.approx([0.2, 0.7])
        assert active["rne"] == pytest.approx([0.2, 0.7])
        active["rne"].clear()
        backend.step(torch.zeros(2, backend.num_dof))
        assert active["step"] == pytest.approx([0.2, 0.7])
        assert active["rne"] == pytest.approx([0.2, 0.7])
        assert active["forward"] == pytest.approx([0.2, 0.7])
    finally:
        backend.close()


def test_mujoco_cpu_friction_partial_update_changes_only_selected_value():
    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend

    backend = MuJocoCPUBackend()
    backend.setup(_friction_cfg(), num_envs=3, device="cpu", task=None)
    try:
        backend.set_contact_friction(torch.tensor([1]), torch.tensor([0.4]))
        torch.testing.assert_close(
            backend.contact_friction, torch.tensor([1.0, 0.4, 1.0])
        )
    finally:
        backend.close()


def test_mujoco_cpu_without_dr_rejects_friction_updates():
    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend

    cfg = _friction_cfg()
    cfg.domain_randomization.startup.contact_friction_range = None
    backend = MuJocoCPUBackend()
    backend.setup(cfg, num_envs=2, device="cpu", task=None)
    try:
        assert all(data.model is backend._mjm for data in backend._datas)
        with pytest.raises(RuntimeError, match="was not enabled"):
            backend.set_contact_friction(torch.tensor([0]), torch.tensor([0.5]))
    finally:
        backend.close()


def test_go2_disables_crashing_mujoco_multiccd_pose():
    code = textwrap.dedent(
        """
        import mujoco
        import numpy as np

        from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
        from gym.envs.go2.go2trot_config import Go2TrotCfg

        qpos = np.array([
            53.8714493, 143.839690, 0.151433455,
            0.651931015, 0.474724545, 0.354051379, 0.473571725,
            -0.736025268, 0.0852538293, -1.37272838,
            0.392659753, -0.143433949, -1.06821755,
            -0.119921852, -0.125259463, -1.09467047,
            0.356757622, 0.819555286, -1.06409946,
        ])

        backend = MuJocoCPUBackend()
        backend.setup(Go2TrotCfg(), num_envs=1, device="cpu", task=None)
        model, data = backend._mjm, backend._datas[0]
        assert model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_MULTICCD
        data.qpos[:] = qpos
        mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all()
        backend.close()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_mujoco_cpu_friction_has_predicted_physical_effect():
    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend

    backend = MuJocoCPUBackend()
    backend.setup(_friction_cfg(), num_envs=2, device="cpu", task=None)
    try:
        _assert_friction_changes_motion(backend)
    finally:
        backend.close()


def test_mujoco_cpu_task_randomizes_friction_only_at_startup():
    _assert_task_startup_friction_randomization("cpu")


def test_mujoco_cpu_task_randomizes_pd_gains_in_common_control_path():
    _assert_task_pd_randomization("cpu")


def test_mujoco_cpu_sparse_reset_mask_isolates_task_buffers():
    env = _build_randomized_task(
        "cpu",
        contact_friction=None,
        reset_mode="reset_to_basic",
    )
    try:
        env.dof_pos[:] = torch.arange(env.num_envs).unsqueeze(1) + 10.0
        env.dof_vel[:] = torch.arange(env.num_envs).unsqueeze(1) + 20.0
        env.root_states[:] = torch.arange(env.num_envs).unsqueeze(1) + 30.0
        env.commands[:] = torch.arange(env.num_envs).unsqueeze(1) + 40.0
        env.dof_pos_target[:] = torch.arange(env.num_envs).unsqueeze(1) + 50.0
        env.dof_pos_history[:] = torch.arange(env.num_envs).unsqueeze(1) + 60.0
        env.episode_length_buf[:] = torch.arange(env.num_envs) + 70
        before = {
            name: getattr(env, name).clone()
            for name in (
                "dof_pos",
                "dof_vel",
                "root_states",
                "commands",
                "dof_pos_target",
                "dof_pos_history",
                "episode_length_buf",
            )
        }
        reset_mask = _reset_mask(env.num_envs, env.device, [1, 3])

        env._reset_idx(reset_mask)

        untouched = torch.tensor([0, 2])
        for name, values in before.items():
            torch.testing.assert_close(getattr(env, name)[untouched], values[untouched])
        assert torch.equal(
            env.episode_length_buf,
            torch.tensor([70, 0, 72, 0]),
        )
    finally:
        env._backend.close()


def test_go2trot_sparse_reset_mask_isolates_task_and_gait_buffers():
    from gym.envs.go2.go2trot import Go2Trot
    from gym.envs.go2.go2trot_config import Go2TrotCfg, Go2TrotRunnerCfg
    from gym.utils.task_registry import select_backend, task_registry

    cfg = Go2TrotCfg()
    runner_cfg = Go2TrotRunnerCfg()
    cfg.env.num_envs = 4
    cfg.seed = 17
    cfg.push_robots.toggle = False
    cfg.domain_randomization.startup.contact_friction_range = None
    cfg.domain_randomization.startup.link_mass_scale_range = None
    task_registry.convert_frequencies_to_params(cfg, runner_cfg)
    env = Go2Trot(cfg, "cpu", True, select_backend(cfg, "cpu", "mujoco"))
    try:
        row = torch.arange(env.num_envs).unsqueeze(1)
        env.dof_pos[:] = row + 10.0
        env.dof_vel[:] = row + 20.0
        env.root_states[:] = row + 30.0
        env.commands[:] = row + 40.0
        env.dof_pos_target[:] = row + 50.0
        env.dof_pos_history[:] = row + 60.0
        env.phase[:] = row + 70.0
        env.phase_frequency[:] = row + 80.0
        env.episode_length_buf[:] = torch.arange(env.num_envs) + 90
        p_scale = env.domain_randomizer.episode_scale("p_gains")
        d_scale = env.domain_randomizer.episode_scale("d_gains")
        before = {
            name: getattr(env, name).clone()
            for name in (
                "dof_pos",
                "dof_vel",
                "root_states",
                "commands",
                "dof_pos_target",
                "dof_pos_history",
                "phase",
                "phase_frequency",
                "episode_length_buf",
            )
        }
        before_p = p_scale.clone()
        before_d = d_scale.clone()

        env._reset_idx(_reset_mask(env.num_envs, env.device, [1, 3]))

        untouched = torch.tensor([0, 2])
        for name, values in before.items():
            torch.testing.assert_close(getattr(env, name)[untouched], values[untouched])
        torch.testing.assert_close(p_scale[untouched], before_p[untouched])
        torch.testing.assert_close(d_scale[untouched], before_d[untouched])
        assert torch.equal(env.episode_length_buf, torch.tensor([90, 0, 92, 0]))
    finally:
        env._backend.close()


def test_mujoco_cpu_link_mass_and_inertia_change_acceleration():
    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend

    backend = MuJocoCPUBackend()
    backend.setup(_link_mass_cfg(), num_envs=2, device="cpu", task=None)
    try:
        assert backend._models is not None
        assert all(
            data.model is model for data, model in zip(backend._datas, backend._models)
        )
        _assert_link_mass_changes_acceleration(backend)
    finally:
        backend.close()


def test_mujoco_cpu_task_randomizes_link_mass_only_at_startup():
    _assert_task_link_mass_randomization("cpu")


@pytest.mark.warp
def test_mujoco_warp_applies_and_consumes_friction_per_world():
    if not torch.cuda.is_available():
        pytest.fail("Warp tests requested but CUDA is not available", pytrace=False)
    from gym.envs.base.mujoco_warp_backend import MuJocoWarpBackend

    backend = MuJocoWarpBackend()
    backend.setup(_friction_cfg(), num_envs=2, device="cuda:0", task=None)
    try:
        _set_explicit_friction(backend)
        expected = backend.contact_friction[:, None].expand_as(
            backend._geom_friction_t[:, :, 0]
        )
        torch.testing.assert_close(backend._geom_friction_t[:, :, 0], expected)
        _assert_friction_changes_motion(backend)
    finally:
        backend.close()


@pytest.mark.warp
def test_mujoco_warp_task_randomizes_friction_only_at_startup():
    if not torch.cuda.is_available():
        pytest.fail("Warp tests requested but CUDA is not available", pytrace=False)
    _assert_task_startup_friction_randomization("cuda:0")


@pytest.mark.warp
def test_mujoco_warp_task_randomizes_pd_gains_in_common_control_path():
    if not torch.cuda.is_available():
        pytest.fail("Warp tests requested but CUDA is not available", pytrace=False)
    _assert_task_pd_randomization("cuda:0")


@pytest.mark.warp
def test_mujoco_warp_link_mass_and_inertia_change_acceleration():
    if not torch.cuda.is_available():
        pytest.fail("Warp tests requested but CUDA is not available", pytrace=False)
    from gym.envs.base.mujoco_warp_backend import MuJocoWarpBackend

    backend = MuJocoWarpBackend()
    backend.setup(_link_mass_cfg(), num_envs=2, device="cuda:0", task=None)
    try:
        assert backend._m.body_mass.shape[0] == 2
        assert backend._m.body_inertia.shape[0] == 2
        assert backend._m.body_subtreemass.shape[0] == 2
        assert backend._m.body_invweight0.shape[0] == 2
        assert backend._m.dof_invweight0.shape[0] == 2
        _assert_link_mass_changes_acceleration(backend)
    finally:
        backend.close()


@pytest.mark.warp
def test_mujoco_warp_task_randomizes_link_mass_only_at_startup():
    if not torch.cuda.is_available():
        pytest.fail("Warp tests requested but CUDA is not available", pytrace=False)
    _assert_task_link_mass_randomization("cuda:0")


@pytest.mark.vsim
def test_vsim_applies_and_consumes_friction_per_environment_set():
    vsim_guard()
    from gym.envs.base.vsim_backend import VSimBackend

    backend = VSimBackend()
    backend.setup(_friction_cfg(), num_envs=2, device="cuda:0", task=None)
    try:
        assert backend._grp.get_num_environment_sets() == 2
        assert backend._grp.get_num_environments() == [1, 1]
        _set_explicit_friction(backend)

        backend._property_mask.fill_(True)
        backend._static_friction.fill_(float("nan"))
        backend._dynamic_friction.fill_(float("nan"))
        backend._gym.get_rigid_material_properties(backend._friction_set_arr)
        torch.testing.assert_close(
            backend._static_friction.cpu(), backend.contact_friction.cpu()
        )
        torch.testing.assert_close(
            backend._dynamic_friction.cpu(), backend.contact_friction.cpu()
        )
        _assert_friction_changes_motion(backend)
    finally:
        backend.close()


@pytest.mark.vsim
def test_vsim_task_randomizes_friction_only_at_startup():
    vsim_guard()
    _assert_task_startup_friction_randomization("cuda:0", "vsim")


@pytest.mark.vsim
def test_vsim_task_randomizes_pd_gains_without_changing_set_topology():
    vsim_guard()
    env = _build_randomized_task(
        "cuda:0",
        "vsim",
        contact_friction=None,
        stiffness=(0.8, 1.2),
        damping=(0.7, 1.3),
    )
    try:
        assert env._backend._grp.get_num_environment_sets() == 1
        assert env._backend._grp.get_num_environments() == [4]
        assert env.domain_randomizer.episode_scale("p_gains") is not None
        assert env.domain_randomizer.episode_scale("d_gains") is not None
    finally:
        env._backend.close()


@pytest.mark.vsim
def test_vsim_link_mass_and_inertia_change_acceleration():
    vsim_guard()
    from gym.envs.base.vsim_backend import VSimBackend

    backend = VSimBackend()
    backend.setup(_link_mass_cfg(), num_envs=2, device="cuda:0", task=None)
    try:
        assert backend._grp.get_num_environment_sets() == 2
        assert backend._grp.get_num_environments() == [1, 1]
        _assert_link_mass_changes_acceleration(backend)

        backend._link_property_mask.fill_(True)
        backend._link_mass_native.fill_(float("nan"))
        backend._link_inertia_native.fill_(float("nan"))
        backend._gym.get_link_properties(backend._link_property_set_arr)
        body_ids = backend._canonical_to_native_body
        torch.testing.assert_close(
            backend._link_mass_native[:, body_ids], backend.link_mass
        )
        torch.testing.assert_close(
            backend._link_inertia_native[:, body_ids], backend.link_inertia
        )
    finally:
        backend.close()


@pytest.mark.vsim
def test_vsim_task_randomizes_link_mass_only_at_startup():
    vsim_guard()
    _assert_task_link_mass_randomization("cuda:0", "vsim")
