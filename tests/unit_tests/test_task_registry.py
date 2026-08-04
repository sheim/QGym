"""The declared task manifest must register completely and exactly."""

import pytest
import torch

import gym.envs as envs
from gym.utils.task_registry import task_registry


UNSUPPORTED_TERRAIN_FIELDS = {
    "border_size",
    "curriculum",
    "horizontal_scale",
    "max_init_terrain_level",
    "measure_heights",
    "measured_points_x",
    "measured_points_y",
    "num_cols",
    "num_rows",
    "selected",
    "slope_treshold",
    "terrain_kwargs",
    "terrain_length",
    "terrain_proportions",
    "terrain_width",
    "vertical_scale",
}
LEGACY_ASSET_FIELDS = {
    "angular_damping",
    "armature",
    "collapse_fixed_joints",
    "default_dof_drive_mode",
    "density",
    "flip_visual_attachments",
    "linear_damping",
    "max_angular_velocity",
    "max_linear_velocity",
    "replace_cylinder_with_capsule",
    "self_collisions",
    "thickness",
}


def test_declared_tasks_register_with_declared_components():
    expected_names = set(envs.task_dict)

    assert set(task_registry.task_classes) == expected_names
    assert set(task_registry.env_cfgs) == expected_names
    assert set(task_registry.train_cfgs) == expected_names

    for task_name, component_names in envs.task_dict.items():
        class_name, config_name, runner_config_name = component_names
        assert task_registry.task_classes[task_name] is getattr(envs, class_name)
        assert isinstance(task_registry.env_cfgs[task_name], getattr(envs, config_name))
        assert isinstance(
            task_registry.train_cfgs[task_name],
            getattr(envs, runner_config_name),
        )


def test_declared_tasks_only_expose_supported_physics_config():
    for task_name, registered_cfg in task_registry.env_cfgs.items():
        cfg = type(registered_cfg)()
        assert cfg.terrain.mesh_type in (None, "plane"), task_name

        terrain_fields = UNSUPPORTED_TERRAIN_FIELDS.intersection(dir(cfg.terrain))
        assert not terrain_fields, f"{task_name}: {sorted(terrain_fields)}"

        asset_fields = LEGACY_ASSET_FIELDS.intersection(dir(cfg.asset))
        assert not asset_fields, f"{task_name}: {sorted(asset_fields)}"


@pytest.mark.parametrize("task_name", envs.task_dict)
def test_declared_task_runs_one_action_step_on_mujoco_cpu(task_name):
    registered_env_cfg, registered_train_cfg = task_registry.get_cfgs(task_name)
    env_cfg = type(registered_env_cfg)()
    train_cfg = type(registered_train_cfg)()
    env_cfg.env.num_envs = 2
    env_cfg.seed = 1

    # Cross-backend randomization and disturbances are separate features. This
    # smoke test only exercises the deterministic task/backend contract.
    if hasattr(env_cfg, "push_robots"):
        env_cfg.push_robots.toggle = False
    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    env = task_registry.make_env(
        task_name,
        env_cfg,
        device="cpu",
        headless=True,
    )

    try:
        action_width = sum(
            getattr(env, action_name).shape[1]
            for action_name in train_cfg.actor.actions
        )
        actions = torch.zeros(env.num_envs, action_width, device=env.device)
        env.set_states(train_cfg.actor.actions, actions)
        env.step()

        actor_obs = env.get_states(train_cfg.actor.obs)
        critic_obs = env.get_states(train_cfg.critic.obs)
        assert actor_obs.shape[0] == env.num_envs
        assert critic_obs.shape[0] == env.num_envs
        assert torch.isfinite(actor_obs).all()
        assert torch.isfinite(critic_obs).all()
        assert torch.isfinite(env.dof_state).all()
        assert torch.isfinite(env.torques).all()
    finally:
        env._backend.close()
