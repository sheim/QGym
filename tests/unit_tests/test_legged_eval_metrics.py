from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gym.utils.legged_eval_metrics import (
    GO2_COMMAND_CASES,
    HARDWARE_COMMAND_CASES,
    LeggedMetricAccumulator,
    actuated_position_target,
    apply_command_profile,
    summarize_metrics,
    velocity_impulse_schedule,
)


def _fake_env(num_envs=2):
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        dt=0.1,
        commands=torch.zeros(num_envs, 3),
        base_lin_vel=torch.zeros(num_envs, 3),
        base_ang_vel=torch.zeros(num_envs, 3),
        base_height=torch.full((num_envs, 1), 0.3),
        projected_gravity=torch.tensor([[0.0, 0.0, -1.0]]).repeat(num_envs, 1),
        actuated_dof_indices=torch.tensor([0, 1]),
        dof_pos=torch.zeros(num_envs, 2),
        dof_vel=torch.zeros(num_envs, 2),
        dof_pos_target=torch.zeros(num_envs, 2),
        torques=torch.tensor([[5.0, 10.0]]).repeat(num_envs, 1),
        actuated_torque_limits=torch.tensor([10.0, 10.0]),
        dof_vel_limits=torch.tensor([2.0, 2.0]),
        dof_pos_limits=torch.tensor([[-1.0, 1.0], [-2.0, 2.0]]),
        default_dof_pos=torch.zeros(1, 2),
        feet_indices=torch.tensor([1]),
        penalised_contact_indices=torch.tensor([0]),
        contact_forces=torch.zeros(num_envs, 2, 3),
        _rigid_body_lin_vel=torch.zeros(num_envs, 2, 3),
        cfg=SimpleNamespace(reward_settings=SimpleNamespace(base_height_target=0.3)),
    )
    env.commands[:, 0] = 1.0
    env.contact_forces[0, 0, 2] = 25.0
    env.contact_forces[0, 1, 2] = 30.0
    env._rigid_body_lin_vel[0, 1, 0] = 0.1
    env._get_ref = lambda: torch.zeros(num_envs, 2)
    env._leg_phases = lambda: torch.tensor([[4.0], [2.0]]).repeat(
        (num_envs + 1) // 2, 1
    )[:num_envs]
    env._update_cmd_switch = lambda: None
    return env


def test_hardware_command_profile_repeats_named_cases():
    env = _fake_env(num_envs=len(HARDWARE_COMMAND_CASES) + 2)

    labels = apply_command_profile(env, "hardware")

    assert labels[0] == "stand"
    assert labels[len(HARDWARE_COMMAND_CASES)] == "stand"
    np.testing.assert_allclose(
        env.commands[1].numpy(),
        HARDWARE_COMMAND_CASES[1][1],
    )


def test_go2_command_profile_includes_training_speed_extreme():
    env = _fake_env(num_envs=len(GO2_COMMAND_CASES))

    labels = apply_command_profile(env, "go2")

    forward_3p0 = np.flatnonzero(labels == "forward_3p0")
    assert forward_3p0.tolist() == [4]
    np.testing.assert_allclose(env.commands[forward_3p0[0]].numpy(), [3.0, 0.0, 0.0])


def test_commanded_target_includes_base_gait_and_policy_residual():
    env = _fake_env(num_envs=1)
    env.default_dof_pos[:] = torch.tensor([0.2, -0.4])
    env.gait_reference = torch.tensor([[0.1, -0.2]])
    env.dof_pos_target[:] = torch.tensor([[0.05, 0.07]])

    target = actuated_position_target(env)

    torch.testing.assert_close(target, torch.tensor([[0.35, -0.53]]))


def test_accumulator_reports_physical_and_contact_metrics_per_environment():
    env = _fake_env()
    accumulator = LeggedMetricAccumulator(
        env,
        settle_steps=0,
        contact_threshold=20.0,
    )
    alive = torch.ones(2, dtype=torch.bool)

    accumulator.update(0, alive)
    env.dof_pos_target += 0.1
    env.dof_vel += 1.0
    env.base_lin_vel[:, 2] += 0.1
    accumulator.update(1, alive)
    env.dof_pos_target += 0.1
    accumulator.update(2, alive)
    metrics = accumulator.finalize()

    np.testing.assert_allclose(metrics["tracking_vx_rmse"], [1.0, 1.0])
    np.testing.assert_allclose(metrics["foot_contact_phase_match"], [1.0, 1.0])
    np.testing.assert_allclose(metrics["unsafe_contact_fraction"], [1.0, 0.0])
    assert metrics["foot_slip_speed_rms"][0] == pytest.approx(0.1 / np.sqrt(2))
    assert np.isnan(metrics["foot_slip_speed_rms"][1])
    np.testing.assert_allclose(metrics["target_velocity_rms"], [1.0, 1.0])
    np.testing.assert_allclose(metrics["target_acceleration_rms"], [0.0, 0.0])
    assert metrics["torque_utilization_peak"][0] == pytest.approx(1.0)
    assert metrics["torque_saturation_fraction"][0] == pytest.approx(0.5)
    assert metrics["target_joint_limit_margin_min"][0] > 0


def test_accumulator_retains_fixed_evaluation_commands_after_task_reset():
    env = _fake_env(num_envs=1)
    accumulator = LeggedMetricAccumulator(env, settle_steps=0)
    env.commands.zero_()

    accumulator.update(0, torch.ones(1, dtype=torch.bool))
    metrics = accumulator.finalize()

    np.testing.assert_allclose(metrics["tracking_vx_rmse"], [1.0])


def test_accumulator_uses_task_stance_convention():
    env = _fake_env(num_envs=1)
    env._leg_phases = lambda: torch.tensor([[1.0]])
    env._expected_stance = lambda: torch.tensor([[True]])
    accumulator = LeggedMetricAccumulator(env, settle_steps=0)

    accumulator.update(0, torch.ones(1, dtype=torch.bool))
    metrics = accumulator.finalize()

    np.testing.assert_allclose(metrics["foot_contact_phase_match"], [1.0])


def test_summary_splits_metrics_by_command_case():
    metrics = {"tracking_vx_rmse": np.array([0.1, 0.3, 0.5])}
    cases = np.array(["stand", "forward", "forward"])

    summary = summarize_metrics(metrics, cases)

    assert summary["overall"]["tracking_vx_rmse"]["mean"] == pytest.approx(0.3)
    assert summary["stand"]["tracking_vx_rmse"]["mean"] == pytest.approx(0.1)
    assert summary["forward"]["tracking_vx_rmse"]["median"] == pytest.approx(0.4)
    assert summary["overall"]["tracking_vx_rmse"]["p10"] == pytest.approx(0.14)


def test_velocity_impulses_cover_directions_and_stagger_steps():
    steps, angles, delta = velocity_impulse_schedule(
        num_envs=72,
        sample_rate_hz=100.0,
        start_time_s=5.0,
        stagger_time_s=0.5,
        magnitude=2.0,
        num_directions=36,
    )

    assert np.unique(angles).size == 36
    np.testing.assert_array_equal(steps[:36], np.full(36, 500))
    np.testing.assert_array_equal(steps[36:], np.full(36, 549))
    np.testing.assert_allclose(np.linalg.norm(delta, axis=1), 2.0)
