import pytest
import torch

from gym.envs.base.legged_robot import LeggedRobot
from gym.envs.mini_cheetah.mini_cheetah import MiniCheetah
from gym.envs.mini_cheetah.mini_cheetah_ref import MiniCheetahRef
from gym.envs.mini_cheetah.mini_cheetah_ref_config import MiniCheetahRefRunnerCfg
from gym.envs.mini_cheetah.mini_cheetah_ref_vsim_config import (
    MiniCheetahRefVSimCfg,
    MiniCheetahRefVSimRunnerCfg,
)


def test_vsim_tuning_rollout_spans_multiple_gait_cycles():
    cfg = MiniCheetahRefVSimCfg()
    runner_cfg = MiniCheetahRefVSimRunnerCfg()

    policy_steps = runner_cfg.algorithm.batch_size // cfg.env.num_envs
    rollout_seconds = policy_steps / runner_cfg.actor.frequency
    gait_cycles = rollout_seconds * cfg.control.gait_freq

    assert runner_cfg.algorithm.batch_size == 2**15
    assert policy_steps == 128
    assert gait_cycles == pytest.approx(3.2)


def test_vsim_tuning_rewards_target_deployment_failures():
    cfg = MiniCheetahRefVSimCfg()
    runner_cfg = MiniCheetahRefVSimRunnerCfg()
    weights = runner_cfg.critic.reward.weights

    assert cfg.commands.ranges.lin_vel_x == [-0.5, 1.5]
    assert cfg.commands.ranges.lin_vel_y == 0.4
    assert cfg.commands.ranges.yaw_vel == 0.75
    assert cfg.commands.axis_aligned_fraction == 0.6
    assert cfg.reward_settings.soft_dof_pos_limit == 0.85
    assert cfg.reward_settings.tracking_ang_vel_scale == 1.0
    assert weights.tracking_lin_vel == 6.0
    assert weights.tracking_ang_vel == 4.0
    assert weights.stance_grf > MiniCheetahRefRunnerCfg.critic.reward.weights.stance_grf
    assert weights.action_rate2 > 0
    assert weights.dof_pos_target_limits > 0
    assert weights.torque_limits > 0


def test_position_target_limit_reward_is_zero_inside_and_normalized_outside():
    class RewardFixture:
        actuated_dof_indices = torch.arange(2)
        dof_pos_limits = torch.tensor([[-1.0, 1.0], [-2.0, 2.0]])
        default_dof_pos = torch.tensor([[0.25, -0.5]])
        dof_pos_target = torch.zeros(3, 2)

        class cfg:
            class reward_settings:
                soft_dof_pos_limit = 0.8

    task = RewardFixture()
    indices = task.actuated_dof_indices
    limits = task.dof_pos_limits.index_select(0, indices)
    center = 0.5 * (limits[:, 0] + limits[:, 1])
    default = task.default_dof_pos.index_select(1, indices)

    task.dof_pos_target[:] = center - default
    assert torch.allclose(
        LeggedRobot._reward_dof_pos_target_limits(task),
        torch.zeros(3),
    )

    task.dof_pos_target[:] = limits[:, 1] - default
    expected = torch.full((3,), -0.1)
    assert torch.allclose(
        LeggedRobot._reward_dof_pos_target_limits(task),
        expected,
        atol=1e-6,
    )


def test_mini_cheetah_yaw_tracking_uses_squared_not_fourth_power_error():
    class RewardFixture:
        commands = torch.tensor([[0.0, 0.0, 1.0]])
        base_ang_vel = torch.tensor([[0.0, 0.0, 0.0]])

        class cfg:
            class reward_settings:
                tracking_sigma = 0.25

        _sqrdexp = LeggedRobot._sqrdexp

    reward = MiniCheetah._reward_tracking_ang_vel(RewardFixture())
    expected = torch.exp(torch.tensor(-((1.0 / 2.5) ** 2) / 0.25))

    assert torch.allclose(reward, expected.expand_as(reward))


def test_vsim_tuning_samples_axis_aligned_commands():
    num_envs = 4096
    task = MiniCheetahRef.__new__(MiniCheetahRef)
    task.cfg = MiniCheetahRefVSimCfg()
    task.device = "cpu"
    task.commands = torch.zeros(num_envs, 3)
    task.command_ranges = {
        "lin_vel_x": task.cfg.commands.ranges.lin_vel_x,
        "lin_vel_y": task.cfg.commands.ranges.lin_vel_y,
        "yaw_vel": task.cfg.commands.ranges.yaw_vel,
    }
    env_ids = torch.arange(num_envs)

    torch.manual_seed(7)
    task._resample_commands(env_ids)

    active_axes = torch.count_nonzero(task.commands, dim=1)
    pure_lateral = (active_axes == 1) & (task.commands[:, 1] != 0)

    assert (active_axes == 1).float().mean() > 0.45
    assert torch.any(task.commands[pure_lateral, 1] > 0)
    assert torch.any(task.commands[pure_lateral, 1] < 0)
