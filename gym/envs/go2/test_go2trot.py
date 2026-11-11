from types import SimpleNamespace

import torch
import pytest

from gym.envs.go2.go2trot import Go2Trot
from gym.envs.go2.go2trot_config import Go2TrotCfg


def test_contact_strength_uses_each_environments_applied_total_mass():
    task = Go2Trot.__new__(Go2Trot)
    task.feet_indices = torch.arange(4)
    task.contact_forces = torch.zeros(2, 4, 3)
    task.contact_forces[:, :, 2] = 98.1
    task._backend = SimpleNamespace(link_mass=torch.tensor([[5.0, 5.0], [10.0, 10.0]]))

    strength = task._foot_contact_strength()

    torch.testing.assert_close(strength[0], torch.ones(4))
    torch.testing.assert_close(strength[1], torch.full((4,), 0.5))


@pytest.mark.parametrize(
    "targets, expected",
    [
        ([0.2, -0.2, 0.2], -0.16),  # Alternation remains visible.
        ([0.4, 0.2, 0.0], -0.08),  # Equal contributions for a constant rate.
        ([0.2, 0.2, 0.2], 0.0),  # Constant targets have no cost.
    ],
)
def test_action_rate_sums_one_step_and_normalized_two_step_costs(targets, expected):
    task = Go2Trot.__new__(Go2Trot)
    task.num_actuators = 12
    task.dof_pos_history = torch.tensor([[q for q in targets for _ in range(12)]])
    assert task._reward_action_rate().item() == pytest.approx(expected)


def test_yaw_tracking_uses_velocity_error_not_its_square():
    task = Go2Trot.__new__(Go2Trot)
    task.cfg = Go2TrotCfg()
    task.commands = torch.zeros(3, 3)
    task.base_ang_vel = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, -0.5]]
    )
    # Keep the existing 2.5 rad/s error scale; square the error only once.
    torch.testing.assert_close(
        task._reward_tracking_ang_vel(),
        torch.tensor(
            [1.0, torch.exp(torch.tensor(-0.16)), torch.exp(torch.tensor(-0.16))]
        ),
    )


def test_reference_reward_tracks_measured_joints_with_normalized_symmetric_error():
    task = Go2Trot.__new__(Go2Trot)
    task.cfg = Go2TrotCfg()
    # Include a passive joint and permute actuator order.
    task.actuated_dof_indices = torch.tensor([2, 0])
    task.default_dof_pos = torch.tensor([[0.4, 10.0, -0.6]])
    task.gait_reference = torch.tensor([[0.2, -0.1]])
    task.scales = {"dof_pos": torch.tensor([2.0, 1.0, 0.5])}
    task.dof_pos = torch.tensor([[0.3, -99.0, -0.4]])
    assert task._reward_reference_traj().item() == pytest.approx(1.0)
    for sign in [-1, 1]:
        task.dof_pos[0, 0] = 0.3 + sign * 1.0
        task.dof_pos[0, 2] = -0.4 + sign * 0.25
        expected = (
            torch.exp(torch.tensor(-0.25 / task.cfg.reward_settings.tracking_sigma))
            - 0.1
        )
        assert task._reward_reference_traj().item() == pytest.approx(expected.item())


def _state_task(num_envs=2):
    """Minimal Go2Trot surface used by capture_state/restore_state."""
    task = Go2Trot.__new__(Go2Trot)
    task.num_envs = num_envs
    task.device = "cpu"
    task.num_dof = 2
    task.dof_pos = torch.tensor([[0.1, -0.2], [0.3, -0.4]])
    task.dof_vel = torch.tensor([[0.5, 0.6], [0.7, 0.8]])
    task.dof_pos_target = torch.tensor([[0.2, -0.1], [0.4, -0.3]])
    task.root_states = torch.tensor(
        [
            [0.0, 0.0, 0.32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.30, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    task.commands = torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.0, 0.1]])
    task.phase = torch.tensor([[0.5], [1.5]])
    task.phase_frequency = torch.tensor([[2.0], [1.5]])
    task.phase_obs = torch.zeros(num_envs, 2)
    task.episode_length_buf = torch.tensor([11, 22])
    task.common_step_counter = 33
    task.base_quat = task.root_states[:, 3:7].clone()
    task.base_lin_vel = torch.zeros(num_envs, 3)
    task.base_ang_vel = torch.zeros(num_envs, 3)
    task.projected_gravity = torch.zeros(num_envs, 3)
    task.gravity_vec = torch.tensor([[0.0, 0.0, -1.0]]).expand(num_envs, 3).clone()
    task.default_dof_pos = torch.zeros(task.num_dof)
    task.dof_pos_obs = task.dof_pos - task.default_dof_pos
    task.gait_reference = torch.zeros(num_envs, task.num_dof)
    task._gait_dof_phase_offsets = torch.zeros(task.num_dof)
    task._gait_joint_offsets = torch.zeros(task.num_dof)
    task._gait_joint_amplitudes = torch.ones(task.num_dof)
    task.to_be_reset = torch.ones(num_envs, dtype=torch.bool)
    task.terminated = torch.ones(num_envs, dtype=torch.bool)
    task.timed_out = torch.ones(num_envs, dtype=torch.bool)
    task.reset_masks = []
    task._backend = SimpleNamespace(reset_state=task.reset_masks.append)
    return task


def test_capture_state_clones_and_restore_state_writes_back():
    task = _state_task()
    snapshot = task.capture_state()
    # A snapshot must not alias live buffers.
    task.dof_pos[:] = 9.0
    task.phase[:] = 9.0
    torch.testing.assert_close(
        snapshot["dof_pos"], torch.tensor([[0.1, -0.2], [0.3, -0.4]])
    )
    torch.testing.assert_close(snapshot["phase"], torch.tensor([[0.5], [1.5]]))

    task.restore_state(snapshot)

    torch.testing.assert_close(task.dof_pos, snapshot["dof_pos"])
    torch.testing.assert_close(task.dof_vel, snapshot["dof_vel"])
    torch.testing.assert_close(task.root_states, snapshot["root_states"])
    torch.testing.assert_close(task.commands, snapshot["commands"])
    torch.testing.assert_close(task.phase, snapshot["phase"])
    torch.testing.assert_close(task.episode_length_buf, snapshot["episode_length_buf"])
    assert task.common_step_counter == 33
    # The backend commit must run exactly once for all worlds.
    assert len(task.reset_masks) == 1
    assert task.reset_masks[0].all()
    # Derived buffers are recomputed from the restored state.
    torch.testing.assert_close(task.dof_pos_obs, task.dof_pos - task.default_dof_pos)
    torch.testing.assert_close(task.phase_obs[:, 0], torch.sin(snapshot["phase"][:, 0]))
    torch.testing.assert_close(task.phase_obs[:, 1], torch.cos(snapshot["phase"][:, 0]))
    # Per-episode flags must be cleared, or the next step reports a stale fall.
    assert not task.terminated.any()
    assert not task.timed_out.any()
    assert not task.to_be_reset.any()
