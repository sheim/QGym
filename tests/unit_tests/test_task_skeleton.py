"""Tests for TaskSkeleton.

Pure PyTorch — no physics engine, no backend.  Tests the get_states /
set_states / scaling interface that the RL runner relies on.
"""

import pytest
import torch

from gym.envs.base.task_skeleton import TaskSkeleton


class _SimpleTask(TaskSkeleton):
    """Minimal concrete subclass for testing."""

    def __init__(self, num_envs: int = 4, device: str = "cpu"):
        super().__init__(num_envs=num_envs, device=device)
        # Pretend these are physics-backed observation tensors
        self.obs_a = torch.ones(num_envs, 3, device=device)
        self.obs_b = torch.full((num_envs, 2), 2.0, device=device)
        self.action = torch.zeros(num_envs, 2, device=device)
        # scales dict (normally populated by _parse_cfg → class_to_dict)
        self.scales = {"obs_a": 2.0, "action": 5.0}

    def step(self):
        pass

    def _reset_idx(self, env_ids):
        pass


# ── Construction ─────────────────────────────────────────────────────────────


class TestConstruction:
    def test_episode_buffer_zeros(self):
        task = _SimpleTask(num_envs=8)
        assert task.episode_length_buf.sum() == 0

    def test_to_be_reset_all_true_initially(self):
        task = _SimpleTask(num_envs=8)
        assert task.to_be_reset.all()

    def test_device_stored(self):
        task = _SimpleTask(device="cpu")
        assert task.device == "cpu"


# ── get_state / get_states ───────────────────────────────────────────────────


class TestGetState:
    def test_get_state_no_scale(self):
        task = _SimpleTask()
        result = task.get_state("obs_b")
        assert torch.allclose(result, task.obs_b)

    def test_get_state_with_scale(self):
        task = _SimpleTask()
        result = task.get_state("obs_a")
        assert torch.allclose(result, task.obs_a / 2.0)

    def test_get_states_concatenation(self):
        task = _SimpleTask()
        result = task.get_states(["obs_a", "obs_b"])
        expected = torch.cat([task.obs_a / 2.0, task.obs_b], dim=-1)
        assert torch.allclose(result, expected)
        assert result.shape == (4, 5)

    def test_get_states_single(self):
        task = _SimpleTask()
        result = task.get_states(["obs_b"])
        assert torch.allclose(result, task.obs_b)


# ── set_state / set_states ───────────────────────────────────────────────────


class TestSetState:
    def test_set_state_no_scale(self):
        task = _SimpleTask()
        new_val = torch.full((4, 2), 7.0)
        task.set_state("obs_b", new_val)
        assert torch.allclose(task.obs_b, new_val)

    def test_set_state_with_scale(self):
        task = _SimpleTask()
        # action scale is 5.0; setting value 1.0 should store 5.0 internally
        task.set_state("action", torch.ones(4, 2))
        assert torch.allclose(task.action, torch.full((4, 2), 5.0))

    def test_set_states_distributes_correctly(self):
        task = _SimpleTask()
        # actions for both obs_b (dim=2) and action (dim=2)
        values = torch.arange(16, dtype=torch.float).view(4, 4)
        task.set_states(["obs_b", "action"], values)
        # obs_b has no scale → stored as-is
        assert torch.allclose(task.obs_b, values[:, :2])
        # action has scale 5.0 → stored * 5
        assert torch.allclose(task.action, values[:, 2:] * 5.0)

    def test_set_states_wrong_total_dim_raises(self):
        task = _SimpleTask()
        bad_values = torch.zeros(4, 3)  # too short
        with pytest.raises(AssertionError):
            task.set_states(["obs_b", "action"], bad_values)


# ── _reset_buffers ────────────────────────────────────────────────────────────


class TestResetBuffers:
    def test_clears_flags(self):
        task = _SimpleTask()
        task.to_be_reset[:] = True
        task.terminated[:] = True
        task.timed_out[:] = True
        task._reset_buffers()
        assert not task.to_be_reset.any()
        assert not task.terminated.any()
        assert not task.timed_out.any()


# ── parse_cfg scaling ─────────────────────────────────────────────────────────


class TestScaling:
    def test_get_and_set_roundtrip(self):
        """get_state then set_state should be identity via the scale."""
        task = _SimpleTask()
        original = task.obs_a.clone()
        scaled = task.get_state("obs_a")  # divides by 2
        task.set_state("obs_a", scaled)  # multiplies by 2
        assert torch.allclose(task.obs_a, original, atol=1e-6)
