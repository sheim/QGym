from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gym.utils.policy_io import (
    component_scales_from_names,
    first_episode_mask,
    phase_binned_stats,
    policy_io_in_space,
    state_component_names,
    state_component_scales,
)


def test_component_names_preserve_axes_joints_and_history_slots():
    env = SimpleNamespace(
        actuated_dof_names=["hip", "knee"],
        dof_names=["hip", "knee"],
        commands=torch.zeros(3, 3),
        dof_pos=torch.zeros(3, 2),
        dof_pos_history=torch.zeros(3, 6),
        phase_obs=torch.zeros(3, 2),
    )

    names = state_component_names(
        env,
        ["commands", "dof_pos", "dof_pos_history", "phase_obs"],
    )

    assert names == [
        "commands.vx",
        "commands.vy",
        "commands.yaw",
        "dof_pos.hip",
        "dof_pos.knee",
        "dof_pos_history.history_0.hip",
        "dof_pos_history.history_0.knee",
        "dof_pos_history.history_1.hip",
        "dof_pos_history.history_1.knee",
        "dof_pos_history.history_2.hip",
        "dof_pos_history.history_2.knee",
        "phase_obs.sin",
        "phase_obs.cos",
    ]


def test_component_scales_expand_scalar_vector_and_unscaled_fields():
    env = SimpleNamespace(
        base_height=torch.zeros(2, 1),
        commands=torch.zeros(2, 3),
        phase_obs=torch.zeros(2, 2),
    )
    fields = ["base_height", "commands", "phase_obs"]
    scales = {"base_height": 0.4, "commands": [3.0, 1.0, 2.0]}

    expanded = state_component_scales(env, fields, scales)

    assert expanded == pytest.approx([0.4, 3.0, 1.0, 2.0, 1.0, 1.0])


def test_component_scales_can_be_recovered_from_artifact_names():
    expanded = component_scales_from_names(
        ["commands", "phase_obs"],
        [
            "commands.vx",
            "commands.vy",
            "commands.yaw",
            "phase_obs.sin",
            "phase_obs.cos",
        ],
        {"commands": [3.0, 1.0, 2.0]},
    )

    assert expanded == pytest.approx([3.0, 1.0, 2.0, 1.0, 1.0])


def test_policy_io_space_conversion_handles_observations_and_applied_actions():
    normalized = np.array([[1.0, -2.0]], dtype=np.float32)
    scales = np.array([0.4, 3.0], dtype=np.float32)

    unnormalized = policy_io_in_space(normalized, scales, "normalized", "unnormalized")
    restored = policy_io_in_space(unnormalized, scales, "unnormalized", "normalized")

    np.testing.assert_allclose(unnormalized, [[0.4, -6.0]])
    np.testing.assert_allclose(restored, normalized)


def test_first_episode_mask_excludes_first_termination_and_later_samples():
    terminated = np.array(
        [
            [False, False],
            [False, True],
            [True, False],
            [False, False],
        ]
    )

    np.testing.assert_array_equal(
        first_episode_mask(terminated),
        [
            [True, True],
            [True, False],
            [False, False],
            [False, False],
        ],
    )


def test_phase_binned_stats_aggregate_valid_time_and_environment_samples():
    phase = np.array([[0.1, 0.2], [3.2, 3.3]])
    values = np.array([[1.0, 3.0], [10.0, 14.0]])
    valid = np.array([[True, True], [True, False]])

    centers, means, stds, counts = phase_binned_stats(phase, values, valid, num_bins=2)

    np.testing.assert_allclose(centers, [np.pi / 2, 3 * np.pi / 2])
    np.testing.assert_allclose(means, [2.0, 10.0])
    np.testing.assert_allclose(stds, [1.0, 0.0])
    np.testing.assert_array_equal(counts, [2, 1])


def test_phase_binned_stats_requires_matching_shapes():
    with pytest.raises(ValueError, match="same"):
        phase_binned_stats(
            np.zeros((2, 1)),
            np.zeros((2, 2)),
            np.ones((2, 2), dtype=bool),
        )
