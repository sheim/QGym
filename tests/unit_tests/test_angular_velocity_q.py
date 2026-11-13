"""Portable contracts for the yaw angular-velocity reward, Q artifact and returns."""

import json

import pytest
import torch

from learning.algorithms.fqe import FQE, QNetwork
from learning.offline.contracts import (
    ANGULAR_VEL_Q_FORMAT,
    ZERO_AFTER_TERMINAL_RETURN_CONVENTION,
    AngularVelocityRewardSpec,
    PolicySpec,
    TrainingSpec,
)
from learning.offline.dataset import TransitionDataset, episode_split
from learning.offline.fixed_state_variance import (
    accumulate_return,
    accumulate_return_zero_after_terminal,
)
from learning.offline.q_artifact import (
    AngularVelocityQ,
    HeightQ,
    save_angular_velocity_q,
)
from learning.offline.rewards import (
    angular_velocity_reward,
    observation_field_slice,
    relabel_angular_velocity,
)
from learning.offline.training import _bellman_metrics

OBS_NAMES = ("base_ang_vel", "projected_gravity", "commands", "dof_pos_obs")
OBS_DIMS = (3, 3, 3, 4)
OBS_DIM = sum(OBS_DIMS)


class FixedPolicy:
    def __init__(self, action_dim=2):
        self.action_dim = action_dim

    def sample_actions(self, observations, generator):
        return torch.zeros(
            observations.shape[0],
            self.action_dim,
            device=observations.device,
        )


def zero_q(obs_dim=OBS_DIM, action_dim=12):
    q = QNetwork(obs_dim, action_dim, (4,), "relu")
    for parameter in q.parameters():
        parameter.data.zero_()
    return q


def make_dataset(*, ang_vel, command, terminated):
    """Successor observation with known yaw slices; other blocks stay zero."""
    count = len(ang_vel)
    next_obs = torch.zeros(count, OBS_DIM)
    next_obs[:, 2] = torch.tensor(ang_vel, dtype=torch.float32)
    next_obs[:, 8] = torch.tensor(command, dtype=torch.float32)
    data = {
        "actor_obs": torch.zeros(count, OBS_DIM),
        "action": torch.zeros(count, 12),
        "next_actor_obs": next_obs,
        "successor_height_m": torch.zeros(count, 1),
        "terminated": torch.tensor(terminated, dtype=torch.bool),
        "timed_out": torch.zeros(count, dtype=torch.bool),
        "episode_id": torch.zeros(count, dtype=torch.int64),
        "step_id": torch.arange(count, dtype=torch.int64),
        "env_id": torch.zeros(count, dtype=torch.int64),
        "behavior_iteration": torch.zeros(count, dtype=torch.int64),
    }
    manifest = {
        "obs_names": list(OBS_NAMES),
        "obs_dims": list(OBS_DIMS),
        "obs_dim": OBS_DIM,
        "action_dim": 12,
    }
    return TransitionDataset(data=data, manifest=manifest, source="memory")


def test_relabel_uses_named_observation_fields_and_zero_after_terminal():
    dataset = make_dataset(
        ang_vel=[0.0, 0.5, -2.5, 1.0],
        command=[0.0, 0.0, 0.0, 2.0],
        terminated=[False, False, False, True],
    )
    spec = AngularVelocityRewardSpec(tracking_sigma=0.25)
    relabeled = relabel_angular_velocity(dataset, spec)
    # exp(-((cmd - ang) / 2.5)**2 / 0.25)
    expected = torch.tensor(
        [
            1.0,
            torch.exp(torch.tensor(-((0.5 / 2.5) ** 2) / 0.25)),
            torch.exp(torch.tensor(-((2.5 / 2.5) ** 2) / 0.25)),
            0.0,
        ]
    )
    torch.testing.assert_close(relabeled.data["reward"], expected)
    assert relabeled.data["reward"].dtype == torch.float32
    assert relabeled.manifest["reward_field"] == "reward"
    assert relabeled.manifest["reward_spec"] == {
        "tracking_sigma": 0.25,
        "error_scale": 2.5,
        "timing": "post_action_pre_reset",
        "terminal_continuation": "zero_after_terminal",
    }
    # Source snapshot is immutable.
    assert "reward" not in dataset.data
    assert dataset.manifest.get("reward_spec") is None


def test_relabel_matches_the_task_reward_term():
    """Formula oracle: Go2Trot._reward_tracking_ang_vel divides yaw error by 2.5."""
    task = torch.tensor(
        [torch.exp(torch.tensor(-0.16)), 1.0, torch.exp(torch.tensor(-4.0))]
    )
    dataset = make_dataset(
        ang_vel=[0.5, 0.0, 2.5],
        command=[0.0, 0.0, 0.0],
        terminated=[False, False, False],
    )
    relabeled = relabel_angular_velocity(
        dataset, AngularVelocityRewardSpec(tracking_sigma=0.25)
    )
    torch.testing.assert_close(relabeled.data["reward"], task)


def test_observation_field_lookup_follows_manifest_order():
    manifest = {"obs_names": list(OBS_NAMES), "obs_dims": list(OBS_DIMS)}
    assert observation_field_slice(manifest, "base_ang_vel") == slice(0, 3)
    assert observation_field_slice(manifest, "commands") == slice(6, 9)
    assert observation_field_slice(manifest, "dof_pos_obs") == slice(9, 13)
    reordered = {
        "obs_names": ["commands", "base_ang_vel"],
        "obs_dims": [3, 3],
    }
    assert observation_field_slice(reordered, "commands") == slice(0, 3)
    assert observation_field_slice(reordered, "base_ang_vel") == slice(3, 6)
    with pytest.raises(ValueError, match="does not record"):
        observation_field_slice(reordered, "projected_gravity")


def test_angular_velocity_spec_invariants():
    spec = AngularVelocityRewardSpec(tracking_sigma=0.25)
    assert spec.failure_value(0.99) == 0.0
    assert spec.failure_value(0.0) == 0.0
    assert (spec.min_reward, spec.max_reward) == (0.0, 1.0)
    with pytest.raises(ValueError, match="gamma"):
        spec.failure_value(1.0)
    with pytest.raises(ValueError, match="tracking_sigma"):
        AngularVelocityRewardSpec(tracking_sigma=0.0)
    with pytest.raises(ValueError, match="error_scale"):
        AngularVelocityRewardSpec(tracking_sigma=0.25, error_scale=-1.0)
    with pytest.raises(ValueError, match="terminal continuation"):
        AngularVelocityRewardSpec(
            tracking_sigma=0.25, terminal_continuation="absorbing_failure"
        )
    with pytest.raises(ValueError, match="reward timing"):
        AngularVelocityRewardSpec(tracking_sigma=0.25, timing="post_reset")
    reward = angular_velocity_reward(torch.tensor([0.0]), torch.tensor([0.0]), spec)
    assert float(reward) == 1.0
    assert (
        float(angular_velocity_reward(torch.tensor([0.0]), torch.tensor([10.0]), spec))
        < 1e-3
    )
    with pytest.raises(ValueError, match="matching shapes"):
        angular_velocity_reward(torch.zeros(2), torch.zeros(3), spec)


def test_fqe_terminal_target_is_reward_with_zero_failure_value():
    algorithm = FQE(
        zero_q(),
        FixedPolicy(action_dim=12),
        gamma=0.9,
        failure_value=0.0,
        target_tau=0.5,
        learning_rate=1e-3,
        device="cpu",
    )
    batch = {
        "q_obs": torch.zeros(2, OBS_DIM),
        "q_next_obs": torch.zeros(2, OBS_DIM),
        "policy_next_obs": torch.zeros(2, OBS_DIM),
        "action": torch.zeros(2, 12),
        "reward": torch.tensor([0.0, 0.5]),
        "terminated": torch.tensor([True, False]),
    }
    generator = torch.Generator().manual_seed(0)
    target = algorithm.compute_target(batch, generator)
    # terminal -> reward + gamma * 0; continuing -> reward + gamma * next_Q(=0)
    torch.testing.assert_close(target, torch.tensor([0.0, 0.5]))

    nonterminal = dict(batch)
    nonterminal["terminated"] = torch.tensor([False, False])
    continuing = algorithm.compute_target(nonterminal, generator)
    torch.testing.assert_close(continuing, batch["reward"])

    # A constant target-Q bias proves continuing rows bootstrap as r + gamma*Q'
    # while terminal rows ignore it (failure_value = 0).
    algorithm.target_q.net[-1].bias.data.fill_(2.0)
    target = algorithm.compute_target(batch, generator)
    torch.testing.assert_close(target, torch.tensor([0.0, 0.5 + 0.9 * 2.0]))
    torch.testing.assert_close(
        algorithm.compute_target(nonterminal, generator),
        batch["reward"] + 0.9 * 2.0,
    )


def make_policy_spec(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not-a-real-checkpoint")
    return PolicySpec(
        checkpoint=str(checkpoint),
        checkpoint_sha256="0" * 64,
        actor_cfg={"activation": "relu", "smooth_exploration": False},
        obs_names=("obs",),
        obs_dims=(OBS_DIM,),
        action_names=("act",),
        action_dim=12,
        noise={},
        scales={},
        control_dt=0.02,
        n_policy_steps=1,
        episode_length_s=5.0,
        smooth_exploration=False,
    )


def save_artifact(tmp_path, *, failure_value=0.0):
    return save_angular_velocity_q(
        tmp_path / "artifact",
        zero_q(),
        obs_mean=torch.zeros(OBS_DIM),
        obs_std=torch.ones(OBS_DIM),
        gamma=0.99,
        failure_value=failure_value,
        reward_spec={
            "tracking_sigma": 0.25,
            "error_scale": 2.5,
            "timing": "post_action_pre_reset",
            "terminal_continuation": "zero_after_terminal",
        },
        policy_spec=make_policy_spec(tmp_path),
        training_spec=TrainingSpec(gamma=0.99),
        dataset_manifest={"source_snapshots": [], "run_id": "test", "valid_count": 1},
        subset={"fraction": 1.0},
        source_revision="test",
    )


def test_artifact_round_trip_identity_and_broadcast(tmp_path):
    path = save_artifact(tmp_path)
    loaded = AngularVelocityQ(path)
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["format"] == ANGULAR_VEL_Q_FORMAT
    assert manifest["return_convention"] == ZERO_AFTER_TERMINAL_RETURN_CONVENTION
    assert manifest["failure_value"] == 0.0
    assert manifest["gamma"] == 0.99
    assert manifest["reward_spec"]["tracking_sigma"] == 0.25
    observations = torch.randn(1, OBS_DIM)
    actions = torch.randn(5, 12)
    prediction = loaded.predict(observations, actions)
    assert prediction.shape == (5,)
    torch.testing.assert_close(prediction, torch.zeros(5))
    with pytest.raises(ValueError, match="wrong width"):
        loaded.predict(torch.zeros(1, 3), actions)
    # The height loader must reject a different reward's artifact.
    with pytest.raises(ValueError, match="format"):
        HeightQ(path)


def test_artifact_rejects_nonzero_terminal_value(tmp_path):
    with pytest.raises(ValueError, match="failure_value"):
        save_artifact(tmp_path, failure_value=-100.0)


def test_artifact_loader_rejects_tampered_manifest(tmp_path):
    path = save_artifact(tmp_path)
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["return_convention"] = "discounted_continuing_absorbing_failure"
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="return convention"):
        AngularVelocityQ(path)
    manifest["return_convention"] = ZERO_AFTER_TERMINAL_RETURN_CONVENTION
    manifest["failure_value"] = -100.0
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="zero terminal value"):
        AngularVelocityQ(path)


def test_episode_split_is_disjoint_deterministic_and_complete():
    rows = 12
    dataset = make_dataset(
        ang_vel=[0.0] * rows,
        command=[0.0] * rows,
        terminated=[False] * rows,
    )
    # Four episodes of three rows, two environments each holding two episodes.
    dataset.data["env_id"] = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    dataset.data["episode_id"] = torch.tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
    train, validation = episode_split(dataset, 0.5, seed=3)
    again = episode_split(dataset, 0.5, seed=3)
    keys = ("env_id", "episode_id")
    train_groups = set(
        zip(train.data["env_id"].tolist(), train.data["episode_id"].tolist())
    )
    validation_groups = set(
        zip(validation.data["env_id"].tolist(), validation.data["episode_id"].tolist())
    )
    assert train_groups and validation_groups
    assert not train_groups & validation_groups
    assert len(train) + len(validation) == rows
    for left, right in ((train, again[0]), (validation, again[1])):
        for name in keys:
            assert torch.equal(left.data[name], right.data[name])
    assert train.manifest["split_side"] == "train"
    assert validation.manifest["validation_fraction"] == 0.5
    with pytest.raises(ValueError, match="validation_fraction"):
        episode_split(dataset, 1.0, seed=3)
    with pytest.raises(ValueError, match="validation_fraction"):
        episode_split(dataset, 0.0, seed=3)


def test_bellman_metrics_use_the_zero_terminal_target():
    dataset = make_dataset(
        ang_vel=[0.0, 0.0],
        command=[0.0, 0.0],
        terminated=[False, False],
    )
    dataset.data["reward"] = torch.tensor([1.0, 2.0])
    data = dict(dataset.data)
    algorithm = FQE(
        zero_q(),
        FixedPolicy(action_dim=12),
        gamma=0.5,
        failure_value=0.0,
        target_tau=0.5,
        learning_rate=1e-3,
        device="cpu",
    )
    metrics = _bellman_metrics(
        data,
        torch.arange(2),
        FixedPolicy(action_dim=12),
        algorithm,
        torch.zeros(OBS_DIM),
        torch.ones(OBS_DIM),
        torch.Generator().manual_seed(0),
    )
    # Zero Q: prediction 0, target = reward + gamma * 0.
    assert metrics["rows"] == 2
    torch.testing.assert_close(metrics["mse"], 2.5, atol=1e-6, rtol=0)
    torch.testing.assert_close(metrics["mae"], 1.5, atol=1e-6, rtol=0)
    torch.testing.assert_close(metrics["bias"], -1.5, atol=1e-6, rtol=0)
    torch.testing.assert_close(metrics["target_mean"], 1.5, atol=1e-6, rtol=0)
    assert metrics["prediction_mean"] == 0.0


def test_zero_after_terminal_accumulator_has_no_absorbing_tail():
    total = torch.zeros(2, dtype=torch.float64)
    alive = torch.ones(2, dtype=torch.bool)
    gamma = 0.5
    total, alive = accumulate_return_zero_after_terminal(
        total, alive, torch.tensor([1.0, 1.0]), torch.tensor([True, False]), 0, gamma
    )
    torch.testing.assert_close(total, torch.tensor([1.0, 1.0], dtype=torch.float64))
    assert alive.tolist() == [False, True]
    total, alive = accumulate_return_zero_after_terminal(
        total, alive, torch.tensor([1.0, 1.0]), torch.tensor([False, False]), 1, gamma
    )
    # The dead world stops accruing; the survivor adds gamma**1 * 1.
    torch.testing.assert_close(total, torch.tensor([1.0, 1.5], dtype=torch.float64))
    assert alive.tolist() == [False, True]
    # The height accumulator keeps its absorbing -1 tail and is untouched.
    height_total, _ = accumulate_return(
        torch.zeros(1, dtype=torch.float64),
        torch.ones(1, dtype=torch.bool),
        torch.tensor([0.0]),
        torch.tensor([True]),
        0,
        gamma,
    )
    torch.testing.assert_close(height_total, torch.tensor([-1.0], dtype=torch.float64))
