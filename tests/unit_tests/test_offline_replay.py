"""Replay recording, relabeling and subset integrity tests."""

import json
from dataclasses import asdict

import pytest
import torch

from learning.offline.contracts import (
    HeightRewardSpec,
    RecorderConfig,
    sample_observation_noise,
)
from learning.offline.dataset import (
    concat_datasets,
    load_snapshot,
    nested_subsets,
    sample_n,
)
from learning.offline.replay_fields import (
    ALL_REPLAY_FIELDS,
    REPLAY_FEATURE_GROUPS,
    resolve_replay_fields,
)
from learning.offline.rewards import relabel_height
from learning.storage.transition_recorder import TransitionRecorder


def make_recorder(
    tmp_path,
    *,
    capacity=4,
    num_envs=2,
    seed=0,
    noise=None,
    extra_fields=(),
    reward_term_names=(),
    field_metadata=None,
):
    return TransitionRecorder(
        num_envs=num_envs,
        obs_names=("obs",),
        obs_dims=(3,),
        action_names=("act",),
        action_dims=(2,),
        noise=noise or {},
        scales={},
        device="cpu",
        output_dir=str(tmp_path / "replay"),
        capacity=capacity,
        snapshot_interval=0,
        seed=seed,
        task_name="test",
        backend="fake",
        source_revision="test",
        log_dir=str(tmp_path / "run"),
        actor_cfg={
            "obs": ["obs"],
            "actions": ["act"],
            "noise": {},
            "smooth_exploration": False,
        },
        env_cfg={"seed": 0},
        runner_cfg={"replay": {"enabled": True}},
        episode_length_s=10.0,
        n_policy_steps=1,
        control_dt=0.02,
        extra_fields=extra_fields,
        reward_term_names=reward_term_names,
        field_metadata=field_metadata,
    )


def batch(offset=0.0, done_index=1, num_envs=2):
    actor_obs = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])[:num_envs] + offset
    post_reset_obs = actor_obs + 10.0
    pre_reset_clean_obs = actor_obs + 20.0
    actions = torch.tensor([[0.1, 0.2], [0.3, 0.4]])[:num_envs] + offset
    height = torch.tensor([[0.30], [0.40]])[:num_envs] + offset
    terminated = torch.zeros(num_envs, dtype=torch.bool)
    terminated[done_index] = True
    timed_out = torch.zeros(num_envs, dtype=torch.bool)
    return {
        "actor_obs": actor_obs,
        "action": actions,
        "post_reset_obs": post_reset_obs,
        "pre_reset_clean_obs": pre_reset_clean_obs,
        "successor_height_m": height,
        "terminated": terminated,
        "timed_out": timed_out,
    }


def test_recorder_keeps_last_capacity_with_chronological_snapshot(tmp_path):
    recorder = make_recorder(tmp_path, capacity=3)
    recorder.append(**batch(offset=0.0), behavior_iteration=1)
    recorder.append(**batch(offset=100.0), behavior_iteration=2)
    assert recorder.valid_count == 3
    assert recorder.total_appended == 4

    snapshot = recorder.snapshot(2)
    dataset = load_snapshot(snapshot)
    assert len(dataset) == 3
    # Last three rows in collection order: second row of batch one, then batch two.
    expected = torch.tensor(
        [
            [4.0, 5.0, 6.0],
            [101.0, 102.0, 103.0],
            [104.0, 105.0, 106.0],
        ]
    )
    torch.testing.assert_close(dataset.data["actor_obs"], expected)
    assert torch.equal(dataset.data["episode_id"], torch.tensor([0, 0, 1]))
    assert torch.equal(dataset.data["step_id"], torch.tensor([0, 1, 0]))


def test_recorder_preserves_pre_reset_successor_for_ended_rows(tmp_path):
    recorder = make_recorder(tmp_path)
    recorder.append(**batch(), behavior_iteration=1)
    snapshot = recorder.snapshot(1)
    dataset = load_snapshot(snapshot)
    # Env 0 did not end: reuse the collected post-reset observation.
    torch.testing.assert_close(
        dataset.data["next_actor_obs"][0],
        torch.tensor([11.0, 12.0, 13.0]),
    )
    # Env 1 ended: keep the clean pre-reset successor (zero recording noise).
    torch.testing.assert_close(
        dataset.data["next_actor_obs"][1],
        torch.tensor([24.0, 25.0, 26.0]),
    )
    assert dataset.data["terminated"].tolist() == [False, True]
    assert dataset.data["timed_out"].tolist() == [False, False]


def test_recorder_does_not_advance_training_rng(tmp_path):
    recorder = make_recorder(tmp_path, noise={"obs": 0.1})
    torch.manual_seed(123)
    state = torch.get_rng_state()
    recorder.append(**batch(), behavior_iteration=1)
    assert torch.equal(state, torch.get_rng_state())
    # The recording noise remains reproducible from its own generator.
    other = make_recorder(tmp_path / "other", noise={"obs": 0.1})
    torch.manual_seed(999)
    state = torch.get_rng_state()
    other.append(**batch(), behavior_iteration=1)
    assert torch.equal(state, torch.get_rng_state())
    torch.testing.assert_close(
        recorder._buffers["next_actor_obs"][1],
        other._buffers["next_actor_obs"][1],
    )


def test_snapshot_is_immutable_and_rejects_empty_or_duplicate(tmp_path):
    recorder = make_recorder(tmp_path)
    with pytest.raises(ValueError):
        recorder.snapshot(1)
    recorder.append(**batch(), behavior_iteration=1)
    first = recorder.snapshot(1)
    assert first.is_dir()
    assert recorder.snapshot(2) is None
    recorder.append(**batch(offset=1.0), behavior_iteration=2)
    with pytest.raises(FileExistsError):
        recorder.snapshot(1)


def test_relabel_height_uses_inclusive_bound(tmp_path):
    recorder = make_recorder(tmp_path)
    recorder.append(**batch(), behavior_iteration=1)
    dataset = load_snapshot(recorder.snapshot(1))
    relabeled = relabel_height(
        dataset,
        HeightRewardSpec(reference_height_m=0.3, bound_m=0.05),
    )
    assert relabeled.data["reward"].tolist() == [0.0, -1.0]
    assert dataset.manifest.get("reward_spec") is None
    assert relabeled.manifest["reward_spec"]["bound_m"] == 0.05


def test_nested_subsets_are_deterministic_and_nested(tmp_path):
    recorder = make_recorder(tmp_path, capacity=10, num_envs=1)
    for iteration in range(10):
        recorder.append(
            **batch(offset=float(iteration), done_index=0, num_envs=1),
            behavior_iteration=iteration,
        )
    dataset = load_snapshot(recorder.snapshot(10))
    first = nested_subsets(dataset, (0.3, 0.6, 1.0), seed=7, group_by="transition")
    second = nested_subsets(dataset, (0.3, 0.6, 1.0), seed=7, group_by="transition")
    assert [len(item) for item in first] == [3, 6, 10]
    for left, right in zip(first, second, strict=True):
        torch.testing.assert_close(left.data["actor_obs"], right.data["actor_obs"])
    torch.testing.assert_close(
        first[0].data["actor_obs"], first[2].data["actor_obs"][:3]
    )


def test_episode_subsets_and_sample_n(tmp_path):
    recorder = make_recorder(tmp_path, capacity=10, num_envs=1)
    for iteration in range(10):
        recorder.append(
            **batch(offset=float(iteration), done_index=0, num_envs=1),
            behavior_iteration=iteration,
        )
    dataset = load_snapshot(recorder.snapshot(10))
    subsets = nested_subsets(dataset, (0.3, 1.0), seed=3, group_by="episode")
    assert len(subsets[0]) < len(subsets[1])
    sampled = sample_n(dataset, 4, seed=3, group_by="transition")
    assert len(sampled) == 4


def test_concat_datasets_deduplicates_overlapping_ids(tmp_path):
    recorder = make_recorder(tmp_path, capacity=4, num_envs=1)
    recorder.append(**batch(done_index=0, num_envs=1), behavior_iteration=1)
    first = load_snapshot(recorder.snapshot(1))
    combined = concat_datasets([first, first])
    assert len(combined) == len(first)
    assert combined.manifest["deduplicated"] == len(first)


def test_recorder_config_and_height_spec_validation():
    with pytest.raises(ValueError):
        RecorderConfig(capacity=0)
    with pytest.raises(ValueError):
        RecorderConfig(enabled=True, task_name="", backend="fake")
    with pytest.raises(ValueError):
        HeightRewardSpec(reference_height_m=0.3, bound_m=-0.1)
    with pytest.raises(ValueError):
        HeightRewardSpec(reference_height_m=0.3, bound_m=0.1, outside_reward=0.0)
    assert asdict(HeightRewardSpec(0.3, 0.1))["inside_reward"] == 0.0


def test_noise_sampling_uses_only_the_owned_generator():
    torch.manual_seed(5)
    state = torch.get_rng_state()
    generator = torch.Generator().manual_seed(11)
    noise = sample_observation_noise(
        ("obs",), (3,), {"obs": 0.1}, {}, 2, "cpu", generator
    )
    assert torch.equal(state, torch.get_rng_state())
    assert noise.shape == (2, 3)
    torch.testing.assert_close(
        noise,
        sample_observation_noise(
            ("obs",),
            (3,),
            {"obs": 0.1},
            {},
            2,
            "cpu",
            torch.Generator().manual_seed(11),
        ),
    )


EXTRA_WIDTHS = {"critic_obs": 5, "root_states": 13, "dof_pos": 4, "reward_terms": 2}


def extra_batch(fields, offset=0.0, num_envs=2):
    return {
        name: torch.full((num_envs, EXTRA_WIDTHS[name]), offset + float(index))
        for index, name in enumerate(fields)
    }


def test_resolve_replay_fields_expands_groups_and_fields():
    assert resolve_replay_fields("") == ()
    assert resolve_replay_fields(None) == ()
    assert resolve_replay_fields("core") == ()
    assert resolve_replay_fields("all") == ALL_REPLAY_FIELDS
    assert resolve_replay_fields("contacts") == REPLAY_FEATURE_GROUPS["contacts"]
    assert resolve_replay_fields("dof_pos, root_states") == tuple(
        name for name in ALL_REPLAY_FIELDS if name in {"dof_pos", "root_states"}
    )
    with pytest.raises(ValueError):
        resolve_replay_fields("dof_poss")
    with pytest.raises(ValueError):
        resolve_replay_fields("all,dof_pos")


def test_recorder_config_requires_canonical_field_order():
    with pytest.raises(ValueError):
        RecorderConfig(
            enabled=True,
            task_name="test",
            backend="fake",
            fields=("dof_pos", "critic_obs"),
        )
    config = RecorderConfig(
        enabled=True,
        task_name="test",
        backend="fake",
        fields=REPLAY_FEATURE_GROUPS["critic"],
    )
    assert config.fields == REPLAY_FEATURE_GROUPS["critic"]


def test_extended_snapshot_round_trips_selected_fields(tmp_path):
    fields = ("critic_obs", "root_states", "dof_pos")
    recorder = make_recorder(
        tmp_path,
        extra_fields=fields,
        field_metadata={"body_names": ["base", "FR_foot"], "dof_names": ["a"]},
    )
    recorder.append(
        **batch(),
        behavior_iteration=1,
        extra_features=extra_batch(fields),
    )
    dataset = load_snapshot(recorder.snapshot(1))
    assert dataset.manifest["extra_fields"] == list(fields)
    assert tuple(dataset.manifest["recorded_fields"][-3:]) == fields
    assert dataset.manifest["field_dims"]["root_states"] == 13
    assert dataset.manifest["field_metadata"]["body_names"] == ["base", "FR_foot"]
    torch.testing.assert_close(dataset.data["dof_pos"], torch.full((2, 4), 2.0))
    assert dataset.data["actor_obs"].shape == (2, 3)


def test_extended_snapshot_requires_exact_extra_field_set(tmp_path):
    recorder = make_recorder(tmp_path, extra_fields=("dof_pos",))
    with pytest.raises(ValueError):
        recorder.append(**batch(), behavior_iteration=1)
    with pytest.raises(ValueError):
        recorder.append(
            **batch(),
            behavior_iteration=1,
            extra_features={
                "dof_pos": torch.zeros(2, 4),
                "root_states": torch.zeros(2, 13),
            },
        )
    with pytest.raises(ValueError):
        recorder.append(
            **batch(),
            behavior_iteration=1,
            extra_features={"dof_pos": torch.full((2, 4), float("nan"))},
        )


def test_extended_snapshot_rejects_width_change(tmp_path):
    recorder = make_recorder(tmp_path, extra_fields=("dof_pos",))
    recorder.append(
        **batch(), behavior_iteration=1, extra_features={"dof_pos": torch.zeros(2, 4)}
    )
    with pytest.raises(ValueError):
        recorder.append(
            **batch(),
            behavior_iteration=2,
            extra_features={"dof_pos": torch.zeros(2, 5)},
        )


def test_reward_terms_requires_and_records_term_names(tmp_path):
    with pytest.raises(ValueError):
        make_recorder(tmp_path / "bad", extra_fields=("reward_terms",))
    recorder = make_recorder(
        tmp_path,
        extra_fields=("reward_terms",),
        reward_term_names=("tracking_lin_vel", "termination"),
    )
    recorder.append(
        **batch(),
        behavior_iteration=1,
        extra_features={"reward_terms": torch.ones(2, 2)},
    )
    dataset = load_snapshot(recorder.snapshot(1))
    assert dataset.manifest["reward_term_names"] == [
        "tracking_lin_vel",
        "termination",
    ]
    assert dataset.data["reward_terms"].shape == (2, 2)


def test_legacy_core_only_snapshot_still_loads(tmp_path):
    recorder = make_recorder(tmp_path)
    recorder.append(**batch(), behavior_iteration=1)
    snapshot = recorder.snapshot(1)
    manifest = json.loads((snapshot / "manifest.json").read_text())
    for key in (
        "recorded_fields",
        "extra_fields",
        "field_dims",
        "core_fields",
        "reward_term_names",
    ):
        manifest.pop(key, None)
    manifest["format"] = "brg-offline-replay-v1"
    (snapshot / "manifest.json").write_text(json.dumps(manifest))
    dataset = load_snapshot(snapshot)
    assert len(dataset) == 2


def test_legacy_format_rejects_declared_extra_fields(tmp_path):
    fields = ("dof_pos",)
    recorder = make_recorder(tmp_path, extra_fields=fields)
    recorder.append(**batch(), behavior_iteration=1, extra_features=extra_batch(fields))
    snapshot = recorder.snapshot(1)
    manifest = json.loads((snapshot / "manifest.json").read_text())
    manifest["format"] = "brg-offline-replay-v1"
    (snapshot / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_snapshot(snapshot)


def test_snapshot_rejects_a_missing_extra_tensor(tmp_path):
    fields = ("dof_pos",)
    recorder = make_recorder(tmp_path, extra_fields=fields)
    recorder.append(**batch(), behavior_iteration=1, extra_features=extra_batch(fields))
    snapshot = recorder.snapshot(1)
    data = torch.load(snapshot / "data.pt", weights_only=True)
    data.pop("dof_pos")
    torch.save(data, snapshot / "data.pt")
    with pytest.raises(ValueError):
        load_snapshot(snapshot)


def test_concat_preserves_extra_fields(tmp_path):
    fields = ("dof_pos",)
    recorder = make_recorder(tmp_path, capacity=1, num_envs=1, extra_fields=fields)
    recorder.append(
        **batch(done_index=0, num_envs=1),
        behavior_iteration=1,
        extra_features={"dof_pos": torch.full((1, 4), 1.0)},
    )
    first = load_snapshot(recorder.snapshot(1))
    recorder.append(
        **batch(done_index=0, num_envs=1),
        behavior_iteration=2,
        extra_features={"dof_pos": torch.full((1, 4), 2.0)},
    )
    second = load_snapshot(recorder.snapshot(2))
    combined = concat_datasets([first, second])
    assert len(combined) == 2
    torch.testing.assert_close(combined.data["dof_pos"][:, 0], torch.tensor([1.0, 2.0]))
    with pytest.raises(ValueError):
        concat_datasets([first, load_snapshot(_core_only(tmp_path))])


def _core_only(tmp_path):
    recorder = make_recorder(tmp_path / "core", capacity=1, num_envs=1)
    recorder.append(**batch(done_index=0, num_envs=1), behavior_iteration=1)
    return recorder.snapshot(1)
