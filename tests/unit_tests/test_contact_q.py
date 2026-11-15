"""Contact-force reward, threshold derivation and artifact-identity tests."""

import json

import pytest
import torch

from learning.offline.contracts import (
    CONTACT_Q_FORMAT,
    ContactForceRewardSpec,
    PolicySpec,
    TrainingSpec,
)
from learning.offline.dataset import TransitionDataset, load_snapshot
from learning.algorithms.fqe import QNetwork
from learning.offline.q_artifact import ContactQ, save_contact_q
from learning.offline.rewards import (
    contact_force_reward,
    derive_contact_thresholds,
    foot_vertical_load,
    relabel_contact_force,
)

BODY_NAMES = ["base", "FL_foot", "FR_foot", "RL_foot", "RR_foot"]
FEET = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")


def make_dataset(
    loads_bw,
    bodyweight=None,
    terminated=None,
    env_id=None,
    episode_id=None,
    step_id=None,
):
    """Build a minimal contact snapshot: loads_bw is [N, 4] in bodyweight units."""
    loads = torch.as_tensor(loads_bw, dtype=torch.float32)
    rows = loads.shape[0]
    bodyweight = (
        torch.full((rows, 1), 168.0)
        if bodyweight is None
        else torch.as_tensor(bodyweight, dtype=torch.float32).reshape(rows, 1)
    )
    forces = torch.zeros(rows, len(BODY_NAMES) * 3)
    for k in range(4):
        forces[:, (k + 1) * 3 + 2] = loads[:, k] * bodyweight[:, 0]
    terminated = (
        torch.zeros(rows, dtype=torch.bool)
        if terminated is None
        else torch.as_tensor(terminated, dtype=torch.bool)
    )
    manifest = {
        "format": "brg-offline-replay-v2",
        "schema_version": 1,
        "obs_names": ["obs"],
        "obs_dims": [2],
        "action_names": ["act"],
        "action_dims": [1],
        "action_dim": 1,
        "obs_dim": 2,
        "valid_count": rows,
        "extra_fields": ["contact_forces", "bodyweight", "next_contact_forces"],
        "field_metadata": {"body_names": list(BODY_NAMES)},
        "env_cfg": {},
        "actor_cfg": {"smooth_exploration": False},
    }
    data = {
        "actor_obs": torch.zeros(rows, 2),
        "next_actor_obs": torch.zeros(rows, 2),
        "action": torch.zeros(rows, 1),
        "successor_height_m": torch.zeros(rows, 1),
        "terminated": terminated,
        "timed_out": torch.zeros(rows, dtype=torch.bool),
        "episode_id": (
            torch.zeros(rows, dtype=torch.int64)
            if episode_id is None
            else torch.as_tensor(episode_id, dtype=torch.int64)
        ),
        "step_id": (
            torch.arange(rows, dtype=torch.int64)
            if step_id is None
            else torch.as_tensor(step_id, dtype=torch.int64)
        ),
        "env_id": (
            torch.zeros(rows, dtype=torch.int64)
            if env_id is None
            else torch.as_tensor(env_id, dtype=torch.int64)
        ),
        "behavior_iteration": torch.zeros(rows, dtype=torch.int64),
        "contact_forces": forces.clone(),
        "next_contact_forces": forces,
        "bodyweight": bodyweight,
    }
    return TransitionDataset(data=data, manifest=manifest, source="test")


def make_policy_spec() -> PolicySpec:
    return PolicySpec(
        checkpoint="model.pt",
        checkpoint_sha256="0" * 64,
        actor_cfg={"smooth_exploration": False},
        obs_names=("obs",),
        obs_dims=(2,),
        action_names=("act",),
        action_dim=1,
        noise={},
        scales={},
        control_dt=0.02,
        n_policy_steps=1,
        episode_length_s=5.0,
        smooth_exploration=False,
    )


def test_spec_validates_thresholds_and_percentile():
    spec = ContactForceRewardSpec(
        foot_names=FEET,
        thresholds_bw=(0.9, 0.95, 0.8, 0.8),
        percentile=90.0,
    )
    assert spec.min_reward == -1.0
    assert spec.max_reward == 0.0
    assert spec.failure_value(0.99) == 0.0
    with pytest.raises(ValueError):
        ContactForceRewardSpec(foot_names=FEET, thresholds_bw=(0.9,), percentile=90.0)
    with pytest.raises(ValueError):
        ContactForceRewardSpec(
            foot_names=FEET, thresholds_bw=(0.9, 0.9, 0.0, 0.8), percentile=90.0
        )
    with pytest.raises(ValueError):
        ContactForceRewardSpec(
            foot_names=FEET, thresholds_bw=(0.9, 0.9, 0.9, 0.9), percentile=100.0
        )
    with pytest.raises(ValueError):
        ContactForceRewardSpec(
            foot_names=("a", "a"),
            thresholds_bw=(0.9, 0.9),
            percentile=90.0,
        )
    with pytest.raises(ValueError):
        ContactForceRewardSpec(
            foot_names=FEET,
            thresholds_bw=(0.9, 0.9, 0.9, 0.9),
            percentile=90.0,
            timing="pre_action",
        )
    with pytest.raises(ValueError):
        ContactForceRewardSpec(
            foot_names=FEET,
            thresholds_bw=(0.9, 0.9, 0.9, 0.9),
            percentile=90.0,
            terminal_continuation="absorbing_failure",
        )


def test_reward_is_one_per_step_indicator_at_the_threshold():
    spec = ContactForceRewardSpec(
        foot_names=FEET,
        thresholds_bw=(0.5, 0.5, 0.5, 0.5),
        percentile=90.0,
    )
    loads = torch.tensor(
        [
            [0.10, 0.10, 0.10, 0.10],  # below
            [0.50, 0.10, 0.10, 0.10],  # exactly at the threshold: allowed
            [0.51, 0.10, 0.10, 0.10],  # above on one foot
            [0.10, 0.10, 0.10, 0.90],  # above on another
        ]
    )
    torch.testing.assert_close(
        contact_force_reward(loads, spec), torch.tensor([0.0, 0.0, -1.0, -1.0])
    )


def test_bodyweight_normalisation_uses_the_recorded_value():
    loads = torch.tensor([[0.6, 0.1, 0.1, 0.1]])
    light = make_dataset(loads, bodyweight=140.0)
    heavy = make_dataset(loads, bodyweight=190.0)
    got_light = foot_vertical_load(light, FEET)
    got_heavy = foot_vertical_load(heavy, FEET)
    torch.testing.assert_close(got_light, loads)
    torch.testing.assert_close(got_heavy, loads)


def test_relabel_masks_terminal_rows_and_records_the_spec():
    loads = torch.tensor(
        [[0.9, 0.1, 0.1, 0.1], [0.9, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]
    )
    dataset = make_dataset(loads, terminated=[False, True, False])
    spec = ContactForceRewardSpec(
        foot_names=FEET,
        thresholds_bw=(0.5, 0.5, 0.5, 0.5),
        percentile=90.0,
    )
    relabeled = relabel_contact_force(dataset, spec)
    # a terminated transition contributes no reward, whatever the force
    torch.testing.assert_close(relabeled.data["reward"], torch.tensor([-1.0, 0.0, 0.0]))
    assert relabeled.manifest["reward_spec"]["thresholds_bw"] == (0.5, 0.5, 0.5, 0.5)
    assert relabeled.manifest["reward_field"] == "reward"
    assert "reward" not in dataset.data  # source snapshot untouched


def test_contact_fields_are_required():
    dataset = make_dataset(torch.zeros(2, 4))
    dataset.manifest["extra_fields"] = ["dof_pos"]
    dataset.data.pop("bodyweight")
    with pytest.raises(ValueError, match="contact-force contract"):
        foot_vertical_load(dataset, FEET)
    dataset = make_dataset(torch.zeros(2, 4))
    with pytest.raises(ValueError, match="not in the recorded body list"):
        foot_vertical_load(dataset, ("tail_foot",))


def test_thresholds_are_the_percentile_of_touchdown_peaks():
    # two episodes, one env; each episode has a contact run of known peak
    step_id = list(range(20)) * 2
    episode_id = [0] * 20 + [1] * 20
    loads = torch.zeros(40, 4)
    for k in range(4):
        loads[2:10, k] = 0.6  # peak 0.6 in episode 0
        loads[22:30, k] = 0.8  # peak 0.8 in episode 1
    dataset = make_dataset(loads, step_id=step_id, episode_id=episode_id)
    derived = derive_contact_thresholds(dataset, FEET, 50.0)
    # the two peaks are 0.6 and 0.8; the median lies between them
    for name in FEET:
        assert 0.6 <= derived[name] <= 0.8
    # a run shorter than min_steps is not a touchdown
    short = torch.zeros(40, 4)
    for k in range(4):
        short[5:8, k] = 0.9  # 3 steps only
    dataset = make_dataset(short, step_id=step_id, episode_id=episode_id)
    with pytest.raises(ValueError, match="no touchdowns"):
        derive_contact_thresholds(dataset, FEET, 50.0)


def test_contact_artifact_round_trip_and_identity(tmp_path):
    q = QNetwork(2, 1, (8,), "relu")
    obs_mean = torch.zeros(2)
    obs_std = torch.ones(2)
    training_spec = TrainingSpec(gamma=0.99)
    spec = ContactForceRewardSpec(
        foot_names=FEET,
        thresholds_bw=(0.9, 0.95, 0.8, 0.8),
        percentile=90.0,
    )
    path = tmp_path / "contact_q"
    save_contact_q(
        path,
        q,
        obs_mean=obs_mean,
        obs_std=obs_std,
        gamma=0.99,
        reward_spec={
            "foot_names": list(spec.foot_names),
            "thresholds_bw": list(spec.thresholds_bw),
            "percentile": spec.percentile,
            "timing": spec.timing,
            "terminal_continuation": spec.terminal_continuation,
        },
        policy_spec=make_policy_spec(),
        training_spec=training_spec,
        dataset_manifest={"run_id": "test", "valid_count": 4, "source": ["a"]},
        subset={"fraction": 1.0},
        source_revision="test",
    )
    artifact = ContactQ(path)
    assert artifact.manifest["format"] == CONTACT_Q_FORMAT
    assert artifact.manifest["failure_value"] == 0.0
    prediction = artifact.predict(torch.zeros(2), torch.zeros(1))
    assert prediction.shape == (1,)

    from learning.offline.q_artifact import AngularVelocityQ, HeightQ

    with pytest.raises(ValueError, match="unsupported Q artifact format"):
        AngularVelocityQ(path)
    with pytest.raises(ValueError, match="unsupported Q artifact format"):
        HeightQ(path)

    # a contact artifact must refuse a spec that is not a contact spec
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["reward_spec"] = {"reference_height_m": 0.4, "bound_m": 0.05}
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="invalid contact reward spec"):
        ContactQ(path)


def test_save_contact_q_refuses_a_nonzero_failure_value(tmp_path):
    q = QNetwork(2, 1, (8,), "relu")
    with pytest.raises(ValueError, match="failure_value"):
        save_contact_q(
            tmp_path / "q",
            q,
            obs_mean=torch.zeros(2),
            obs_std=torch.ones(2),
            gamma=0.99,
            reward_spec={
                "foot_names": list(FEET),
                "thresholds_bw": [0.9, 0.9, 0.9, 0.9],
                "percentile": 90.0,
            },
            policy_spec=make_policy_spec(),
            training_spec=TrainingSpec(gamma=0.99),
            dataset_manifest={},
            subset={"fraction": 1.0},
            source_revision="test",
            failure_value=-100.0,
        )


def test_legacy_snapshot_without_contacts_is_rejected(tmp_path):
    dataset = make_dataset(torch.zeros(4, 4))
    dataset.manifest["format"] = "brg-offline-replay-v1"
    dataset.manifest.pop("extra_fields")
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    torch.save(
        {
            name: value
            for name, value in dataset.data.items()
            if name
            in (
                "actor_obs",
                "next_actor_obs",
                "action",
                "successor_height_m",
                "terminated",
                "timed_out",
                "episode_id",
                "step_id",
                "env_id",
                "behavior_iteration",
            )
        },
        snapshot / "data.pt",
    )
    (snapshot / "manifest.json").write_text(json.dumps(dataset.manifest))
    loaded = load_snapshot(snapshot)
    with pytest.raises(ValueError, match="contact-force contract"):
        relabel_contact_force(
            loaded,
            ContactForceRewardSpec(
                foot_names=FEET,
                thresholds_bw=(0.9, 0.9, 0.9, 0.9),
                percentile=90.0,
            ),
        )
