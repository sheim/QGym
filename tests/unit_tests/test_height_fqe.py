"""Fixed-policy FQE semantics and artifact round-trip tests."""

import pytest
import torch

from learning.algorithms.fqe import FQE, QNetwork
from learning.modules.actor import Actor
from learning.offline.contracts import (
    HeightRewardSpec,
    PolicySpec,
    TrainingSpec,
)
from learning.offline.nominal_policy import FrozenTrotPolicy
from learning.offline.evaluation import simulate_candidate
from learning.offline.q_artifact import HeightQ, save_height_q


class FixedPolicy:
    def __init__(self, action_dim=2):
        self.action_dim = action_dim

    def sample_actions(self, observations, generator):
        return torch.zeros(
            observations.shape[0],
            self.action_dim,
            device=observations.device,
        )


def zero_q():
    q = QNetwork(3, 2, (4,), "relu")
    for parameter in q.parameters():
        parameter.data.zero_()
    return q


def make_batch(terminated):
    return {
        "q_obs": torch.zeros(2, 3),
        "q_next_obs": torch.zeros(2, 3),
        "policy_next_obs": torch.zeros(2, 3),
        "action": torch.zeros(2, 2),
        "reward": torch.zeros(2),
        "terminated": terminated,
    }


def test_terminal_target_uses_absorbing_failure():
    algorithm = FQE(
        zero_q(),
        FixedPolicy(),
        gamma=0.9,
        failure_value=-10.0,
        target_tau=0.5,
        learning_rate=1e-3,
        device="cpu",
    )
    generator = torch.Generator().manual_seed(0)
    target = algorithm.compute_target(
        make_batch(torch.tensor([True, False])), generator
    )
    torch.testing.assert_close(target, torch.tensor([-9.0, 0.0]))


def test_terminal_precedes_timeout_and_timeout_bootstraps():
    algorithm = FQE(
        zero_q(),
        FixedPolicy(),
        gamma=0.9,
        failure_value=-10.0,
        target_tau=0.5,
        learning_rate=1e-3,
        device="cpu",
    )
    generator = torch.Generator().manual_seed(0)
    batch = make_batch(torch.tensor([True, False]))
    batch["timed_out"] = torch.tensor([True, True])
    target = algorithm.compute_target(batch, generator)
    torch.testing.assert_close(target, torch.tensor([-9.0, 0.0]))


def test_fqe_update_changes_q_and_target():
    algorithm = FQE(
        zero_q(),
        FixedPolicy(),
        gamma=0.9,
        failure_value=-10.0,
        target_tau=0.5,
        learning_rate=1e-2,
        device="cpu",
    )
    generator = torch.Generator().manual_seed(0)
    batch = make_batch(torch.tensor([True, False]))
    batch["reward"] = torch.tensor([1.0, 1.0])
    before = [parameter.detach().clone() for parameter in algorithm.q.parameters()]
    loss = algorithm.update(batch, generator)
    assert loss >= 0.0
    after = [parameter.detach().clone() for parameter in algorithm.q.parameters()]
    assert any(not torch.equal(left, right) for left, right in zip(before, after))


def make_policy_spec(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not-a-real-checkpoint")
    return PolicySpec(
        checkpoint=str(checkpoint),
        checkpoint_sha256="0" * 64,
        actor_cfg={"activation": "relu", "smooth_exploration": False},
        obs_names=("obs",),
        obs_dims=(3,),
        action_names=("act",),
        action_dim=2,
        noise={},
        scales={},
        control_dt=0.02,
        n_policy_steps=1,
        episode_length_s=5.0,
        smooth_exploration=False,
    )


def test_q_artifact_round_trip_and_broadcast(tmp_path):
    q = QNetwork(3, 2, (4,), "relu")
    spec = make_policy_spec(tmp_path)
    path = save_height_q(
        tmp_path / "artifact",
        q,
        obs_mean=torch.zeros(3),
        obs_std=torch.ones(3),
        gamma=0.99,
        failure_value=-100.0,
        reward_spec={"reference_height_m": 0.3, "bound_m": 0.1},
        policy_spec=spec,
        training_spec=TrainingSpec(gamma=0.99),
        dataset_manifest={
            "source_snapshots": [],
            "run_id": "test",
            "valid_count": 1,
        },
        subset={"fraction": 1.0},
        source_revision="test",
    )
    loaded = HeightQ(path)
    observations = torch.randn(1, 3)
    actions = torch.randn(5, 2)
    prediction = loaded.predict(observations, actions)
    expected = q(observations.expand(5, -1), actions)
    torch.testing.assert_close(prediction, expected)
    assert (
        loaded.manifest["return_convention"]
        == "discounted_continuing_absorbing_failure"
    )
    with pytest.raises(ValueError):
        loaded.predict(torch.zeros(1, 2), actions)


def test_frozen_policy_loads_actor_and_does_not_train(tmp_path):
    actor_cfg = {
        "hidden_dims": [4],
        "activation": "relu",
        "init_noise_std": 0.5,
        "normalize_obs": False,
        "smooth_exploration": False,
        "obs": ["obs"],
        "actions": ["act"],
    }
    actor = Actor(3, 2, **actor_cfg)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"actor_state_dict": actor.state_dict()}, checkpoint)
    policy = FrozenTrotPolicy.from_checkpoint(
        checkpoint,
        actor_cfg,
        ("obs",),
        (3,),
        ("act",),
        2,
        "cpu",
    )
    observations = torch.zeros(2, 3)
    generator = torch.Generator().manual_seed(0)
    actions = policy.sample_actions(observations, generator)
    assert actions.shape == (2, 2)
    assert torch.isfinite(actions).all()
    assert policy.mean_actions(observations).shape == (2, 2)
    assert all(not p.requires_grad for p in policy.actor.parameters())
    with pytest.raises(ValueError):
        FrozenTrotPolicy.from_checkpoint(
            checkpoint,
            dict(actor_cfg, smooth_exploration=True),
            ("obs",),
            (3,),
            ("act",),
            2,
            "cpu",
        )


def test_policy_and_training_spec_validation(tmp_path):
    with pytest.raises(ValueError):
        PolicySpec(
            checkpoint="x",
            checkpoint_sha256="",
            actor_cfg={},
            obs_names=("obs",),
            obs_dims=(3,),
            action_names=("act",),
            action_dim=2,
            noise={},
            scales={},
            control_dt=0.02,
            n_policy_steps=1,
            episode_length_s=5.0,
            smooth_exploration=False,
        )
    with pytest.raises(ValueError):
        TrainingSpec(gamma=1.0)
    with pytest.raises(ValueError):
        HeightRewardSpec(reference_height_m=0.3, bound_m=-0.1)


class _FakeEnv:
    num_envs = 1
    device = torch.device("cpu")

    def __init__(self, heights, terminations):
        self.heights = heights
        self.terminations = terminations
        self.index = 0
        self.base_height = torch.tensor([[heights[0]]])
        self.terminated = torch.tensor([False])

    def set_states(self, names, values):
        self.action = values.clone()

    def step(self):
        self.base_height = torch.tensor([[self.heights[self.index]]])
        self.terminated = torch.tensor([self.terminations[self.index]])
        self.index += 1

    def get_states(self, names):
        return torch.zeros(1, 3)


def _evaluation_policy_spec():
    return PolicySpec(
        checkpoint="fake",
        checkpoint_sha256="0" * 64,
        actor_cfg={"activation": "relu", "smooth_exploration": False},
        obs_names=("obs",),
        obs_dims=(3,),
        action_names=("act",),
        action_dim=2,
        noise={},
        scales={},
        control_dt=0.02,
        n_policy_steps=1,
        episode_length_s=5.0,
        smooth_exploration=False,
    )


def test_simulate_candidate_terminal_and_tail():
    spec = _evaluation_policy_spec()
    reward = HeightRewardSpec(reference_height_m=0.3, bound_m=0.05)
    policy = FixedPolicy()
    terminal = simulate_candidate(
        _FakeEnv([0.3, 0.5], [False, True]),
        policy,
        spec,
        torch.zeros(2),
        reward_spec=reward,
        gamma=0.9,
        failure_value=-10.0,
        horizon=5,
        continuation_seed=0,
    )
    assert terminal["terminated"] is True
    assert terminal["steps"] == 2
    torch.testing.assert_close(
        torch.tensor(terminal["observed_return"]), torch.tensor(-9.0)
    )
    torch.testing.assert_close(
        torch.tensor(terminal["return_lower"]), torch.tensor(-9.0)
    )
    torch.testing.assert_close(
        torch.tensor(terminal["return_upper"]), torch.tensor(-9.0)
    )

    surviving = simulate_candidate(
        _FakeEnv([0.3, 0.4], [False, False]),
        policy,
        spec,
        torch.zeros(2),
        reward_spec=reward,
        gamma=0.9,
        failure_value=-10.0,
        horizon=2,
        continuation_seed=0,
    )
    assert surviving["terminated"] is False
    torch.testing.assert_close(
        torch.tensor(surviving["observed_return"]), torch.tensor(-0.9)
    )
    torch.testing.assert_close(
        torch.tensor(surviving["return_lower"]), torch.tensor(-9.0)
    )
    torch.testing.assert_close(
        torch.tensor(surviving["return_upper"]), torch.tensor(-0.9)
    )


def test_batched_evaluator_defaults_to_scoring_the_reset_state():
    """insert_step must stay 0 by default so the original protocol is unchanged."""
    from scripts.evaluate_height_q_batched import build_parser

    required = [
        "--artifact",
        "q",
        "--load_run",
        "run",
        "--checkpoint",
        "ckpt",
        "--output",
        "out",
    ]
    args = build_parser().parse_args(required)
    assert args.insert_step == 0
    assert args.repetitions == 4
    assert args.num_candidates == 8
    assert args.horizon == 500

    mid_gait = build_parser().parse_args([*required, "--insert_step", "100"])
    assert mid_gait.insert_step == 100
