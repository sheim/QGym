"""Portable contracts for exact training-reward dispatch and dual-objective returns."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from learning.offline.reward_tradeoff import rollout_tradeoff
from learning.offline.training_reward import (
    TrainingRewardModel,
    nonzero_weights,
    reward_term_order,
)
from learning.runners.on_policy_runner import OnPolicyRunner


class RewardEnv:
    device = "cpu"
    dt = 0.02
    num_envs = 3
    terminated = torch.tensor([False, True, False])

    def _reward_tracking(self):
        return torch.tensor([1.0, 99.0, 0.0])

    def _reward_penalty(self):
        return torch.tensor([-2.0, -99.0, -4.0])

    def _reward_termination(self):
        return -self.terminated.float()


def model():
    return TrainingRewardModel(
        RewardEnv(),
        {"tracking": 4.0, "penalty": 0.5, "unused": 0.0},
        {"termination": 0.01},
        gamma=0.9,
    )


def test_weight_masks_dt_scale_and_zero_weights():
    terms = model().step_terms(RewardEnv())
    assert set(terms) == {"tracking", "penalty", "termination"}
    torch.testing.assert_close(terms["tracking"], torch.tensor([0.08, 0.0, 0.0]))
    torch.testing.assert_close(terms["penalty"], torch.tensor([-0.02, 0.0, -0.04]))
    torch.testing.assert_close(terms["termination"], torch.tensor([0.0, -0.0002, 0.0]))
    torch.testing.assert_close(
        model().total(terms), torch.tensor([0.06, -0.0002, -0.04])
    )
    assert nonzero_weights({"a": 0.0, "b": -1.0}) == {"b": -1.0}
    assert reward_term_order({"a": 1.0}, {"t": 1.0}) == ("a", "t")


def test_missing_reward_function_fails_at_construction():
    with pytest.raises(KeyError, match="missing reward functions"):
        TrainingRewardModel(
            RewardEnv(), {"missing": 1.0}, {"termination": 0.01}, gamma=0.99
        )


def test_dispatch_matches_real_on_policy_runner_update():
    env = RewardEnv()
    weights = {"tracking": 4.0, "penalty": 0.5, "unused": 0.0}
    termination = {"termination": 0.01}
    runner = object.__new__(OnPolicyRunner)
    runner.env = env
    runner.device = env.device
    runner.setup_reward_functions()
    runner.critic_cfg = {
        "reward": {"weights": dict(weights), "termination_weight": dict(termination)}
    }
    from learning.utils import remove_zero_weighted_rewards

    remove_zero_weighted_rewards(runner.critic_cfg["reward"]["weights"])
    remove_zero_weighted_rewards(runner.critic_cfg["reward"]["termination_weight"])
    rewards = TensorDict({}, batch_size=(1, env.num_envs), device=env.device)
    for key in list(runner.critic_cfg["reward"]["termination_weight"]) + list(
        runner.critic_cfg["reward"]["weights"]
    ):
        rewards[key] = torch.zeros(1, env.num_envs)
    runner.update_rewards_dict(rewards, 0)
    expected = model().step_terms(env)
    for name, value in expected.items():
        assert torch.equal(rewards[0][name], value), name


class Policy:
    actor = SimpleNamespace(std=torch.tensor([0.1, 0.1]))

    def mean_actions(self, obs):
        return torch.zeros_like(obs)


class StepEnv:
    device = "cpu"
    num_envs = 2
    dt = 0.02
    commands = torch.zeros(2, 3)
    base_lin_vel = torch.zeros(2, 3)
    base_ang_vel = torch.zeros(2, 3)
    base_height = torch.tensor([[0.32], [0.32]])
    timed_out = torch.zeros(2, dtype=torch.bool)
    terminated = torch.zeros(2, dtype=torch.bool)

    def __init__(self, timeout_at=None):
        self.steps = 0
        self.timeout_at = timeout_at

    def set_states(self, names, action):
        self.action = action

    def step(self):
        self.steps += 1
        self.terminated = torch.tensor(
            [self.steps == 2, self.steps == 3], dtype=torch.bool
        )
        if self.timeout_at is not None and self.steps == self.timeout_at:
            self.timed_out = torch.tensor([True, False])

    def get_states(self, names):
        return torch.zeros(2, 2)

    def _reward_pos(self):
        return torch.ones(2)

    def _reward_neg(self):
        return -torch.ones(2)

    def _reward_termination(self):
        return -self.terminated.float()


def rollout(env, *, horizon=5, trace=False):
    reward_model = TrainingRewardModel(
        env, {"pos": 2.0, "neg": 1.0}, {"termination": 0.01}, gamma=0.9
    )
    snapshot = SimpleNamespace(restore=lambda: None, assert_restored=lambda: None)
    spec = SimpleNamespace(
        action_names=("action",),
        obs_names=("obs",),
        obs_dims=(2,),
        noise={"obs": 0.0},
        scales={"obs": 1.0},
    )
    return rollout_tradeoff(
        env,
        Policy(),
        spec,
        snapshot,
        torch.zeros(2, 2),
        None,
        10,
        reward_model=reward_model,
        horizon=horizon,
        trace=trace,
    )


def test_terminal_state_zero_original_reward_and_exact_term_sum():
    result = rollout(StepEnv(), horizon=5)
    # pos and neg are both 1/-1, so each live nonterminal step contributes
    # dt * (2 - 1) = 0.02. Environment 0 terminates at step 1 (0-index) and
    # environment 1 at step 2; termination adds dt * 0.01 * -1 at that step.
    expected = torch.tensor(
        [
            0.02 + 0.9 * (-0.0002),
            0.02 + 0.9 * 0.02 + 0.9**2 * (-0.0002),
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(result["original_returns"], expected)
    assert result["steps"].tolist() == [2.0, 3.0]
    assert result["termination_step"].tolist() == [1.0, 2.0]
    assert result["survived"].tolist() == [False, False]
    term_sum = result["original_terms"].sum(-1)
    torch.testing.assert_close(term_sum, result["original_returns"], atol=0, rtol=0)
    names = result["term_names"]
    assert names == ["pos", "neg", "termination"]
    # The dispatch test above establishes the terminal-only slot; here the
    # per-term discounted sums exactly rebuild the reported original return.
    assert result["original_terms"][0, names.index("termination")] < 0
    assert result["original_terms"][1, names.index("termination")] < 0


def test_no_absorbing_tail_and_no_reward_after_terminal():
    result = rollout(StepEnv(), horizon=5)
    assert result["original_returns"].min() > -0.1
    assert result["original_returns"].max() < 0.1
    # Height still uses the absorbing-failure convention; this is a different
    # objective on the same trajectory.
    assert result["height_returns"].min() < -7


def test_timeout_is_a_hard_failure():
    with pytest.raises(RuntimeError, match="timeout"):
        rollout(StepEnv(timeout_at=1), horizon=3)


def test_short_trace_is_deterministic():
    env = StepEnv()
    first = rollout(env, horizon=5, trace=True)
    second = rollout(StepEnv(), horizon=5, trace=True)
    assert torch.equal(first["actions"], second["actions"])
    assert torch.equal(first["original_terms"], second["original_terms"])
    assert torch.equal(first["height_returns"], second["height_returns"])


def test_student_t_and_state_bootstrap_contracts():
    from learning.offline.tradeoff_analysis import (
        paired_t_summary,
        state_cluster_bootstrap,
        student_t_critical,
        student_t_two_sided_p,
    )

    critical = student_t_critical(0.95, 127)
    assert abs(critical - 1.9788195) < 1e-5
    assert student_t_two_sided_p(0.0, 127) == 1.0
    assert abs(student_t_two_sided_p(critical, 127) - 0.05) < 1e-10
    summary = paired_t_summary(torch.linspace(-1.0, 1.0, 128))
    assert summary["df"] == 127 and abs(summary["mean"]) < 1e-15
    first = state_cluster_bootstrap(torch.arange(128.0), 7, draws=2000)
    second = state_cluster_bootstrap(torch.arange(128.0), 7, draws=2000)
    assert first == second
    assert first[0] < 63.5 < first[1]
