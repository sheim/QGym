"""Persistent selection and common-random-number contracts without a simulator."""

from types import SimpleNamespace

import pytest
import torch

from learning.offline.persistent_intervention import (
    paired_survivor_tail_interval,
    rejection_action_indices,
    rollout,
    rollout_mean,
    rollout_rejection,
    rollout_threshold,
    sample_candidates,
    sample_candidates_with_mean,
    sample_rejection_candidates,
    select_action,
    select_rejection_action,
    select_threshold_action,
    survivor_tail_interval,
    threshold_action_indices,
)


class Policy:
    actor = SimpleNamespace(std=torch.tensor([0.3, 0.7]))

    def mean_actions(self, obs):
        return obs[:, :2] * 2


class Critic:
    def __init__(self):
        self.calls = 0

    def predict(self, obs, actions):
        self.calls += 1
        return -(actions - obs[:, :2]).square().sum(-1)


def test_candidate_zero_marginal_pairing_and_independent_lanes():
    p = Policy()
    obs = torch.ones(32768, 2)
    a = sample_candidates(p, obs, torch.Generator().manual_seed(1))
    b = sample_candidates(p, obs + 3, torch.Generator().manual_seed(1))
    torch.testing.assert_close(a - 2, b - 8)
    selected, idx, _ = select_action(obs, a, None)
    assert torch.equal(selected, a[:, 0]) and not idx.any()
    torch.testing.assert_close(
        selected.mean(0), torch.tensor([2.0, 2.0]), atol=0.015, rtol=0
    )
    torch.testing.assert_close(selected.std(0), p.actor.std, atol=0.015, rtol=0)
    assert not torch.equal(a[:4096], a[4096:8192])


def test_selection_batch_equivalence_and_current_observation():
    obs = torch.randn(16, 2)
    a = sample_candidates(Policy(), obs, torch.Generator().manual_seed(22))
    q = Critic()
    action, idx, gain = select_action(obs, a, q)
    scores = torch.stack([q.predict(obs, a[:, c]) for c in range(8)], -1)
    assert torch.equal(idx, scores.argmax(-1))
    assert torch.equal(action, a[torch.arange(16), scores.argmax(-1)])
    torch.testing.assert_close(gain, scores.max(-1).values - scores[:, 0])


def test_every_decision_masks_absorbing_tail_and_history():
    class Env:
        device = "cpu"
        num_envs = 2
        commands = torch.zeros(2, 3)
        base_lin_vel = torch.ones(2, 3)
        base_ang_vel = torch.ones(2, 3)
        base_height = torch.tensor([[0.32], [0.2]])
        timed_out = torch.zeros(2, dtype=torch.bool)
        terminated = torch.zeros(2, dtype=torch.bool)
        steps = 0

        def set_states(self, names, action):
            self.action = action

        def step(self):
            self.steps += 1
            self.terminated = torch.tensor([self.steps == 2, False])

        def get_states(self, names):
            return torch.full((2, 2), float(self.steps))

    env = Env()
    snapshot = SimpleNamespace(restore=lambda: None, assert_restored=lambda: None)
    spec = SimpleNamespace(
        action_names=("a",),
        obs_names=("x",),
        obs_dims=(2,),
        noise={"x": 0.0},
        scales={"x": 1.0},
    )
    q = Critic()
    result = rollout(env, Policy(), spec, snapshot, torch.zeros(2, 2), q, 10, horizon=4)
    assert q.calls == 4
    torch.testing.assert_close(
        result["steps"], torch.tensor([2.0, 4.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        result["returns"],
        torch.tensor(
            [-(0.99**2) / 0.01, -sum(0.99**i for i in range(4))], dtype=torch.float64
        ),
    )
    assert result["selection_counts"].sum(1).tolist() == [2, 4]
    assert result["tracking_steps"].tolist() == [1, 4]
    assert result["survived"].tolist() == [False, True]


def test_survivor_tail_bounds_are_not_double_counted():
    tail = 0.99**500 / 0.01
    # Single controller: omitted tail [-tail, 0] for survivors only.
    low, high = survivor_tail_interval(-5.0, 0.25, tail)
    assert high == -5.0
    assert low == -5.0 - 0.25 * tail

    # Paired difference: analytic corner enumeration over two survivors.
    corners = torch.stack(
        [
            torch.tensor(-6.0 + t1 - (-5.0 + t2), dtype=torch.float64)
            for t1 in (-tail, 0.0)
            for t2 in (-tail, 0.0)
        ]
    )
    low, high = paired_survivor_tail_interval(-1.0, 1.0, 1.0, tail)
    atol = 1e-12
    assert abs(low - corners.min().item()) <= atol
    assert abs(high - corners.max().item()) <= atol
    assert abs((high - low) - 2 * tail) <= atol

    # Asymmetric survivor fractions set each side independently.
    low, high = paired_survivor_tail_interval(1.7, 0.5, 0.25, tail)
    assert abs(low - (1.7 - 0.5 * tail)) <= atol
    assert abs(high - (1.7 + 0.25 * tail)) <= atol


def test_nine_candidates_keep_eight_stochastic_draws_and_exact_mean():
    p = Policy()
    obs = torch.randn(64, 2)
    seed = 7
    nine = sample_candidates_with_mean(p, obs, torch.Generator().manual_seed(seed))
    eight = sample_candidates(p, obs, torch.Generator().manual_seed(seed))
    assert nine.shape == (64, 9, 2)
    torch.testing.assert_close(nine[:, 0], p.mean_actions(obs))
    # Index 1 is the old candidate0; the stochastic block must be byte-identical.
    torch.testing.assert_close(nine[:, 1:], eight)


def test_nine_candidates_reject_nonpositive_or_nonfinite_std():
    obs = torch.randn(4, 2)
    for bad in (0.0, -0.3, float("nan"), float("inf")):
        p = Policy()
        p.actor = SimpleNamespace(std=torch.tensor([bad, 0.7]))
        with pytest.raises(ValueError, match="finite and positive"):
            sample_candidates_with_mean(p, obs, torch.Generator().manual_seed(1))


def test_threshold_feasibility_distance_fallback_ties_and_negative_values():
    # Rows: mean infeasible, mean feasible, all infeasible, distance tie.
    candidates = torch.zeros(4, 9, 2)
    for index in range(1, 9):
        candidates[:, index, 0] = float(index)
    mean = candidates[:, 0]
    std = torch.tensor([1.0, 1.0])
    scores = torch.full((4, 9), -10.0)
    scores[0, 0] = -6.0  # mean below threshold
    scores[0, 1] = -4.0  # feasible, distance 1
    scores[0, 2] = -4.5  # feasible, distance 4
    scores[1, 0] = -4.0  # mean feasible, distance 0
    scores[1, 1] = -3.0
    scores[2, 5] = -8.0  # all below -5, fallback argmax
    # Tie: candidates 1 and 2 are both feasible and equidistant from the mean.
    candidates[3, 1, 0] = 1.0
    candidates[3, 2, 0] = -1.0
    scores[3, 1] = -4.0
    scores[3, 2] = -4.0
    out = threshold_action_indices(candidates, scores, mean, std, -5.0)
    assert out["indices"].tolist() == [1, 0, 5, 1]
    assert out["fallback"].tolist() == [False, False, True, False]
    assert out["mean_feasible"].tolist() == [False, True, False, False]
    assert out["stochastic_feasible"].tolist() == [True, True, False, True]

    # A more negative threshold is permissive: the same row0 mean still fails,
    # but an all-negative score row can now be feasible.
    permissive = threshold_action_indices(candidates, scores, mean, std, -9.0)
    assert permissive["fallback"].tolist() == [False, False, False, False]


def test_threshold_rejects_nonpositive_or_nonfinite_std_and_bad_scores():
    candidates = torch.zeros(2, 9, 2)
    scores = torch.zeros(2, 9)
    mean = candidates[:, 0]
    for bad in (torch.tensor([0.0, 1.0]), torch.tensor([float("nan"), 1.0])):
        with pytest.raises(ValueError, match="finite and positive"):
            threshold_action_indices(candidates, scores, mean, bad, -5.0)
    with pytest.raises(ValueError, match="scalar or \\[action_dim\\]"):
        threshold_action_indices(candidates, scores, mean, torch.ones(2, 2), -5.0)
    bad_scores = scores.clone()
    bad_scores[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        threshold_action_indices(candidates, bad_scores, mean, torch.ones(2), -5.0)


def test_select_threshold_action_scores_all_nine_and_returns_chosen_action():
    class ActionCritic:
        def predict(self, obs, actions):
            return actions[:, 0]

    obs = torch.zeros(3, 2)
    candidates = torch.zeros(3, 9, 2)
    for index in range(1, 9):
        candidates[:, index, 0] = float(index) * 0.1
    # Mean Q is 0; candidate Q rises with index. Threshold 0.45 is feasible
    # only for candidates 5 and above; candidate 5 is closest to the mean.
    candidates[:, 0, 0] = 0.0
    result = select_threshold_action(
        obs, candidates, ActionCritic(), 0.45, torch.tensor([1.0, 1.0])
    )
    assert result["indices"].tolist() == [5, 5, 5]
    torch.testing.assert_close(
        result["action"], candidates[torch.arange(3), result["indices"]]
    )
    assert result["mean_feasible"].tolist() == [False, False, False]


def test_rejection_selects_first_feasible_not_best_not_closest():
    scores = torch.full((5, 8), -10.0)
    # Row 0: feasible at 2 and 5; index 5 has the higher Q but rejection takes 2.
    scores[0, 2] = -4.0
    scores[0, 5] = -1.0
    # Row 1: no feasible candidate -> argmax fallback.
    scores[1, 6] = -6.0
    # Row 2: feasible at 1 and 3; first in fixed draw order wins.
    scores[2, 1] = -4.0
    scores[2, 3] = -1.0
    # Row 3: only the last candidate is feasible.
    scores[3, 7] = -4.5
    # Row 4: exact threshold equality is feasible.
    scores[4, 4] = -5.0
    out = rejection_action_indices(scores, -5.0)
    assert out["indices"].tolist() == [2, 6, 1, 7, 4]
    assert out["fallback"].tolist() == [False, True, False, False, False]
    assert out["attempts"].tolist() == [3, 8, 2, 8, 5]

    # Selector-level controls: permissive threshold keeps the first draw;
    # restrictive threshold reduces to the eight-way max-Q fallback.
    assert rejection_action_indices(scores, -float("inf"))["indices"].tolist() == [
        0,
        0,
        0,
        0,
        0,
    ]
    assert rejection_action_indices(scores, float("inf"))["indices"].tolist() == [
        5,
        6,
        3,
        7,
        4,
    ]
    with pytest.raises(ValueError, match="NaN"):
        rejection_action_indices(scores, float("nan"))


def test_rejection_candidates_share_the_paired_stochastic_block():
    p = Policy()
    obs = torch.randn(32, 2)
    seed = 17
    rejection, mean = sample_rejection_candidates(
        p, obs, torch.Generator().manual_seed(seed)
    )
    eight = sample_candidates(p, obs, torch.Generator().manual_seed(seed))
    nine = sample_candidates_with_mean(p, obs, torch.Generator().manual_seed(seed))
    torch.testing.assert_close(rejection, eight)
    torch.testing.assert_close(rejection, nine[:, 1:])
    torch.testing.assert_close(mean, p.mean_actions(obs))
    # The mean is never part of the rejection candidate axis.
    assert rejection.shape == (32, 8, 2)


def test_select_rejection_action_scores_eight_and_returns_chosen_action():
    class ActionCritic:
        def predict(self, obs, actions):
            return actions[:, 0]

    obs = torch.zeros(3, 2)
    candidates = torch.stack(
        [torch.full((3, 2), float(index) * 0.1) for index in range(8)], 1
    )
    mean = torch.full((3, 2), 99.0)  # diagnostics only, not selected
    result = select_rejection_action(obs, candidates, mean, ActionCritic(), 0.45)
    assert result["indices"].tolist() == [5, 5, 5]
    torch.testing.assert_close(
        result["action"], candidates[torch.arange(3), result["indices"]]
    )
    torch.testing.assert_close(result["selected_q"], torch.full((3,), 0.5))


def test_rejection_and_plain_negative_controls_match_existing_rollouts():
    class Env:
        device = "cpu"
        num_envs = 2
        commands = torch.zeros(2, 3)
        base_lin_vel = torch.ones(2, 3)
        base_ang_vel = torch.ones(2, 3)
        base_height = torch.tensor([[0.32], [0.2]])
        timed_out = torch.zeros(2, dtype=torch.bool)
        terminated = torch.zeros(2, dtype=torch.bool)
        steps = 0

        def set_states(self, names, action):
            self.action = action

        def step(self):
            self.steps += 1
            self.terminated = torch.tensor([self.steps == 2, False])

        def get_states(self, names):
            return torch.full((2, 2), float(self.steps))

    env = Env()
    snapshot = SimpleNamespace(restore=lambda: None, assert_restored=lambda: None)
    spec = SimpleNamespace(
        action_names=("a",),
        obs_names=("x",),
        obs_dims=(2,),
        noise={"x": 0.0},
        scales={"x": 1.0},
    )
    q = Critic()

    def reset_env():
        env.steps = 0
        env.terminated = torch.zeros(2, dtype=torch.bool)

    reset_env()
    plain = rollout(env, Policy(), spec, snapshot, torch.zeros(2, 2), None, 10, 4)
    reset_env()
    permissive = rollout_rejection(
        env,
        Policy(),
        spec,
        snapshot,
        torch.zeros(2, 2),
        q,
        10,
        -1e9,
        horizon=4,
    )
    for key in ("returns", "first_rewards", "survived", "selection_counts", "steps"):
        torch.testing.assert_close(permissive[key], plain[key], atol=0, rtol=0)
    assert permissive["fallback_sum"].sum() == 0
    assert permissive["attempt_counts"][:, 0].sum() == permissive["steps"].sum()

    reset_env()
    best8 = rollout(env, Policy(), spec, snapshot, torch.zeros(2, 2), q, 10, 4)
    reset_env()
    restrictive = rollout_rejection(
        env,
        Policy(),
        spec,
        snapshot,
        torch.zeros(2, 2),
        q,
        10,
        1e9,
        horizon=4,
    )
    for key in ("returns", "first_rewards", "survived", "selection_counts", "steps"):
        torch.testing.assert_close(restrictive[key], best8[key], atol=0, rtol=0)
    torch.testing.assert_close(restrictive["fallback_sum"], restrictive["steps"])
    torch.testing.assert_close(
        restrictive["attempt_counts"][:, 7], restrictive["steps"]
    )


def test_threshold_and_mean_rollouts_every_decision_and_counts():
    class Env:
        device = "cpu"
        num_envs = 2
        commands = torch.zeros(2, 3)
        base_lin_vel = torch.ones(2, 3)
        base_ang_vel = torch.ones(2, 3)
        base_height = torch.tensor([[0.32], [0.2]])
        timed_out = torch.zeros(2, dtype=torch.bool)
        terminated = torch.zeros(2, dtype=torch.bool)
        steps = 0

        def set_states(self, names, action):
            self.action = action

        def step(self):
            self.steps += 1
            self.terminated = torch.tensor([self.steps == 2, False])

        def get_states(self, names):
            return torch.full((2, 2), float(self.steps))

    env = Env()
    snapshot = SimpleNamespace(restore=lambda: None, assert_restored=lambda: None)
    spec = SimpleNamespace(
        action_names=("a",),
        obs_names=("x",),
        obs_dims=(2,),
        noise={"x": 0.0},
        scales={"x": 1.0},
    )
    q = Critic()

    def reset_env():
        env.steps = 0
        env.terminated = torch.zeros(2, dtype=torch.bool)

    # Strict threshold: no candidate feasible, so every live decision falls back.
    reset_env()
    strict = rollout_threshold(
        env,
        Policy(),
        spec,
        snapshot,
        torch.zeros(2, 2),
        q,
        10,
        threshold=1.0,
        horizon=4,
    )
    assert strict["selection_counts"].shape == (2, 9)
    torch.testing.assert_close(
        strict["selection_counts"].sum(1).double(), strict["steps"]
    )
    torch.testing.assert_close(strict["fallback_sum"], strict["steps"])
    torch.testing.assert_close(
        strict["mean_selected_sum"], torch.zeros(2, dtype=torch.float64)
    )
    torch.testing.assert_close(
        strict["fallback_sum"]
        + strict["mean_selected_sum"]
        + strict["stochastic_selected_sum"],
        strict["steps"],
    )

    # Permissive threshold: the exact mean is always feasible and selected.
    reset_env()
    loose = rollout_threshold(
        env,
        Policy(),
        spec,
        snapshot,
        torch.zeros(2, 2),
        q,
        10,
        threshold=-1e9,
        horizon=4,
    )
    torch.testing.assert_close(loose["mean_selected_sum"], loose["steps"])
    torch.testing.assert_close(loose["selection_counts"][:, 0], loose["steps"].long())
    torch.testing.assert_close(
        loose["fallback_sum"], torch.zeros(2, dtype=torch.float64)
    )

    # Mean control never samples a candidate and always selects index 0.
    reset_env()
    mean = rollout_mean(env, Policy(), spec, snapshot, torch.zeros(2, 2), 10, horizon=4)
    assert mean["selection_counts"].shape == (2, 1)
    torch.testing.assert_close(mean["selection_counts"][:, 0], mean["steps"].long())
    torch.testing.assert_close(mean["mean_selected_sum"], mean["steps"])
    torch.testing.assert_close(
        mean["returns"],
        torch.tensor(
            [-(0.99**2) / 0.01, -sum(0.99**i for i in range(4))], dtype=torch.float64
        ),
    )
