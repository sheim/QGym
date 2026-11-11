"""Persistent selection and common-random-number contracts without a simulator."""

from types import SimpleNamespace

import torch

from learning.offline.persistent_intervention import (
    paired_survivor_tail_interval,
    rollout,
    sample_candidates,
    select_action,
    survivor_tail_interval,
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
