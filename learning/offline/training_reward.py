"""Exact post-step training-reward dispatch for offline fixed-state rollouts.

``OnPolicyRunner.update_rewards_dict`` applies ordinary weights under
``~env.terminated`` and termination weights under ``env.terminated``, then
multiplies every term by ``env.dt``. Reward methods are read-only consumers of
post-step task state, so this helper must never mutate environment buffers.
"""

from collections.abc import Mapping

import torch


def reward_functions(env):
    """Return the runner's name-to-method reward map for one environment."""
    functions = {}
    for method in dir(env):
        if not method.startswith("_reward_"):
            continue
        function = getattr(env, method)
        if callable(function):
            functions[method.removeprefix("_reward_")] = function
    return functions


def nonzero_weights(weights: Mapping[str, float]):
    """Drop exactly-zero configured weights, matching the runner's parse step."""
    return {
        str(name): float(weight)
        for name, weight in weights.items()
        if float(weight) != 0.0
    }


def reward_term_order(weights, termination_weights):
    """Stable per-term order: ordinary terms then termination terms."""
    return tuple(nonzero_weights(weights)) + tuple(nonzero_weights(termination_weights))


class TrainingRewardModel:
    """Weighted training reward terms exactly as the on-policy runner forms them.

    The configured gamma and dt are required rather than defaulted. Missing
    reward methods fail at construction, and the dispatch is parity-tested
    against ``OnPolicyRunner.update_rewards_dict`` with a real environment step.
    """

    def __init__(
        self,
        env,
        weights: Mapping[str, float],
        termination_weights: Mapping[str, float],
        *,
        gamma: float,
    ):
        self.dt = float(env.dt)
        self.gamma = float(gamma)
        if not self.dt > 0.0:
            raise ValueError("environment dt must be positive")
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("training gamma must be in (0, 1)")
        self.weights = nonzero_weights(weights)
        self.termination_weights = nonzero_weights(termination_weights)
        if not self.weights and not self.termination_weights:
            raise ValueError("no nonzero training reward weights")
        self.functions = reward_functions(env)
        missing = (set(self.weights) | set(self.termination_weights)) - set(
            self.functions
        )
        if missing:
            raise KeyError(f"missing reward functions: {sorted(missing)}")
        self.term_names = reward_term_order(weights, termination_weights)

    def step_terms(self, env):
        """Return weighted, dt-scaled terms for the current post-step state.

        The multiplication order is ``mask * ((weight * dt) * f())`` to match
        ``BaseRunner.get_rewards``. Ordinary terms are exactly zero on a
        terminal transition; termination terms are exactly zero otherwise.
        """
        terminated = env.terminated
        if terminated.ndim != 1:
            raise ValueError("environment terminated mask must be one-dimensional")
        terms = {}
        for name, weight in self.weights.items():
            terms[name] = (~terminated) * (weight * self.dt * self.functions[name]())
        for name, weight in self.termination_weights.items():
            terms[name] = terminated * (weight * self.dt * self.functions[name]())
        return terms

    def total(self, terms):
        """Sum the ordered terms with the same order used for reporting."""
        return torch.stack([terms[name] for name in self.term_names]).sum(dim=0)
