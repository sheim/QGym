"""Frozen-policy sample-eight control; candidate axis never enters physics."""

import torch

from learning.offline.evaluation import _policy_observation
from learning.offline.fixed_state_variance import accumulate_return


def sample_candidates(policy, obs, generator):
    mean = policy.mean_actions(obs)
    eps = torch.randn(
        obs.shape[0], 8, mean.shape[-1], device=obs.device, generator=generator
    )
    return mean[:, None] + policy.actor.std.detach() * eps


def survivor_tail_interval(mean_return, survivor_fraction, tail_width):
    """Mean interval from omitted survivor tails in [-tail_width, 0]."""
    if not 0.0 <= survivor_fraction <= 1.0:
        raise ValueError("survivor fraction must be in [0, 1]")
    return (
        float(mean_return - tail_width * survivor_fraction),
        float(mean_return),
    )


def paired_survivor_tail_interval(gain, base_fraction, controller_fraction, tail_width):
    """Paired mean-difference interval with asymmetric survivor bounds.

    Each execution's omitted tail is [-tail_width, 0]. In a baseline/controller
    pair the lower bound is set by the baseline survivor and the upper bound by
    the controller survivor; adding both fractions to both sides doubles the
    width.
    """
    for fraction in (base_fraction, controller_fraction):
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("survivor fraction must be in [0, 1]")
    return (
        float(gain - tail_width * base_fraction),
        float(gain + tail_width * controller_fraction),
    )


def select_action(obs, candidates, critic):
    """Candidate0 has exactly the nominal stochastic marginal, not the mean."""
    if critic is None:
        indices = torch.zeros(obs.shape[0], dtype=torch.long, device=obs.device)
        gain = torch.zeros(obs.shape[0], device=obs.device)
    else:
        scores = critic.predict(
            obs[:, None].expand(-1, 8, -1).flatten(0, 1),
            candidates.flatten(0, 1),
        ).reshape(-1, 8)
        indices = scores.argmax(-1)
        gain = scores.amax(-1) - scores[:, 0]
    return (
        candidates[torch.arange(obs.shape[0], device=obs.device), indices],
        indices,
        gain,
    )


def sample_rejection_candidates(policy, obs, generator):
    """Eight stochastic draws in the fixed order of the rejection protocol.

    The draw block is byte-identical to ``sample_candidates`` so a paired seed
    gives the same eight actions to every threshold controller. The policy mean
    is returned for diagnostics only; it is never a candidate here.
    """
    mean = policy.mean_actions(obs)
    eps = torch.randn(
        obs.shape[0], 8, mean.shape[-1], device=obs.device, generator=generator
    )
    return mean[:, None] + policy.actor.std.detach() * eps, mean


def rejection_action_indices(scores, threshold):
    """First feasible candidate in fixed sample order, else max-Q fallback.

    This is the bounded sequential rejection rule evaluated in parallel: sampling
    all eight candidates and taking the first ``Q >= threshold`` has the same
    distribution as drawing and testing them one at a time. It is deliberately
    not "highest Q among feasible" and not "closest to the mean". Ties in the
    fallback select the lowest index. ``threshold`` may be infinite for control
    runs; NaN is rejected.
    """
    if scores.ndim != 2 or scores.shape[1] != 8:
        raise ValueError("rejection selection requires eight candidates")
    if not torch.isfinite(scores).all():
        raise ValueError("candidate scores must be finite")
    if threshold != threshold:
        raise ValueError("threshold must not be NaN")
    feasible = scores >= threshold
    any_feasible = feasible.any(-1)
    first = feasible.to(torch.int8).argmax(-1)
    indices = torch.where(any_feasible, first, scores.argmax(-1))
    attempts = torch.where(
        any_feasible,
        first + 1,
        torch.full_like(first, 8),
    )
    return {
        "indices": indices,
        "feasible": feasible,
        "fallback": ~any_feasible,
        "attempts": attempts,
    }


def sample_candidates_with_mean(policy, obs, generator):
    """Exact policy mean at index 0, then the same eight stochastic draws.

    The stochastic block is drawn exactly like ``sample_candidates`` so a
    paired seed keeps the original candidate0 (first draw) at index 1.
    """
    mean = policy.mean_actions(obs)
    std = policy.actor.std.detach()
    if not torch.isfinite(std).all() or not (std > 0).all():
        raise ValueError("action standard deviation must be finite and positive")
    eps = torch.randn(
        obs.shape[0], 8, mean.shape[-1], device=obs.device, generator=generator
    )
    stochastic = mean[:, None] + std * eps
    return torch.cat((mean[:, None], stochastic), dim=1)


def score_candidates(critic, obs, candidates):
    """Score every candidate action against the same current observation."""
    n, count, _ = candidates.shape
    scores = critic.predict(
        obs[:, None].expand(-1, count, -1).flatten(0, 1),
        candidates.flatten(0, 1),
    )
    return scores.reshape(n, count)


def threshold_action_indices(candidates, scores, mean, std, threshold):
    """Return the closest feasible candidate, else the max-Q fallback.

    Feasibility is ``Q >= threshold``. Distance is the policy-normalized
    squared action norm ``sum(((a - mean) / std) ** 2)``; the exact mean has
    distance zero and is therefore selected whenever it is feasible.
    Ties select the lowest candidate index.
    """
    if candidates.ndim != 3 or candidates.shape[1] != 9:
        raise ValueError("threshold selection requires nine candidates")
    if scores.shape != candidates.shape[:2]:
        raise ValueError("scores must match candidates [batch, 9]")
    if mean.shape != (candidates.shape[0], candidates.shape[2]):
        raise ValueError("mean must match candidates [batch, action_dim]")
    if std.ndim == 1 and std.shape[0] != candidates.shape[2]:
        raise ValueError("std must be scalar or [action_dim]")
    if std.ndim > 1:
        raise ValueError("std must be scalar or [action_dim]")
    if not torch.isfinite(std).all() or not (std > 0).all():
        raise ValueError("action standard deviation must be finite and positive")
    if not torch.isfinite(scores).all():
        raise ValueError("candidate scores must be finite")
    feasible = scores >= threshold
    scaled = (candidates - mean[:, None]) / std
    distance = scaled.square().sum(-1)
    masked = torch.where(feasible, distance, torch.full_like(distance, torch.inf))
    constrained = masked.argmin(-1)
    any_feasible = feasible.any(-1)
    indices = torch.where(any_feasible, constrained, scores.argmax(-1))
    return {
        "indices": indices,
        "feasible": feasible,
        "fallback": ~any_feasible,
        "mean_feasible": feasible[:, 0],
        "stochastic_feasible": feasible[:, 1:].any(-1),
        "distance": distance,
    }


def select_rejection_action(obs, candidates, mean, critic, threshold):
    """Rejection selection over the eight stochastic candidates only."""
    scores = score_candidates(critic, obs, candidates)
    result = rejection_action_indices(scores, threshold)
    result["action"] = candidates[
        torch.arange(candidates.shape[0], device=candidates.device), result["indices"]
    ]
    result["scores"] = scores
    result["selected_q"] = scores.gather(1, result["indices"][:, None]).squeeze(1)
    result["gain"] = scores.amax(-1) - scores[:, 0]
    return result


def select_threshold_action(obs, candidates, critic, threshold, std):
    """Constrained selection over mean plus eight stochastic candidates."""
    scores = score_candidates(critic, obs, candidates)
    result = threshold_action_indices(
        candidates, scores, candidates[:, 0], std, threshold
    )
    result["action"] = candidates[
        torch.arange(candidates.shape[0], device=candidates.device), result["indices"]
    ]
    result["scores"] = scores
    result["selected_q"] = scores.gather(1, result["indices"][:, None]).squeeze(1)
    result["selected_distance"] = (
        result["distance"].gather(1, result["indices"][:, None]).squeeze(1)
    )
    result["gain"] = scores.amax(-1) - scores[:, 0]
    return result


def _rollout_common(
    env, policy, spec, snapshot, initial_obs, seed, horizon, choose, n_candidates
):
    snapshot.restore()
    snapshot.assert_restored()
    torch.manual_seed(seed)
    obs_rng = torch.Generator(device=env.device).manual_seed(seed + 1)
    act_rng = torch.Generator(device=env.device).manual_seed(seed + 2)
    n = env.num_envs
    alive = torch.ones(n, dtype=torch.bool, device=env.device)
    sums = {
        key: torch.zeros(n, dtype=torch.float64, device=env.device)
        for key in (
            "returns",
            "steps",
            "oob",
            "tracking_steps",
            "xy_error_sq",
            "yaw_error_sq",
            "speed_sum",
            "command_speed_sum",
            "q_gain_sum",
        )
    }
    sums["termination_step"] = torch.full(
        (n,), float(horizon), dtype=torch.float64, device=env.device
    )
    selection = torch.zeros(n, n_candidates, dtype=torch.int64, device=env.device)
    extras = {}
    obs = initial_obs
    first_reward = None
    first_terminated = None
    timeout_seen = torch.zeros(n, dtype=torch.bool, device=env.device)
    for step in range(horizon):
        action, index, gains, extra_values = choose(obs, act_rng)
        selection.scatter_add_(1, index[:, None], alive.long()[:, None])
        for key, value in gains.items():
            sums[key] += value * alive
        for key, value in extra_values.items():
            if key not in extras:
                extras[key] = torch.zeros(
                    (n, *value.shape[1:]), dtype=torch.float64, device=env.device
                )
            if value.ndim == 1:
                extras[key] += value * alive
            else:
                # Per-decision vectors (for example an attempt histogram) keep
                # their trailing axis and are masked by the same liveness mask.
                extras[key] += value * alive[:, None]
        # Commands may resample during step; compare successor velocity with the
        # command that actually governed this interval, excluding terminal steps.
        command = env.commands.clone()
        env.set_states(spec.action_names, action)
        env.step()
        reward = -((env.base_height[:, 0] - 0.32).abs() > 0.05).double()
        if step == 0:
            first_reward = reward.clone()
            first_terminated = env.terminated.clone()
        timeout_seen |= env.timed_out
        sums["termination_step"] = torch.where(
            alive & env.terminated,
            torch.full_like(sums["termination_step"], float(step)),
            sums["termination_step"],
        )
        sums["steps"] += alive
        sums["oob"] += -reward * alive
        sums["returns"], alive = accumulate_return(
            sums["returns"], alive, reward, env.terminated, step, 0.99
        )
        sums["tracking_steps"] += alive
        sums["xy_error_sq"] += (env.base_lin_vel[:, :2] - command[:, :2]).square().sum(
            -1
        ) * alive
        sums["yaw_error_sq"] += (
            env.base_ang_vel[:, 2] - command[:, 2]
        ).square() * alive
        sums["speed_sum"] += env.base_lin_vel[:, :2].norm(dim=-1) * alive
        sums["command_speed_sum"] += command[:, :2].norm(dim=-1) * alive
        if step != horizon - 1:
            obs = _policy_observation(
                env, spec.obs_names, spec.obs_dims, spec.noise, spec.scales, obs_rng
            )
    sums.update(extras)
    sums.update(
        survived=alive,
        first_rewards=first_reward,
        first_terminated=first_terminated,
        selection_counts=selection,
        timeout_seen=timeout_seen,
    )
    if timeout_seen.any() or not all(torch.isfinite(v).all() for v in sums.values()):
        raise RuntimeError("timeout or nonfinite persistent rollout")
    return {k: v.cpu() for k, v in sums.items()}


@torch.no_grad()
def rollout(env, policy, spec, snapshot, initial_obs, critic, seed, horizon=500):
    def choose(obs, act_rng):
        candidates = sample_candidates(policy, obs, act_rng)
        action, index, gain = select_action(obs, candidates, critic)
        return action, index, {"q_gain_sum": gain}, {}

    return _rollout_common(
        env, policy, spec, snapshot, initial_obs, seed, horizon, choose, 8
    )


@torch.no_grad()
def rollout_rejection(
    env, policy, spec, snapshot, initial_obs, critic, seed, threshold, horizon=500
):
    """Execute the first Q-feasible stochastic draw, else the max-Q draw.

    The mean is not a candidate. Attempts 1..8 are recorded for every live
    decision, with a fallback counted as eight attempts.
    """

    def choose(obs, act_rng):
        candidates, mean = sample_rejection_candidates(policy, obs, act_rng)
        result = select_rejection_action(obs, candidates, mean, critic, threshold)
        extra = {
            "fallback_sum": result["fallback"].double(),
            "selected_q_sum": result["selected_q"],
            "attempt_counts": torch.nn.functional.one_hot(
                result["attempts"] - 1, 8
            ).double(),
        }
        return (
            result["action"],
            result["indices"],
            {"q_gain_sum": result["gain"]},
            extra,
        )

    return _rollout_common(
        env, policy, spec, snapshot, initial_obs, seed, horizon, choose, 8
    )


@torch.no_grad()
def rollout_threshold(
    env, policy, spec, snapshot, initial_obs, critic, seed, threshold, horizon=500
):
    std = policy.actor.std.detach()

    def choose(obs, act_rng):
        candidates = sample_candidates_with_mean(policy, obs, act_rng)
        result = select_threshold_action(obs, candidates, critic, threshold, std)
        extra = {
            "fallback_sum": result["fallback"].double(),
            "mean_selected_sum": (
                (result["indices"] == 0) & ~result["fallback"]
            ).double(),
            "stochastic_selected_sum": (
                (result["indices"] != 0) & ~result["fallback"]
            ).double(),
            "mean_feasible_sum": result["mean_feasible"].double(),
            "stochastic_feasible_sum": result["stochastic_feasible"].double(),
            "selected_q_sum": result["selected_q"],
            "selected_distance_sum": result["selected_distance"],
        }
        return (
            result["action"],
            result["indices"],
            {"q_gain_sum": result["gain"]},
            extra,
        )

    return _rollout_common(
        env, policy, spec, snapshot, initial_obs, seed, horizon, choose, 9
    )


@torch.no_grad()
def rollout_mean(env, policy, spec, snapshot, initial_obs, seed, horizon=500):
    def choose(obs, act_rng):
        action = policy.mean_actions(obs)
        index = torch.zeros(obs.shape[0], dtype=torch.long, device=obs.device)
        gain = torch.zeros(obs.shape[0], device=obs.device)
        return (
            action,
            index,
            {"q_gain_sum": gain},
            {"mean_selected_sum": torch.ones_like(gain, dtype=torch.float64)},
        )

    return _rollout_common(
        env, policy, spec, snapshot, initial_obs, seed, horizon, choose, 1
    )
