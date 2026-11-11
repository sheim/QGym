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


@torch.no_grad()
def rollout(env, policy, spec, snapshot, initial_obs, critic, seed, horizon=500):
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
    selection = torch.zeros(n, 8, dtype=torch.int64, device=env.device)
    obs = initial_obs
    first_reward = None
    first_terminated = None
    timeout_seen = torch.zeros(n, dtype=torch.bool, device=env.device)
    for step in range(horizon):
        candidates = sample_candidates(policy, obs, act_rng)
        action, index, gain = select_action(obs, candidates, critic)
        selection.scatter_add_(1, index[:, None], alive.long()[:, None])
        sums["q_gain_sum"] += gain * alive
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
