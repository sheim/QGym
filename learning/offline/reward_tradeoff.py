"""Dual-objective fixed-state rollout: original training reward and height return.

The trajectory, candidate draws, action selection and environment steps are
identical to the persistent-intervention rollout. Only accounting differs:

* original PPO return: ``sum_t gamma**t * training_reward_t``, where
  ``training_reward_t`` is the runner's dt-scaled masked term sum, and every
  reward after a true termination is zero;
* height return: the existing binary band reward with its discounted
  absorbing-failure tail.

No per-step state tensors are retained.
"""

import torch

from learning.offline.evaluation import _policy_observation
from learning.offline.fixed_state_variance import accumulate_return
from learning.offline.persistent_intervention import sample_candidates, select_action


@torch.no_grad()
def rollout_tradeoff(
    env,
    policy,
    spec,
    snapshot,
    initial_obs,
    critic,
    seed,
    *,
    reward_model,
    reference_height=0.32,
    height_bound=0.05,
    horizon=500,
    trace=False,
):
    """Run one controller and record both objectives on the same trajectory.

    ``critic=None`` selects candidate 0 of the eight shared draws (plain). A
    critic selects its argmax (original height Q). ``trace`` is diagnostic only
    and retains the selected actions, never full state.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0.0 < reward_model.gamma < 1.0:
        raise ValueError("gamma must be in (0, 1)")
    snapshot.restore()
    snapshot.assert_restored()
    torch.manual_seed(seed)
    obs_rng = torch.Generator(device=env.device).manual_seed(seed + 1)
    act_rng = torch.Generator(device=env.device).manual_seed(seed + 2)
    n = env.num_envs
    alive = torch.ones(n, dtype=torch.bool, device=env.device)
    term_sums = {
        name: torch.zeros(n, dtype=torch.float64, device=env.device)
        for name in reward_model.term_names
    }
    height_total = torch.zeros(n, dtype=torch.float64, device=env.device)
    original_walk = torch.zeros(n, dtype=torch.float64, device=env.device)
    steps = torch.zeros(n, dtype=torch.float64, device=env.device)
    termination_step = torch.full(
        (n,), float(horizon), dtype=torch.float64, device=env.device
    )
    selection = torch.zeros(n, 8, dtype=torch.int64, device=env.device)
    q_gain_sum = torch.zeros(n, dtype=torch.float64, device=env.device)
    tracking_steps = torch.zeros(n, dtype=torch.float64, device=env.device)
    xy_error_sq = torch.zeros(n, dtype=torch.float64, device=env.device)
    yaw_error_sq = torch.zeros(n, dtype=torch.float64, device=env.device)
    speed_sum = torch.zeros(n, dtype=torch.float64, device=env.device)
    command_speed_sum = torch.zeros(n, dtype=torch.float64, device=env.device)
    timeout_seen = torch.zeros(n, dtype=torch.bool, device=env.device)
    terminal_violation = torch.zeros(n, dtype=torch.bool, device=env.device)
    actions = [] if trace else None
    obs = initial_obs
    first_original = None
    first_height = None
    first_terminated = None
    for step in range(horizon):
        candidates = sample_candidates(policy, obs, act_rng)
        action, index, gain = select_action(obs, candidates, critic)
        selection.scatter_add_(1, index[:, None], alive.long()[:, None])
        q_gain_sum += gain * alive
        if trace:
            actions.append(action.clone())
        # Commands may resample during the step; compare the successor against
        # the command that governed this interval, excluding terminal steps.
        command = env.commands.clone()
        env.set_states(spec.action_names, action)
        env.step()
        height_reward = -(
            (env.base_height[:, 0] - reference_height).abs() > height_bound
        ).double()
        terms = reward_model.step_terms(env)
        if step == 0:
            first_original = reward_model.total(terms).clone()
            first_height = height_reward.clone()
            first_terminated = env.terminated.clone()
        timeout_seen |= env.timed_out
        alive_before = alive
        terminal_now = alive_before & env.terminated
        for name in reward_model.weights:
            terminal_violation |= env.terminated & (terms[name] != 0)
        for name in reward_model.termination_weights:
            terminal_violation |= (~env.terminated) & (terms[name] != 0)
        termination_step = torch.where(
            terminal_now,
            torch.full_like(termination_step, float(step)),
            termination_step,
        )
        discount = reward_model.gamma**step
        for name, value in terms.items():
            term_sums[name] += discount * value.double() * alive_before
        step_total = torch.stack(
            [terms[name].double() for name in reward_model.term_names]
        ).sum(dim=0)
        original_walk += discount * step_total * alive_before
        height_total, alive = accumulate_return(
            height_total,
            alive_before,
            height_reward,
            env.terminated,
            step,
            reward_model.gamma,
        )
        steps += alive_before
        tracking_steps += alive_before
        xy_error_sq += (env.base_lin_vel[:, :2] - command[:, :2]).square().sum(
            -1
        ) * alive_before
        yaw_error_sq += (env.base_lin_vel[:, 2] - command[:, 2]).square() * alive_before
        speed_sum += env.base_lin_vel[:, :2].norm(dim=-1) * alive_before
        command_speed_sum += command[:, :2].norm(dim=-1) * alive_before
        if step != horizon - 1:
            obs = _policy_observation(
                env, spec.obs_names, spec.obs_dims, spec.noise, spec.scales, obs_rng
            )
    original_return = torch.stack(
        [term_sums[name] for name in reward_model.term_names]
    ).sum(dim=0)
    if not torch.allclose(original_walk, original_return, atol=1e-10, rtol=0.0):
        raise RuntimeError(
            "per-term original return does not match stepwise accumulation: "
            f"{(original_walk - original_return).abs().max().item()}"
        )
    if timeout_seen.any():
        raise RuntimeError("timeout observed in a tradeoff rollout")
    if terminal_violation.any():
        raise RuntimeError("terminal reward mask invariant violated")
    values = [original_return, height_total, *term_sums.values()]
    if not all(torch.isfinite(value).all() for value in values):
        raise RuntimeError("nonfinite tradeoff return")
    result = {
        "original_returns": original_return,
        "height_returns": height_total,
        "original_terms": torch.stack(
            [term_sums[name] for name in reward_model.term_names], dim=-1
        ),
        "term_names": list(reward_model.term_names),
        "survived": alive,
        "steps": steps,
        "termination_step": termination_step,
        "selection_counts": selection,
        "q_gain_sum": q_gain_sum,
        "tracking_steps": tracking_steps,
        "xy_error_sq": xy_error_sq,
        "yaw_error_sq": yaw_error_sq,
        "speed_sum": speed_sum,
        "command_speed_sum": command_speed_sum,
        "first_original_reward": first_original,
        "first_height_reward": first_height,
        "first_terminated": first_terminated,
        "timeout_seen": timeout_seen,
    }
    if trace:
        result["actions"] = torch.stack(actions, dim=1)
    return {
        key: value.cpu() if isinstance(value, torch.Tensor) else value
        for key, value in result.items()
    }
