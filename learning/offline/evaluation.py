"""Independent simulator validation for fixed-policy height Q artifacts."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Callable

import torch

from learning.offline.contracts import (
    EvaluationSpec,
    HeightRewardSpec,
    PolicySpec,
    sample_observation_noise,
)
from learning.offline.nominal_policy import FrozenTrotPolicy
from learning.offline.q_artifact import HeightQ


def policy_spec_from_artifact(artifact: HeightQ) -> PolicySpec:
    spec = artifact.manifest["policy_spec"]
    return PolicySpec(
        checkpoint=spec["checkpoint"],
        checkpoint_sha256=spec["checkpoint_sha256"],
        actor_cfg=spec["actor_cfg"],
        obs_names=tuple(spec["obs_names"]),
        obs_dims=tuple(int(dim) for dim in spec["obs_dims"]),
        action_names=tuple(spec["action_names"]),
        action_dim=int(spec["action_dim"]),
        noise=spec["noise"],
        scales=spec["scales"],
        control_dt=float(spec["control_dt"]),
        n_policy_steps=int(spec["n_policy_steps"]),
        episode_length_s=float(spec["episode_length_s"]),
        smooth_exploration=bool(spec["actor_cfg"]["smooth_exploration"]),
    )


def load_policy_from_artifact(
    artifact: HeightQ,
    checkpoint: str,
    device: str,
) -> tuple[FrozenTrotPolicy, PolicySpec]:
    spec = policy_spec_from_artifact(artifact)
    policy = FrozenTrotPolicy.from_checkpoint(
        checkpoint,
        spec.actor_cfg,
        spec.obs_names,
        spec.obs_dims,
        spec.action_names,
        spec.action_dim,
        device,
    )
    return policy, spec


def _policy_observation(
    env,
    obs_names: tuple[str, ...],
    obs_dims: tuple[int, ...],
    noise: dict,
    scales: dict,
    generator: torch.Generator,
) -> torch.Tensor:
    clean = env.get_states(obs_names)
    noise_vec = sample_observation_noise(
        obs_names,
        obs_dims,
        noise,
        scales,
        env.num_envs,
        env.device,
        generator,
    )
    return clean + noise_vec


def _height_reward(height: float, spec: HeightRewardSpec) -> float:
    if abs(height - spec.reference_height_m) <= spec.bound_m:
        return spec.inside_reward
    return spec.outside_reward


def simulate_candidate(
    env,
    policy: FrozenTrotPolicy,
    policy_spec: PolicySpec,
    candidate: torch.Tensor,
    *,
    reward_spec: HeightRewardSpec,
    gamma: float,
    failure_value: float,
    horizon: int,
    continuation_seed: int,
) -> dict:
    """Run one candidate first action, then the frozen nominal policy."""
    if policy_spec.n_policy_steps != 1:
        raise ValueError("evaluation requires n_policy_steps == 1")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if env.num_envs != 1:
        raise ValueError("evaluation requires a single-environment rollout")
    device = env.device
    torch.manual_seed(continuation_seed)
    observation_generator = torch.Generator(device=device).manual_seed(
        continuation_seed + 1
    )
    policy_generator = torch.Generator(device=device).manual_seed(continuation_seed + 2)

    discount = 1.0
    total = 0.0
    terminated = False
    steps = 0
    env.set_states(
        policy_spec.action_names,
        candidate.to(env.device).reshape(1, -1),
    )
    for _ in range(horizon):
        env.step()
        height = float(env.base_height[0, 0])
        total += discount * _height_reward(height, reward_spec)
        discount *= gamma
        steps += 1
        if bool(env.terminated[0]):
            terminated = True
            total += discount * failure_value
            break
        observation = _policy_observation(
            env,
            policy_spec.obs_names,
            policy_spec.obs_dims,
            policy_spec.noise,
            policy_spec.scales,
            observation_generator,
        )
        action = policy.sample_actions(observation, policy_generator)
        env.set_states(policy_spec.action_names, action)

    if terminated:
        lower = total
        upper = total
    else:
        tail_min = discount * reward_spec.min_reward / (1.0 - gamma)
        tail_max = discount * reward_spec.max_reward / (1.0 - gamma)
        lower = total + tail_min
        upper = total + tail_max
    return {
        "observed_return": total,
        "return_lower": lower,
        "return_upper": upper,
        "terminated": terminated,
        "steps": steps,
    }


def _close_env(env) -> None:
    if hasattr(env, "_backend"):
        env._backend.close()


def _aggregate(results: list[dict], thresholds: tuple[float, ...]) -> dict:
    grouped: dict[tuple[int, int], list[dict]] = {}
    for result in results:
        grouped.setdefault((result["trial"], result["candidate"]), []).append(result)
    summary = []
    for (trial, candidate), rows in sorted(grouped.items()):
        returns = torch.tensor(
            [row["observed_return"] for row in rows],
            dtype=torch.float64,
        )
        lower = torch.tensor(
            [row["return_lower"] for row in rows],
            dtype=torch.float64,
        )
        upper = torch.tensor(
            [row["return_upper"] for row in rows],
            dtype=torch.float64,
        )
        prediction = float(rows[0]["prediction"])
        mean = float(returns.mean())
        sampling = 0.0
        if len(rows) > 1:
            sampling = float(1.96 * returns.std(unbiased=True) / math.sqrt(len(rows)))
        summary.append(
            {
                "trial": trial,
                "candidate": candidate,
                "prediction": prediction,
                "mean_return": mean,
                "std_return": float(returns.std(unbiased=False)),
                "return_lower": float(lower.mean()) - sampling,
                "return_upper": float(upper.mean()) + sampling,
                "mae": abs(prediction - mean),
                "repetitions": len(rows),
            }
        )

    thresholds_report = []
    for threshold in thresholds:
        false_accept = 0
        false_reject = 0
        ambiguous = 0
        for row in summary:
            prediction = row["prediction"]
            lower = row["return_lower"]
            upper = row["return_upper"]
            if lower <= threshold <= upper:
                ambiguous += 1
            if prediction >= threshold and upper < threshold:
                false_accept += 1
            if prediction < threshold and lower > threshold:
                false_reject += 1
        thresholds_report.append(
            {
                "threshold": threshold,
                "false_accept": false_accept,
                "false_reject": false_reject,
                "ambiguous": ambiguous,
                "cases": len(summary),
            }
        )

    trial_groups: dict[int, list[dict]] = {}
    for row in summary:
        trial_groups.setdefault(row["trial"], []).append(row)
    ranking_matches = 0
    ranking_pairs = 0
    for rows in trial_groups.values():
        for left in range(len(rows)):
            for right in range(left + 1, len(rows)):
                prediction_order = rows[left]["prediction"] >= rows[right]["prediction"]
                return_order = rows[left]["mean_return"] >= rows[right]["mean_return"]
                ranking_matches += int(prediction_order == return_order)
                ranking_pairs += 1
    return {
        "candidates": summary,
        "thresholds": thresholds_report,
        "ranking_pair_accuracy": (
            ranking_matches / ranking_pairs if ranking_pairs else None
        ),
        "ranking_pairs": ranking_pairs,
    }


def evaluate_height_q(
    *,
    artifact: HeightQ,
    policy: FrozenTrotPolicy,
    policy_spec: PolicySpec,
    env_factory: Callable[[int], object],
    candidates: torch.Tensor,
    spec: EvaluationSpec,
    reward_spec: HeightRewardSpec,
) -> dict:
    if candidates.ndim != 2:
        raise ValueError("candidates must have shape [num_candidates, action_dim]")
    if candidates.shape[1] != policy_spec.action_dim:
        raise ValueError("candidate action width does not match the policy")
    if not torch.isfinite(candidates).all():
        raise ValueError("candidate actions must be finite")
    gamma = float(artifact.manifest["gamma"])
    failure_value = reward_spec.failure_value(gamma)
    results: list[dict] = []
    for trial in range(spec.trials):
        trial_seed = spec.seed + trial
        reference_obs = None
        for repetition in range(spec.repetitions):
            continuation_seed = spec.seed + 1_000_000 + trial * 10_000 + repetition
            for candidate_index, candidate in enumerate(candidates):
                env = env_factory(trial_seed)
                if torch.device(env.device) != policy.device:
                    _close_env(env)
                    raise ValueError("environment and policy devices differ")
                try:
                    initial_height = float(env.base_height[0, 0])
                    observation_generator = torch.Generator(
                        device=env.device
                    ).manual_seed(trial_seed + 3)
                    initial_obs = _policy_observation(
                        env,
                        policy_spec.obs_names,
                        policy_spec.obs_dims,
                        policy_spec.noise,
                        policy_spec.scales,
                        observation_generator,
                    )
                    if reference_obs is None:
                        reference_obs = initial_obs.clone()
                    elif not torch.allclose(initial_obs, reference_obs):
                        raise RuntimeError(
                            "reproducible initialization produced different "
                            "initial observations"
                        )
                    prediction = float(artifact.predict(initial_obs, candidate)[0])
                    rollout = simulate_candidate(
                        env,
                        policy,
                        policy_spec,
                        candidate,
                        reward_spec=reward_spec,
                        gamma=gamma,
                        failure_value=failure_value,
                        horizon=spec.horizon,
                        continuation_seed=continuation_seed,
                    )
                finally:
                    _close_env(env)
                results.append(
                    {
                        "trial": trial,
                        "repetition": repetition,
                        "candidate": candidate_index,
                        "prediction": prediction,
                        "initial_height": initial_height,
                        **rollout,
                    }
                )
    aggregate = _aggregate(results, spec.thresholds)
    return {
        "artifact": artifact.manifest,
        "policy_spec": asdict(policy_spec),
        "evaluation_spec": asdict(spec),
        "candidates": candidates.detach().cpu().tolist(),
        "results": results,
        "aggregate": aggregate,
    }
