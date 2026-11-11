"""Batched fixed-policy height Q evaluation.

One environment is created per candidate batch with `num_envs = trials`.  A
fresh environment with the same seed reproduces the same trial states, so
candidate batches and repetitions are paired by world index.  This is much
faster than the single-world evaluator, while keeping the same per-trial
comparison semantics.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from gym.utils.helpers import class_to_dict
from gym.utils.task_registry import task_registry
from learning.offline.contracts import (
    HeightRewardSpec,
    replay_config_fingerprint,
    sha256_file,
)
from learning.offline.evaluation import (
    _aggregate,
    _close_env,
    _policy_observation,
    load_policy_from_artifact,
)
from learning.offline.q_artifact import HeightQ


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--experiment_name", default="go2trot")
    parser.add_argument("--load_run", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trials", type=int, default=1024)
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument(
        "--insert_step",
        type=int,
        default=0,
        help=(
            "Frozen-policy steps taken from the reset state before the candidate "
            "actions are inserted. 0 scores the reset state (original protocol); "
            "k > 0 scores a mid-gait state reached by the frozen policy."
        ),
    )
    parser.add_argument("--num_candidates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[-0.5, -1.0, -2.0, -5.0, -10.0],
    )
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.trials <= 0 or args.repetitions <= 0 or args.horizon <= 0:
        raise SystemExit("trials, repetitions, and horizon must be positive")
    if args.num_candidates < 2:
        raise SystemExit("num_candidates must be at least 2")
    if args.insert_step < 0:
        raise SystemExit("insert_step must be non-negative")
    import gym.envs  # noqa: F401

    artifact = HeightQ(args.artifact)
    policy, policy_spec = load_policy_from_artifact(
        artifact, args.checkpoint, args.device
    )
    if sha256_file(args.checkpoint) != policy_spec.checkpoint_sha256:
        raise SystemExit("checkpoint hash does not match the Q artifact")

    reward_data = artifact.manifest["reward_spec"]
    reward_spec = HeightRewardSpec(
        reference_height_m=reward_data["reference_height_m"],
        bound_m=reward_data["bound_m"],
        inside_reward=reward_data["inside_reward"],
        outside_reward=reward_data["outside_reward"],
    )
    gamma = float(artifact.manifest["gamma"])
    failure_value = reward_spec.failure_value(gamma)

    env_cfg, train_cfg = copy.deepcopy(
        task_registry.get_cfgs(
            args.experiment_name,
            original_cfg=False,
            experiment_name=args.experiment_name,
            load_run=args.load_run,
        )
    )
    env_cfg.env.num_envs = args.trials
    train_cfg.runner.device = args.device
    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    recorded_fingerprint = replay_config_fingerprint(class_to_dict(env_cfg))
    artifact_fingerprint = artifact.manifest["dataset"].get("config_fingerprint")
    if (
        artifact_fingerprint is not None
        and recorded_fingerprint != artifact_fingerprint
    ):
        raise SystemExit(
            "environment configuration does not match the Q artifact dataset"
        )
    control_dt = env_cfg.control.ctrl_dt
    # The rollout now covers the pre-insertion prefix plus the measured horizon;
    # the episode must outlast both so an administrative timeout cannot reset a
    # world in the middle of the measurement.
    env_cfg.env.episode_length_s = max(
        env_cfg.env.episode_length_s,
        (args.insert_step + args.horizon) * control_dt + 1.0,
    )

    def env_factory(seed: int):
        cfg = copy.deepcopy(env_cfg)
        cfg.seed = seed
        return task_registry.make_env(
            args.experiment_name,
            cfg,
            args.device,
            True,
            "mujoco",
        )

    # Generate the paired trial states and candidates once.  Every batch below
    # recreates the same environment and therefore the same trial states.
    probe = env_factory(args.seed)
    try:
        initial_generator = torch.Generator(device=probe.device).manual_seed(
            args.seed + 3
        )
        initial_obs = _policy_observation(
            probe,
            policy_spec.obs_names,
            policy_spec.obs_dims,
            policy_spec.noise,
            policy_spec.scales,
            initial_generator,
        )
        mean_action = policy.mean_actions(initial_obs)
        std = policy.actor.std.detach().to(mean_action.device)
        candidate_generator = torch.Generator(device=mean_action.device).manual_seed(
            args.seed + 5
        )
        noise = torch.randn(
            args.num_candidates - 1,
            args.trials,
            mean_action.shape[1],
            generator=candidate_generator,
            device=mean_action.device,
        )
        candidates = torch.cat(
            [mean_action.unsqueeze(0), mean_action.unsqueeze(0) + std * noise],
            dim=0,
        ).cpu()
    finally:
        _close_env(probe)

    predictions = torch.zeros(args.num_candidates, args.trials)
    returns = torch.zeros(args.repetitions, args.num_candidates, args.trials)
    lower = torch.zeros_like(returns)
    upper = torch.zeros_like(returns)
    terminated = torch.zeros(
        args.repetitions, args.num_candidates, args.trials, dtype=torch.bool
    )
    steps = torch.zeros(
        args.repetitions, args.num_candidates, args.trials, dtype=torch.int64
    )
    first_rewards = torch.zeros(args.repetitions, args.num_candidates, args.trials)
    first_terminated = torch.zeros(
        args.repetitions, args.num_candidates, args.trials, dtype=torch.bool
    )
    first_scores = torch.zeros(args.repetitions, args.num_candidates, args.trials)
    second_rewards = torch.zeros(args.repetitions, args.num_candidates, args.trials)
    second_terminated = torch.zeros(
        args.repetitions, args.num_candidates, args.trials, dtype=torch.bool
    )
    two_step_scores = torch.zeros(args.repetitions, args.num_candidates, args.trials)

    reference_obs = initial_obs.detach().cpu()
    # Insertion diagnostics are shared by all candidates of a repetition because
    # the prefix runs with repetition-scoped seeds only.
    reference_insert_obs = [None] * args.repetitions
    insert_heights = torch.zeros(args.repetitions, args.trials)
    prefix_alive = torch.zeros(args.repetitions, args.trials, dtype=torch.bool)
    for repetition in range(args.repetitions):
        continuation_seed = args.seed + 1_000_000 + repetition * 10
        state_snapshot = None
        for candidate_index in range(args.num_candidates):
            env = env_factory(args.seed)
            try:
                check_generator = torch.Generator(device=env.device).manual_seed(
                    args.seed + 3
                )
                batch_initial_obs = _policy_observation(
                    env,
                    policy_spec.obs_names,
                    policy_spec.obs_dims,
                    policy_spec.noise,
                    policy_spec.scales,
                    check_generator,
                )
                if not torch.allclose(
                    batch_initial_obs.detach().cpu(),
                    reference_obs,
                    rtol=1e-4,
                    atol=1e-5,
                ):
                    raise RuntimeError(
                        "batched environment did not reproduce trial states"
                    )

                torch.manual_seed(continuation_seed)
                observation_generator = torch.Generator(device=env.device).manual_seed(
                    continuation_seed + 1
                )
                policy_generator = torch.Generator(device=env.device).manual_seed(
                    continuation_seed + 2
                )
                alive = torch.ones(args.trials, dtype=torch.bool, device=env.device)
                if args.insert_step == 0:
                    # Preserve the original protocol exactly: score the reset
                    # state using the observation already verified above.
                    insert_obs = batch_initial_obs
                else:
                    if state_snapshot is None:
                        # First candidate of this repetition: advance the frozen
                        # policy to a mid-gait state and snapshot it. The other
                        # candidates replay the snapshot instead of re-running
                        # the prefix, because Warp solver noise makes a repeated
                        # prefix land on a measurably different state.
                        for _ in range(args.insert_step):
                            prefix_observation = _policy_observation(
                                env,
                                policy_spec.obs_names,
                                policy_spec.obs_dims,
                                policy_spec.noise,
                                policy_spec.scales,
                                observation_generator,
                            )
                            prefix_actions = policy.sample_actions(
                                prefix_observation, policy_generator
                            )
                            prefix_actions = torch.where(
                                alive.unsqueeze(1),
                                prefix_actions,
                                torch.zeros_like(prefix_actions),
                            )
                            env.set_states(policy_spec.action_names, prefix_actions)
                            env.step()
                            alive = alive & ~env.terminated.bool()
                        state_snapshot = env.capture_state()
                    # Restore for every candidate including the first: the
                    # backend clamps joint positions to the asset limits on
                    # restore, so scoring the raw post-prefix state would put
                    # candidate 0 on a slightly different state than the rest.
                    env.restore_state(state_snapshot)
                    alive = (
                        torch.ones(args.trials, dtype=torch.bool, device=env.device)
                        if candidate_index == 0
                        else prefix_alive[repetition].to(env.device).clone()
                    )
                    insert_generator = torch.Generator(device=env.device).manual_seed(
                        continuation_seed + 5
                    )
                    insert_obs = _policy_observation(
                        env,
                        policy_spec.obs_names,
                        policy_spec.obs_dims,
                        policy_spec.noise,
                        policy_spec.scales,
                        insert_generator,
                    )

                insert_obs_cpu = insert_obs.detach().cpu()
                if reference_insert_obs[repetition] is None:
                    reference_insert_obs[repetition] = insert_obs_cpu
                    insert_heights[repetition] = env.base_height[:, 0].detach().cpu()
                    prefix_alive[repetition] = alive.detach().cpu()
                elif not torch.allclose(
                    insert_obs_cpu,
                    reference_insert_obs[repetition],
                    rtol=1e-4,
                    atol=1e-5,
                ):
                    # The snapshot is restored, not replayed, so the paired
                    # state is exact and only observation-noise arithmetic can
                    # differ here.
                    raise RuntimeError(
                        "insertion state was not reproduced across candidates"
                    )

                predictions[candidate_index] = (
                    artifact.predict(
                        insert_obs,
                        candidates[candidate_index].to(env.device),
                    )
                    .detach()
                    .cpu()
                )

                actions = candidates[candidate_index].to(env.device)
                env.set_states(policy_spec.action_names, actions)
                active = alive.clone()
                total = torch.zeros(args.trials, dtype=torch.float32, device=env.device)
                discount = torch.ones(
                    args.trials, dtype=torch.float32, device=env.device
                )
                done = torch.zeros(args.trials, dtype=torch.bool, device=env.device)
                step_count = torch.zeros(
                    args.trials, dtype=torch.int64, device=env.device
                )
                first_reward_device = None
                first_term_device = None
                first_score_device = None
                for step_index in range(args.horizon):
                    env.step()
                    height = env.base_height[:, 0]
                    reward = torch.where(
                        torch.abs(height - reward_spec.reference_height_m)
                        <= reward_spec.bound_m,
                        torch.full_like(height, reward_spec.inside_reward),
                        torch.full_like(height, reward_spec.outside_reward),
                    )
                    total = total + torch.where(
                        active, discount * reward, torch.zeros_like(total)
                    )
                    discount = discount * gamma
                    step_count = step_count + active.long()
                    terminal_now = env.terminated.bool() & active
                    total = total + torch.where(
                        terminal_now,
                        discount * failure_value,
                        torch.zeros_like(total),
                    )
                    done = done | terminal_now
                    active = active & ~terminal_now
                    first_observation = None
                    if step_index == 0:
                        first_observation = _policy_observation(
                            env,
                            policy_spec.obs_names,
                            policy_spec.obs_dims,
                            policy_spec.noise,
                            policy_spec.scales,
                            observation_generator,
                        )
                        first_rewards[repetition, candidate_index] = (
                            reward.detach().cpu()
                        )
                        first_terminated[repetition, candidate_index] = (
                            terminal_now.detach().cpu()
                        )
                        next_mean = policy.mean_actions(first_observation)
                        next_q = artifact.predict(first_observation, next_mean).to(
                            env.device
                        )
                        lookahead_score = reward + gamma * torch.where(
                            terminal_now,
                            torch.full_like(next_q, failure_value),
                            next_q,
                        )
                        first_scores[repetition, candidate_index] = (
                            lookahead_score.detach().cpu()
                        )
                        first_reward_device = reward.detach()
                        first_term_device = terminal_now.detach()
                        first_score_device = lookahead_score.detach()
                        two_step_scores[repetition, candidate_index] = (
                            first_score_device.detach().cpu()
                        )
                    if not bool(active.any()):
                        break
                    if first_observation is None:
                        observation = _policy_observation(
                            env,
                            policy_spec.obs_names,
                            policy_spec.obs_dims,
                            policy_spec.noise,
                            policy_spec.scales,
                            observation_generator,
                        )
                    else:
                        observation = first_observation
                    if step_index == 1:
                        second_rewards[repetition, candidate_index] = (
                            reward.detach().cpu()
                        )
                        second_terminated[repetition, candidate_index] = (
                            terminal_now.detach().cpu()
                        )
                        second_mean = policy.mean_actions(observation)
                        second_q = artifact.predict(observation, second_mean).to(
                            env.device
                        )
                        two_step_score = (
                            first_reward_device
                            + gamma * reward
                            + gamma**2
                            * torch.where(
                                terminal_now,
                                torch.full_like(second_q, failure_value),
                                second_q,
                            )
                        )
                        two_step_score = torch.where(
                            first_term_device,
                            first_score_device,
                            two_step_score,
                        )
                        two_step_scores[repetition, candidate_index] = (
                            two_step_score.detach().cpu()
                        )
                    next_actions = policy.sample_actions(observation, policy_generator)
                    next_actions = torch.where(
                        active.unsqueeze(1),
                        next_actions,
                        torch.zeros_like(next_actions),
                    )
                    env.set_states(policy_spec.action_names, next_actions)

                tail_min = discount * reward_spec.min_reward / (1.0 - gamma)
                tail_max = discount * reward_spec.max_reward / (1.0 - gamma)
                lower[repetition, candidate_index] = (
                    torch.where(active, total + tail_min, total).detach().cpu()
                )
                upper[repetition, candidate_index] = (
                    torch.where(active, total + tail_max, total).detach().cpu()
                )
                returns[repetition, candidate_index] = total.detach().cpu()
                terminated[repetition, candidate_index] = done.detach().cpu()
                steps[repetition, candidate_index] = step_count.detach().cpu()
            finally:
                _close_env(env)

    results = []
    for repetition in range(args.repetitions):
        for candidate_index in range(args.num_candidates):
            for trial in range(args.trials):
                if not bool(prefix_alive[repetition, trial]):
                    # The world terminated before the candidate could be applied,
                    # so it has no insertion state to score.
                    continue
                results.append(
                    {
                        "trial": trial,
                        "repetition": repetition,
                        "candidate": candidate_index,
                        "prediction": float(predictions[candidate_index, trial]),
                        "observed_return": float(
                            returns[repetition, candidate_index, trial]
                        ),
                        "return_lower": float(
                            lower[repetition, candidate_index, trial]
                        ),
                        "return_upper": float(
                            upper[repetition, candidate_index, trial]
                        ),
                        "terminated": bool(
                            terminated[repetition, candidate_index, trial]
                        ),
                        "steps": int(steps[repetition, candidate_index, trial]),
                    }
                )
    aggregate = _aggregate(results, tuple(args.thresholds))
    lookahead_scores = first_scores.mean(dim=0)
    lookahead_results = []
    for repetition in range(args.repetitions):
        for candidate_index in range(args.num_candidates):
            for trial in range(args.trials):
                if not bool(prefix_alive[repetition, trial]):
                    continue
                lookahead_results.append(
                    {
                        "trial": trial,
                        "repetition": repetition,
                        "candidate": candidate_index,
                        "prediction": float(lookahead_scores[candidate_index, trial]),
                        "observed_return": float(
                            returns[repetition, candidate_index, trial]
                        ),
                        "return_lower": float(
                            lower[repetition, candidate_index, trial]
                        ),
                        "return_upper": float(
                            upper[repetition, candidate_index, trial]
                        ),
                        "terminated": bool(
                            terminated[repetition, candidate_index, trial]
                        ),
                        "steps": int(steps[repetition, candidate_index, trial]),
                    }
                )
    lookahead_aggregate = _aggregate(lookahead_results, tuple(args.thresholds))
    two_step_scores_mean = two_step_scores.mean(dim=0)
    two_step_results = []
    for repetition in range(args.repetitions):
        for candidate_index in range(args.num_candidates):
            for trial in range(args.trials):
                if not bool(prefix_alive[repetition, trial]):
                    continue
                two_step_results.append(
                    {
                        "trial": trial,
                        "repetition": repetition,
                        "candidate": candidate_index,
                        "prediction": float(
                            two_step_scores_mean[candidate_index, trial]
                        ),
                        "observed_return": float(
                            returns[repetition, candidate_index, trial]
                        ),
                        "return_lower": float(
                            lower[repetition, candidate_index, trial]
                        ),
                        "return_upper": float(
                            upper[repetition, candidate_index, trial]
                        ),
                        "terminated": bool(
                            terminated[repetition, candidate_index, trial]
                        ),
                        "steps": int(steps[repetition, candidate_index, trial]),
                    }
                )
    two_step_aggregate = _aggregate(two_step_results, tuple(args.thresholds))
    excluded_per_repetition = (~prefix_alive).sum(dim=1).tolist()
    in_band = (insert_heights - reward_spec.reference_height_m).abs() <= (
        reward_spec.bound_m
    )
    report = {
        "artifact": artifact.manifest,
        "policy_spec": policy_spec.__dict__,
        "trials": args.trials,
        "repetitions": args.repetitions,
        "horizon": args.horizon,
        "insert_step": args.insert_step,
        "num_candidates": args.num_candidates,
        "seed": args.seed,
        "predictions": predictions.tolist(),
        "mean_returns": returns.mean(dim=0).tolist(),
        "first_rewards": first_rewards.mean(dim=0).tolist(),
        "first_terminated": first_terminated.float().mean(dim=0).tolist(),
        "insertion": {
            "step": args.insert_step,
            "heights": insert_heights.tolist(),
            "height_mean": float(insert_heights.mean()),
            "inside_band_fraction": float(in_band.float().mean()),
            "excluded_trials_per_repetition": excluded_per_repetition,
            "scored_cases": len(results),
        },
        "lookahead": {
            "action": "frozen_policy_mean_at_first_successor",
            "scores": lookahead_scores.tolist(),
            "aggregate": lookahead_aggregate,
        },
        "two_step": {
            "action": "frozen_policy_mean_at_second_successor",
            "scores": two_step_scores_mean.tolist(),
            "aggregate": two_step_aggregate,
        },
        "aggregate": aggregate,
        "results": results,
        "environment": {
            "experiment_name": args.experiment_name,
            "load_run": args.load_run,
            "device": args.device,
            "env_config": class_to_dict(env_cfg),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "trials": args.trials,
                "insert_step": args.insert_step,
                "inside_band_fraction": float(in_band.float().mean()),
                "excluded_trials": int((~prefix_alive).sum()),
                "ranking_pair_accuracy": aggregate["ranking_pair_accuracy"],
                "mae": sum(row["mae"] for row in aggregate["candidates"])
                / len(aggregate["candidates"]),
                "lookahead_ranking_pair_accuracy": lookahead_aggregate[
                    "ranking_pair_accuracy"
                ],
                "lookahead_mae": sum(
                    row["mae"] for row in lookahead_aggregate["candidates"]
                )
                / len(lookahead_aggregate["candidates"]),
                "two_step_ranking_pair_accuracy": two_step_aggregate[
                    "ranking_pair_accuracy"
                ],
                "two_step_mae": sum(
                    row["mae"] for row in two_step_aggregate["candidates"]
                )
                / len(two_step_aggregate["candidates"]),
            }
        )
    )


if __name__ == "__main__":
    main()
