"""Rank the eight stored insertion candidates by Q and by realized yaw return.

Fixed-state diagnostic: each stored candidate action is applied as the first
action from its insertion state, then the frozen policy controls H steps. All
candidate copies of a state share commands/pushes (paired environment noise), so
realized differences are attributable to the candidate (plus Warp trajectory
divergence, which is reported separately). Returns use the task's yaw tracking
term with the zero-after-terminal convention.
"""

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import torch

from gym.utils.helpers import class_to_dict
from gym.utils.task_registry import task_registry
from learning.offline.contracts import (
    AngularVelocityRewardSpec,
    replay_config_fingerprint,
    sha256_file,
)
from learning.offline.evaluation import _policy_observation, load_policy_from_artifact
from learning.offline.fixed_state_variance import (
    WarpInsertionSnapshot,
    accumulate_return_zero_after_terminal,
    install_paired_environment_noise,
)
from learning.offline.q_artifact import AngularVelocityQ
from learning.offline.rewards import angular_velocity_reward


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--load_run", required=True)
    p.add_argument("--artifacts", nargs="+", required=True)
    p.add_argument("--artifact_names", nargs="+", default=None)
    p.add_argument("--horizons", nargs="+", type=int, default=[100, 500])
    p.add_argument("--seeds", nargs="+", type=int, default=[7400000, 7400100])
    return p


def rank(values: torch.Tensor) -> torch.Tensor:
    """Average ranks (1 = lowest) with ties shared."""
    count = values.numel()
    order = values.argsort()
    ranks = torch.empty(count, dtype=torch.float64)
    ranks[order] = torch.arange(1, count + 1, dtype=torch.float64)
    sorted_values = values[order]
    start = 0
    for index in range(1, count + 1):
        if index == count or not torch.equal(
            sorted_values[index], sorted_values[start]
        ):
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return ranks


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    rank_x = rank(x)
    rank_y = rank(y)
    if float(rank_x.std()) == 0.0 or float(rank_y.std()) == 0.0:
        return 0.0
    centered_x = rank_x - rank_x.mean()
    centered_y = rank_y - rank_y.mean()
    return float(
        (centered_x * centered_y).sum()
        / (centered_x.square().sum() * centered_y.square().sum()).sqrt()
    )


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.double().flatten()
    y = y.double().flatten()
    if float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return 0.0
    return float(
        ((x - x.mean()) * (y - y.mean())).mean()
        / (x.std(unbiased=False) * y.std(unbiased=False))
    )


def bootstrap_mean(values: torch.Tensor, seed: int, draws: int = 10_000):
    generator = torch.Generator().manual_seed(seed)
    index = torch.randint(len(values), (draws, len(values)), generator=generator)
    means = values.double()[index].mean(-1)
    return [float(means.quantile(0.025)), float(means.quantile(0.975))]


@torch.no_grad()
def rollout_candidates(
    env, policy, policy_spec, snapshot, candidates, reward_spec, seed, horizon
):
    """Candidate first action, then the frozen policy; returns [state, candidate]."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    snapshot.restore()
    snapshot.assert_restored()
    torch.manual_seed(seed)
    obs_rng = torch.Generator(device=env.device).manual_seed(seed + 1)
    act_rng = torch.Generator(device=env.device).manual_seed(seed + 2)
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    total = torch.zeros(env.num_envs, dtype=torch.float64, device=env.device)
    timeout = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env.set_states(policy_spec.action_names, candidates.flatten(0, 1))
    for step in range(horizon):
        env.step()
        reward = angular_velocity_reward(
            env.commands[:, 2], env.base_ang_vel[:, 2], reward_spec
        ) * (~env.terminated)
        timeout |= env.timed_out
        total, alive = accumulate_return_zero_after_terminal(
            total, alive, reward, env.terminated, step, 0.99
        )
        if step != horizon - 1:
            observation = _policy_observation(
                env,
                policy_spec.obs_names,
                policy_spec.obs_dims,
                policy_spec.noise,
                policy_spec.scales,
                obs_rng,
            )
            env.set_states(
                policy_spec.action_names,
                policy.sample_actions(observation, act_rng),
            )
    if timeout.any():
        raise RuntimeError("timeout observed in the candidate rollout")
    return (
        total.reshape(-1, candidates.shape[1]).cpu(),
        alive.reshape(-1, candidates.shape[1]).cpu(),
    )


@torch.no_grad()
def rank_metrics(scores: torch.Tensor, returns: torch.Tensor) -> dict:
    """Per-state ranking quality of Q against the realized finite return."""
    states = returns.shape[0]
    spearmans = []
    hits = 0
    regrets = []
    normalized_regrets = []
    realized_ranges = []
    q_advantages = scores[:, 1:] - scores[:, :1]
    realized_advantages = returns[:, 1:] - returns[:, :1]
    for state in range(states):
        q_row = scores[state]
        r_row = returns[state]
        spearmans.append(spearman(q_row, r_row))
        best = int(r_row.argmax())
        chosen = int(q_row.argmax())
        hits += int(best == chosen)
        regret = float(r_row[best] - r_row[chosen])
        spread = float(r_row.max() - r_row.min())
        regrets.append(regret)
        realized_ranges.append(spread)
        normalized_regrets.append(regret / spread if spread > 0 else 0.0)
    spearmans_tensor = torch.tensor(spearmans, dtype=torch.float64)
    regrets_tensor = torch.tensor(regrets, dtype=torch.float64)
    return {
        "states": states,
        "spearman_mean": float(spearmans_tensor.mean()),
        "spearman_median": float(spearmans_tensor.median()),
        "spearman_positive_fraction": float((spearmans_tensor > 0).double().mean()),
        "spearman_state_ci": bootstrap_mean(spearmans_tensor, 202611131),
        "argmax_hit_rate": hits / states,
        "top1_regret_mean": float(regrets_tensor.mean()),
        "top1_regret_median": float(regrets_tensor.median()),
        "top1_regret_normalized_mean": float(
            torch.tensor(normalized_regrets, dtype=torch.float64).mean()
        ),
        "realized_within_state_range_mean": float(
            torch.tensor(realized_ranges, dtype=torch.float64).mean()
        ),
        "q_advantage_mean": float(q_advantages.mean()),
        "q_max_advantage_mean": float((scores.max(-1).values - scores[:, 0]).mean()),
        "realized_advantage_mean": float(realized_advantages.mean()),
        "realized_max_advantage_mean": float(
            (returns.max(-1).values - returns[:, 0]).mean()
        ),
        "advantage_pearson_pairs": pearson(q_advantages, realized_advantages),
        "advantage_pearson_state_means": pearson(
            q_advantages.mean(-1), realized_advantages.mean(-1)
        ),
        "candidate0_q_share": float((scores.argmax(-1) == 0).double().mean()),
    }


@torch.no_grad()
def main():
    args = build_parser().parse_args()
    if args.artifact_names is None:
        args.artifact_names = [Path(path).name for path in args.artifacts]
    if len(args.artifact_names) != len(args.artifacts):
        raise ValueError("--artifact_names must match --artifacts")
    if len(set(args.artifact_names)) != len(args.artifact_names):
        raise ValueError("--artifact_names must be unique")
    if not args.horizons or any(horizon <= 0 for horizon in args.horizons):
        raise ValueError("horizons must be positive")
    import gym.envs  # noqa: F401

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    source = Path(args.source) / "insertion.pt"
    saved = torch.load(source, weights_only=True)
    artifacts = [AngularVelocityQ(path) for path in args.artifacts]
    policy, policy_spec = load_policy_from_artifact(
        artifacts[0], args.checkpoint, "cuda:0"
    )
    for q in artifacts:
        if q.manifest["policy_spec"] != artifacts[0].manifest["policy_spec"]:
            raise ValueError("policy mismatch")
        if q.manifest["reward_spec"]["tracking_sigma"] != 0.25:
            raise ValueError("reward mismatch")
    if (
        sha256_file(args.checkpoint) != policy_spec.checkpoint_sha256
        or saved["checkpoint_sha256"] != policy_spec.checkpoint_sha256
        or policy_spec.n_policy_steps != 1
    ):
        raise ValueError("checkpoint/interval mismatch")

    cfg, train = copy.deepcopy(
        task_registry.get_cfgs(
            "go2trot",
            original_cfg=False,
            experiment_name="go2trot",
            load_run=args.load_run,
        )
    )
    cfg.env.num_envs = 1024
    train.runner.device = "cuda:0"
    task_registry.convert_frequencies_to_params(cfg, train)
    if (
        replay_config_fingerprint(class_to_dict(cfg))
        != artifacts[0].manifest["dataset"]["config_fingerprint"]
    ):
        raise ValueError("environment fingerprint mismatch")
    cfg.env.episode_length_s = max(
        cfg.env.episode_length_s, 601 * cfg.control.ctrl_dt + 1
    )
    cfg.seed = saved["seed"]
    env = task_registry.make_env("go2trot", cfg, "cuda:0", True, "mujoco")
    try:
        # State-major, candidate-minor destination order: all eight candidate
        # worlds of one state share that state's snapshot row. The snapshot
        # inventory has 128 states x 8 legacy lanes, so use all 128 stored
        # states and read lane 0 of each block.
        states = 128
        ids = torch.arange(states).repeat_interleave(8) * 8
        snapshot = WarpInsertionSnapshot(env)
        snapshot.load_export(saved["snapshot"], ids)
        snapshot.restore()
        snapshot.assert_restored()
        install_paired_environment_noise(env, 8)
        observations = saved["insertion_observations"][ids].to(env.device)
        candidates = saved["candidates"][:states].to(env.device)
        reward_spec = AngularVelocityRewardSpec(**artifacts[0].manifest["reward_spec"])
        actions = candidates.flatten(0, 1)
        scores = {
            name: q.predict(observations, actions).cpu().reshape(-1, 8)
            for name, q in zip(args.artifact_names, artifacts, strict=True)
        }
        report = {
            "protocol": {
                "source": str(source),
                "source_sha256": sha256_file(source),
                "checkpoint_sha256": policy_spec.checkpoint_sha256,
                "artifacts": {
                    path: {
                        field: sha256_file(Path(path) / field)
                        for field in ("manifest.json", "q.pt")
                    }
                    for path in args.artifacts
                },
                "artifact_names": args.artifact_names,
                "states": states,
                "candidates_per_state": 8,
                "horizons": args.horizons,
                "seeds": args.seeds,
                "order": "state-major, candidate-minor; candidate is the first action",
                "commands": (
                    "paired within each state (install_paired_environment_noise)"
                ),
                "reward": (
                    "post-step exp(-((cmd-ang)/2.5)^2/0.25), zero after termination"
                ),
                "revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
            },
            "q_scores": {
                name: {
                    "per_candidate_mean": scores[name].mean(0).tolist(),
                    "mean": float(scores[name].mean()),
                    "std": float(scores[name].std()),
                }
                for name in args.artifact_names
            },
            "realized": {},
            "critics": {name: {} for name in args.artifact_names},
        }
        for horizon in args.horizons:
            per_seed = []
            alive_rows = []
            for seed in args.seeds:
                returns, alive = rollout_candidates(
                    env,
                    policy,
                    policy_spec,
                    snapshot,
                    candidates,
                    reward_spec,
                    seed,
                    horizon,
                )
                per_seed.append(returns)
                alive_rows.append(alive)
                for name in args.artifact_names:
                    metrics = rank_metrics(scores[name], returns)
                    report["critics"][name].setdefault(f"h{horizon}", []).append(
                        {"seed": seed, **metrics}
                    )
            stacked = torch.stack(per_seed)
            report["realized"][f"h{horizon}"] = {
                "mean_return": float(stacked.mean()),
                "state_candidate_sd": float(stacked.std()),
                "within_state_range_mean": float(
                    (stacked.max(-1).values - stacked.min(-1).values).mean()
                ),
                "within_state_sd_mean": float(stacked.std(-1).mean()),
                "survivor_fraction": float(torch.stack(alive_rows).double().mean()),
                "per_seed_state_correlation": (
                    None
                    if len(per_seed) < 2
                    else float(pearson(stacked[0].flatten(), stacked[1].flatten()))
                ),
                "cross_seed_ranking": (
                    None
                    if len(per_seed) < 2
                    else {
                        "within_state_spearman_mean": float(
                            torch.tensor(
                                [
                                    spearman(stacked[0][state], stacked[1][state])
                                    for state in range(stacked.shape[1])
                                ],
                                dtype=torch.float64,
                            ).mean()
                        ),
                        "within_state_argmax_agreement": float(
                            (stacked[0].argmax(-1) == stacked[1].argmax(-1))
                            .double()
                            .mean()
                        ),
                        "note": (
                            "test-retest reliability of the realized ranking; this "
                            "is the ceiling a Q ranker could reach"
                        ),
                    }
                ),
            }
            for name in args.artifact_names:
                rows = report["critics"][name][f"h{horizon}"]
                report["critics"][name][f"h{horizon}"] = {
                    "per_seed": rows,
                    "mean": {
                        key: float(torch.tensor([row[key] for row in rows]).mean())
                        for key in (
                            "spearman_mean",
                            "spearman_median",
                            "spearman_positive_fraction",
                            "argmax_hit_rate",
                            "top1_regret_mean",
                            "top1_regret_normalized_mean",
                            "q_advantage_mean",
                            "q_max_advantage_mean",
                            "realized_advantage_mean",
                            "realized_max_advantage_mean",
                            "advantage_pearson_pairs",
                            "advantage_pearson_state_means",
                            "candidate0_q_share",
                        )
                    },
                }
        (output / "ranking.json").write_text(json.dumps(report, indent=2))
        with (output / "commands.txt").open("a") as handle:
            handle.write("uv run --frozen python " + " ".join(sys.argv) + "\n")
        print(json.dumps(report["realized"], indent=2))
        for name in args.artifact_names:
            for horizon in args.horizons:
                summary = report["critics"][name][f"h{horizon}"]["mean"]
                print(
                    name,
                    f"h{horizon}",
                    json.dumps({k: round(v, 4) for k, v in summary.items()}),
                )
    finally:
        env._backend.close()


if __name__ == "__main__":
    main()
