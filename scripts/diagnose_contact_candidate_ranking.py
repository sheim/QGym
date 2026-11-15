"""Within-state spread and candidate ranking for the contact-force Q.

Fixed-state diagnostic. For 128 insertion states, eight candidate first actions
(policy mean plus one standard deviation of the actor's own noise, candidate 0 =
the mean) are applied, then the frozen policy controls the next H steps. All
eight candidates of one state share commands and pushes, so realized
differences are attributable to the candidate.

The realized return is the contact-force label
``-1 if any foot above its threshold else 0`` accumulated with
``discount_episodic_zero_after_terminal`` at the evaluation gamma.

Reports, per horizon and seed pair:

* the realized within-state spread (how much the first action can change the
  contact return at all — if this is near zero, no ranker can help);
* Spearman between Q and the realized return across candidates, argmax hit
  rate, top-1 regret, and the advantage correlation;
* the same-controller seed-to-seed rank correlation, which is the ceiling a
  ranker can reach under Warp replay noise.

Usage:

    uv run --frozen python scripts/diagnose_contact_candidate_ranking.py \
      --output logs/go2trot/<run>/contact_ranking \
      --checkpoint logs/go2trot/<run>/model_616.pt \
      --load_run <run> \
      --artifacts logs/go2trot/<run>/offline_q_60k/contact_q/data/100pct
"""

import argparse
import copy
import json
import subprocess
from pathlib import Path

import torch

from evaluate_height_q_variance import fidelity, physical
from gym.utils.helpers import class_to_dict
from gym.utils.task_registry import task_registry
from learning.offline.contracts import (
    ContactForceRewardSpec,
    replay_config_fingerprint,
    sha256_file,
)
from learning.offline.evaluation import _policy_observation, load_policy_from_artifact
from learning.offline.fixed_state_variance import (
    WarpInsertionSnapshot,
    accumulate_return_zero_after_terminal,
    expand_states,
    install_paired_environment_noise,
)
from learning.offline.q_artifact import ContactQ
from learning.offline.rewards import contact_force_reward

STATES = 128
CANDIDATES = 8
PREFIX_STEPS = 100
EVAL_GAMMA = 0.99


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--load_run", required=True)
    parser.add_argument("--artifacts", nargs="+", required=True)
    parser.add_argument("--artifact_names", nargs="+", default=None)
    parser.add_argument("--horizons", nargs="+", type=int, default=[100, 500])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7600000, 7600100])
    parser.add_argument("--export_seed", type=int, default=7600001)
    parser.add_argument("--device", default="cuda:0")
    return parser


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


def foot_indices(env, spec: ContactForceRewardSpec) -> torch.Tensor:
    names = list(env.robot_layout.body_names)
    missing = [name for name in spec.foot_names if name not in names]
    if missing:
        raise ValueError(f"feet {missing} missing from the robot layout")
    return torch.tensor(
        [names.index(name) for name in spec.foot_names],
        dtype=torch.long,
        device=env.device,
    )


@torch.no_grad()
def export_insertion(env, policy, spec, feet, reward_spec, seed):
    """Freeze 128 alive states, replicate each eight times, build candidates."""
    prefix_rng = torch.Generator(device=env.device).manual_seed(seed + 3)
    action_rng = torch.Generator(device=env.device).manual_seed(seed + 4)
    alive = ~env.terminated.clone()
    for _ in range(PREFIX_STEPS):
        obs = _policy_observation(
            env, spec.obs_names, spec.obs_dims, spec.noise, spec.scales, prefix_rng
        )
        env.set_states(spec.action_names, policy.sample_actions(obs, action_rng))
        env.step()
        alive &= ~env.terminated
    selected = alive.nonzero().flatten()[:STATES]
    if selected.numel() != STATES:
        raise RuntimeError("fewer than 128 surviving insertion states")
    obs = _policy_observation(
        env, spec.obs_names, spec.obs_dims, spec.noise, spec.scales, prefix_rng
    )
    mean = policy.mean_actions(obs)
    native = WarpInsertionSnapshot(env)
    validation = fidelity(env, mean, native)
    snapshot = WarpInsertionSnapshot(env, expand_states(selected, CANDIDATES))
    snapshot.restore()
    snapshot.assert_restored()
    insertion_obs = expand_states(obs[selected], CANDIDATES)
    generator = torch.Generator(device=env.device).manual_seed(seed + 5)
    eps = torch.randn(
        STATES,
        CANDIDATES,
        spec.action_dim,
        device=env.device,
        generator=generator,
    )
    eps[:, 0] = 0.0
    candidates = mean[selected][:, None] + policy.actor.std.detach() * eps

    # Replication fidelity against the uninterrupted source step.
    native.restore()
    torch.manual_seed(654321)
    env.set_states(spec.action_names, mean)
    env.step()
    reference = physical(env)
    snapshot.restore()
    torch.manual_seed(654321)
    env.set_states(spec.action_names, expand_states(mean[selected], CANDIDATES))
    env.step()
    errors = {
        key: float(
            (physical(env)[key] - expand_states(value[selected], CANDIDATES))
            .abs()
            .max()
        )
        for key, value in reference.items()
    }
    validation["replication_max_absolute_error"] = errors
    if any(errors[key] > validation["limits"][key] for key in errors):
        raise RuntimeError(f"replication fidelity gate failed: {errors}")
    snapshot.restore()
    return {
        "snapshot": snapshot.export(),
        "insertion_observations": insertion_obs.cpu(),
        "candidates": candidates.cpu(),
        "selected_world_ids": selected.cpu(),
        "prefix_alive": alive.cpu(),
        "seed": seed,
        "validation": validation,
    }


@torch.no_grad()
def rollout_candidates(
    env, policy, spec, snapshot, candidates, feet, reward_spec, seed, horizon
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
    violations = torch.zeros(env.num_envs, dtype=torch.float64, device=env.device)
    steps = torch.zeros(env.num_envs, dtype=torch.float64, device=env.device)
    timeout = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env.set_states(spec.action_names, candidates.flatten(0, 1))
    for step in range(horizon):
        env.step()
        load = env.contact_forces.index_select(1, feet)[..., 2] / env.bodyweight
        reward = contact_force_reward(load, reward_spec) * (~env.terminated)
        timeout |= env.timed_out
        violations += (reward < 0).double() * alive
        steps += alive
        total, alive = accumulate_return_zero_after_terminal(
            total, alive, reward, env.terminated, step, EVAL_GAMMA
        )
        if step != horizon - 1:
            observation = _policy_observation(
                env,
                spec.obs_names,
                spec.obs_dims,
                spec.noise,
                spec.scales,
                obs_rng,
            )
            env.set_states(
                spec.action_names, policy.sample_actions(observation, act_rng)
            )
    if timeout.any():
        raise RuntimeError("timeout observed in the candidate rollout")
    return (
        total.reshape(STATES, CANDIDATES).cpu(),
        violations.reshape(STATES, CANDIDATES).cpu(),
        steps.reshape(STATES, CANDIDATES).cpu(),
    )


@torch.no_grad()
def rank_metrics(scores: torch.Tensor, returns: torch.Tensor) -> dict:
    states = returns.shape[0]
    spearmans = []
    hits = 0
    regrets = []
    normalized_regrets = []
    ranges = []
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
        ranges.append(spread)
        normalized_regrets.append(regret / spread if spread > 0 else 0.0)
    spearman_tensor = torch.tensor(spearmans, dtype=torch.float64)
    regret_tensor = torch.tensor(regrets, dtype=torch.float64)
    range_tensor = torch.tensor(ranges, dtype=torch.float64)
    return {
        "states": states,
        "spearman_mean": float(spearman_tensor.mean()),
        "spearman_median": float(spearman_tensor.median()),
        "spearman_positive_fraction": float((spearman_tensor > 0).double().mean()),
        "spearman_state_ci": bootstrap_mean(spearman_tensor, 202611141),
        "argmax_hit_rate": hits / states,
        "top1_regret_mean": float(regret_tensor.mean()),
        "top1_regret_normalized_mean": float(
            torch.tensor(normalized_regrets, dtype=torch.float64).mean()
        ),
        "realized_within_state_range_mean": float(range_tensor.mean()),
        "realized_within_state_range_p50": float(range_tensor.median()),
        "fraction_states_with_spread": float((range_tensor > 0).double().mean()),
        "q_advantage_mean": float(q_advantages.mean()),
        "realized_advantage_mean": float(realized_advantages.mean()),
        "advantage_pearson_pairs": pearson(q_advantages, realized_advantages),
        "advantage_pearson_state_means": pearson(
            q_advantages.mean(-1), realized_advantages.mean(-1)
        ),
        "candidate0_q_share": float((scores.argmax(-1) == 0).double().mean()),
    }


def ceiling(returns_a: torch.Tensor, returns_b: torch.Tensor) -> dict:
    """Seed-to-seed rank agreement of the same candidate set (Warp noise floor)."""
    correlations = []
    argmax_agreements = 0
    for state in range(returns_a.shape[0]):
        correlations.append(spearman(returns_a[state], returns_b[state]))
        argmax_agreements += int(
            int(returns_a[state].argmax()) == int(returns_b[state].argmax())
        )
    tensor = torch.tensor(correlations, dtype=torch.float64)
    return {
        "spearman_mean": float(tensor.mean()),
        "spearman_median": float(tensor.median()),
        "argmax_agreement": argmax_agreements / returns_a.shape[0],
        "return_abs_difference_mean": float((returns_a - returns_b).abs().mean()),
    }


@torch.no_grad()
def main():
    args = build_parser().parse_args()
    if args.artifact_names is None:
        args.artifact_names = [Path(path).name for path in args.artifacts]
    if len(args.artifact_names) != len(args.artifacts):
        raise ValueError("--artifact_names must match --artifacts")
    if len(args.seeds) != 2:
        raise ValueError("rankings need exactly two seeds for the ceiling")
    import gym.envs  # noqa: F401

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = [ContactQ(path) for path in args.artifacts]
    for artifact in artifacts[1:]:
        if artifact.manifest["policy_spec"] != artifacts[0].manifest["policy_spec"]:
            raise ValueError("policy mismatch between artifacts")
        if artifact.manifest["reward_spec"] != artifacts[0].manifest["reward_spec"]:
            raise ValueError("reward spec mismatch between artifacts")
    reward_spec = ContactForceRewardSpec(**artifacts[0].manifest["reward_spec"])
    policy, spec = load_policy_from_artifact(artifacts[0], args.checkpoint, args.device)
    if (
        sha256_file(args.checkpoint) != spec.checkpoint_sha256
        or spec.n_policy_steps != 1
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
    train.runner.device = args.device
    task_registry.convert_frequencies_to_params(cfg, train)
    if (
        replay_config_fingerprint(class_to_dict(cfg))
        != artifacts[0].manifest["dataset"]["config_fingerprint"]
    ):
        raise ValueError("environment fingerprint mismatch")
    # The prefix consumes PREFIX_STEPS before the insertion state, and the
    # longest rollout runs from there, so the episode must outlast both.
    cfg.env.episode_length_s = max(
        cfg.env.episode_length_s,
        (PREFIX_STEPS + max(args.horizons) + 5) * cfg.control.ctrl_dt + 1,
    )
    cfg.seed = args.export_seed
    env = task_registry.make_env("go2trot", cfg, args.device, True, "mujoco")
    feet = foot_indices(env, reward_spec)
    try:
        exported_path = output / "insertion.pt"
        if exported_path.exists():
            exported = torch.load(exported_path, weights_only=True)
            if exported["checkpoint_sha256"] != spec.checkpoint_sha256:
                raise ValueError("insertion snapshot belongs to another checkpoint")
            snapshot = WarpInsertionSnapshot(env)
            snapshot.load_export(exported["snapshot"])
            snapshot.restore()
            snapshot.assert_restored()
        else:
            torch.manual_seed(args.export_seed)
            exported = export_insertion(
                env, policy, spec, feet, reward_spec, args.export_seed
            )
            exported["checkpoint_sha256"] = spec.checkpoint_sha256
            exported["env_config"] = json.loads(json.dumps(class_to_dict(cfg)))
            torch.save(exported, exported_path)
            snapshot = WarpInsertionSnapshot(env)
            snapshot.load_export(exported["snapshot"])
            snapshot.restore()
            snapshot.assert_restored()
        install_paired_environment_noise(env, CANDIDATES)
        observations = exported["insertion_observations"].to(env.device)
        candidates = exported["candidates"].to(env.device)
        scores = {
            name: artifact.predict(observations, candidates.flatten(0, 1))
            .cpu()
            .reshape(STATES, CANDIDATES)
            for name, artifact in zip(args.artifact_names, artifacts, strict=True)
        }
        report = {
            "protocol": {
                "source": str(exported_path),
                "source_sha256": sha256_file(exported_path),
                "checkpoint_sha256": spec.checkpoint_sha256,
                "artifact_names": args.artifact_names,
                "artifacts": {
                    path: {
                        field: sha256_file(Path(path) / field)
                        for field in ("manifest.json", "q.pt")
                    }
                    for path in args.artifacts
                },
                "states": STATES,
                "candidates_per_state": CANDIDATES,
                "candidate_0": "policy mean (eps = 0)",
                "horizons": args.horizons,
                "seeds": args.seeds,
                "gamma": EVAL_GAMMA,
                "reward": "contact force label, zero after terminal",
                "reward_spec": artifacts[0].manifest["reward_spec"],
                "replication_validation": exported.get("validation"),
                "revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
            },
            "results": {},
        }
        for horizon in args.horizons:
            realized = []
            for seed in args.seeds:
                returns, violations, steps = rollout_candidates(
                    env,
                    policy,
                    spec,
                    snapshot,
                    candidates,
                    feet,
                    reward_spec,
                    seed,
                    horizon,
                )
                realized.append(
                    {
                        "seed": seed,
                        "returns": returns,
                        "violations": violations,
                        "steps": steps,
                    }
                )
            entry = {
                "horizon": horizon,
                "violation_rate_mean": float(
                    (realized[0]["violations"].sum() / realized[0]["steps"].sum())
                ),
                "return_mean": float(realized[0]["returns"].mean()),
                "ceiling": ceiling(realized[0]["returns"], realized[1]["returns"]),
                "rankings": {},
            }
            for name in args.artifact_names:
                entry["rankings"][name] = {
                    "seed_%d" % realized[0]["seed"]: rank_metrics(
                        scores[name], realized[0]["returns"]
                    ),
                }
            report["results"][str(horizon)] = entry
            print(
                json.dumps(
                    {
                        "horizon": horizon,
                        "return_mean": entry["return_mean"],
                        "spread": entry["rankings"][args.artifact_names[0]][
                            "seed_%d" % realized[0]["seed"]
                        ]["realized_within_state_range_mean"],
                        "spearman": entry["rankings"][args.artifact_names[0]][
                            "seed_%d" % realized[0]["seed"]
                        ]["spearman_mean"],
                        "ceiling": entry["ceiling"]["spearman_mean"],
                    }
                ),
                flush=True,
            )
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"report={output / 'report.json'}")
    finally:
        if hasattr(env, "_backend"):
            env._backend.close()


if __name__ == "__main__":
    main()
