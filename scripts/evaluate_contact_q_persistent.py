"""Persistent contact-force intervention: plain, max-Q and Q-threshold.

Persistent means the controller acts at *every* decision, so this is not
``Q^pi`` validation: executing Q changes the continuation policy. It measures
what an operator would get by deploying the Q as a selector over eight
stochastic policy draws.

Controllers per artifact: ``plain`` (first stochastic draw), ``maxq`` (argmax
over the same eight draws), and one ``rejection`` controller per threshold
(first draw with ``Q >= threshold``, otherwise the max-Q draw). All controllers
of a block consume the same draw block, and all worlds start from the frozen
insertion snapshot, so differences are paired.

Reported per controller: discounted contact-force return (zero after terminal),
violation rate and per-foot violations, velocity/yaw tracking error, the
original PPO reward per step (``TrainingRewardModel``), survival, and the
selection/attempt statistics. The paired gain is against ``plain`` on the same
lane and state.

Usage:

    uv run --frozen python scripts/evaluate_contact_q_persistent.py \
      --source logs/go2trot/<run>/contact_ranking \
      --output logs/go2trot/<run>/contact_persistent \
      --checkpoint logs/go2trot/<run>/model_616.pt --load_run <run> \
      --artifacts logs/go2trot/<run>/offline_q_60k/contact_q/data/100pct \
      --thresholds -2.5 -2.0 -1.5 --blocks 4
"""

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import torch

from evaluate_height_q_variance import fidelity
from gym.utils.helpers import class_to_dict
from gym.utils.task_registry import task_registry
from learning.offline.contracts import (
    ContactForceRewardSpec,
    replay_config_fingerprint,
    sha256_file,
)
from learning.offline.evaluation import load_policy_from_artifact
from learning.offline.fixed_state_variance import (
    WarpInsertionSnapshot,
    install_paired_environment_noise,
)
from learning.offline.persistent_intervention import (
    rollout_contact,
    rollout_rejection_contact,
)
from learning.offline.q_artifact import ContactQ
from learning.offline.training_reward import TrainingRewardModel

STATES = 128
LANES = 8


class NegatedCritic:
    """Argmax over negated scores = argmin over the real Q (control)."""

    def __init__(self, critic):
        self.critic = critic

    def predict(self, observations, actions):
        return -self.critic.predict(observations, actions)


class RandomCritic:
    """Uniform random selection among the eight draws (control)."""

    def predict(self, observations, actions):
        return torch.randn(observations.shape[0], device=observations.device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--load_run", required=True)
    parser.add_argument("--artifacts", nargs="+", required=True)
    parser.add_argument("--artifact_names", nargs="+", default=None)
    parser.add_argument(
        "--rejection_artifact",
        type=int,
        default=0,
        help="Artifact index used by the rejection controllers.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[-2.5, -2.0, -1.5],
        help="Q thresholds for the rejection controllers (higher = stricter).",
    )
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7700000)
    parser.add_argument("--device", default="cuda:0")
    return parser


def paired_summary(values: torch.Tensor) -> dict:
    values = values.double()
    generator = torch.Generator().manual_seed(202611142)
    draws = torch.randint(len(values), (10000, len(values)), generator=generator)
    means = values[draws].mean(-1)
    return {
        "rows": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "ci": [float(means.quantile(0.025)), float(means.quantile(0.975))],
        "positive_fraction": float((values > 0).double().mean()),
        "negative_fraction": float((values < 0).double().mean()),
    }


def summarize(result: dict, states: int, lanes: int) -> dict:
    returns = result["returns"].reshape(states, lanes).mean(1)
    alive = result["survived"].reshape(lanes, states)
    by_state = {
        "contact_return": returns,
        "violation_rate": (
            result["violation_steps"].reshape(states, lanes).mean(1)
            / result["steps"].reshape(states, lanes).mean(1)
        ),
        "xy_error": (
            result["xy_error_sq"].reshape(states, lanes).mean(1)
            / result["steps"].reshape(states, lanes).mean(1)
        ).sqrt(),
        "yaw_error": (
            result["yaw_error_sq"].reshape(states, lanes).mean(1)
            / result["steps"].reshape(states, lanes).mean(1)
        ).sqrt(),
    }
    training_steps = result["steps"].reshape(states, lanes).mean(1)
    by_state["training_reward_per_step"] = (
        result["training_reward"].reshape(states, lanes).mean(1) / training_steps
    )
    summary = {
        "contact_return": {
            "mean": float(returns.mean()),
            "ci": [
                float(
                    returns.mean()
                    - 1.96 * returns.std(unbiased=False) / len(returns) ** 0.5
                ),
                float(
                    returns.mean()
                    + 1.96 * returns.std(unbiased=False) / len(returns) ** 0.5
                ),
            ],
        },
        "violation_rate_mean": float(by_state["violation_rate"].mean()),
        "xy_error_rms": float(by_state["xy_error"].mean()),
        "yaw_error_rms": float(by_state["yaw_error"].mean()),
        "training_reward_per_step": float(by_state["training_reward_per_step"].mean()),
        "survived_fraction": float(alive.double().mean()),
        "mean_termination_step": float(result["termination_step"].mean()),
        "selection_share": (result["selection_counts"].double().mean(0).tolist()),
    }
    for key in ("attempts", "fallback"):
        if key in result:
            summary[f"{key}_mean"] = float(result[key].mean())
    return summary, by_state


@torch.no_grad()
def main():
    args = build_parser().parse_args()
    if args.artifact_names is None:
        args.artifact_names = [Path(path).name for path in args.artifacts]
    if len(args.artifact_names) != len(args.artifacts):
        raise ValueError("--artifact_names must match --artifacts")
    if not 0 <= args.rejection_artifact < len(args.artifacts):
        raise ValueError("--rejection_artifact is out of range")
    if args.blocks <= 0 or args.horizon <= 0:
        raise ValueError("blocks and horizon must be positive")
    import gym.envs  # noqa: F401

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(args.source) / "insertion.pt"
    saved = torch.load(source, weights_only=True)
    artifacts = [ContactQ(path) for path in args.artifacts]
    reward_spec = ContactForceRewardSpec(**artifacts[0].manifest["reward_spec"])
    for artifact in artifacts[1:]:
        if artifact.manifest["policy_spec"] != artifacts[0].manifest["policy_spec"]:
            raise ValueError("policy mismatch between artifacts")
        if artifact.manifest["reward_spec"] != artifacts[0].manifest["reward_spec"]:
            raise ValueError("reward spec mismatch between artifacts")
    policy, spec = load_policy_from_artifact(artifacts[0], args.checkpoint, args.device)
    if (
        sha256_file(args.checkpoint) != spec.checkpoint_sha256
        or saved["checkpoint_sha256"] != spec.checkpoint_sha256
        or spec.n_policy_steps != 1
    ):
        raise ValueError("checkpoint/interval mismatch")

    provenance = {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": spec.checkpoint_sha256,
        "artifacts": {
            path: {
                name: sha256_file(Path(path) / name)
                for name in ("manifest.json", "q.pt")
            }
            for path in args.artifacts
        },
        "artifact_names": args.artifact_names,
        "rejection_artifact": args.rejection_artifact,
        "thresholds": args.thresholds,
        "blocks": args.blocks,
        "horizon": args.horizon,
        "reward_spec": artifacts[0].manifest["reward_spec"],
        "reward_timing": "post_step contact_forces/bodyweight, zero on termination",
        "mode": "persistent intervention (not Q^pi validation)",
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (output / "commands.txt").open("a").write(
        "uv run --frozen python " + " ".join(sys.argv) + "\n"
    )

    cfg, train = copy.deepcopy(
        task_registry.get_cfgs(
            "go2trot",
            original_cfg=False,
            experiment_name="go2trot",
            load_run=args.load_run,
        )
    )
    cfg.env.num_envs = STATES * LANES
    train.runner.device = args.device
    task_registry.convert_frequencies_to_params(cfg, train)
    if (
        replay_config_fingerprint(class_to_dict(cfg))
        != artifacts[0].manifest["dataset"]["config_fingerprint"]
    ):
        raise ValueError("environment fingerprint mismatch")
    # The snapshot carries the ranking export's episode counter (~100 steps of
    # prefix), and the intervention runs another `horizon` steps from there.
    cfg.env.episode_length_s = max(
        cfg.env.episode_length_s, (args.horizon + 150) * cfg.control.ctrl_dt + 1
    )
    cfg.seed = saved["seed"]
    env = task_registry.make_env("go2trot", cfg, args.device, True, "mujoco")
    try:
        # The rollout compares Q scores against on-device liveness masks, so the
        # critic must produce device tensors.
        for artifact in artifacts:
            artifact.q.to(env.device)
            artifact.obs_mean = artifact.obs_mean.to(env.device)
            artifact.obs_std = artifact.obs_std.to(env.device)
        # The exported insertion snapshot already holds the replicated worlds in
        # state-major, lane-minor order (world 8*s+l is lane l of state s), so it
        # restores by identity and worlds 8s..8s+7 are the same physical state.
        actions = saved["candidates"].flatten(0, 1).to(env.device)
        snapshot = WarpInsertionSnapshot(env)
        snapshot.load_export(saved["snapshot"])
        snapshot.restore()
        snapshot.assert_restored()
        validation = {"lane_fidelity": fidelity(env, actions, snapshot)}
        snapshot.restore()
        install_paired_environment_noise(env, LANES)
        observations = saved["insertion_observations"].to(env.device)
        names = list(env.robot_layout.body_names)
        feet = torch.tensor(
            [names.index(name) for name in reward_spec.foot_names],
            dtype=torch.long,
            device=env.device,
        )
        training_model = TrainingRewardModel(
            env,
            class_to_dict(train.critic.reward.weights),
            class_to_dict(train.critic.reward.termination_weight),
            gamma=float(train.algorithm.gamma),
        )
        report_path = output / "report.json"
        report = {
            "provenance": provenance,
            "validation": validation,
            "blocks": [],
        }
        # One draw block per callback; every controller of a block consumes it.
        for block in range(args.blocks):
            seed = args.seed + 10 * block
            torch.manual_seed(seed)
            results = {}
            for name, artifact in zip(args.artifact_names, artifacts, strict=True):
                results[f"{name}_plain"] = rollout_contact(
                    env,
                    policy,
                    spec,
                    snapshot,
                    observations,
                    None,
                    seed,
                    reward_spec,
                    feet,
                    args.horizon,
                    training_model=training_model,
                )
                # Controls are critic-independent apart from "minq", which uses
                # this artifact; run them once with the first one.
                if name == args.artifact_names[0]:
                    results[f"{name}_minq"] = rollout_contact(
                        env,
                        policy,
                        spec,
                        snapshot,
                        observations,
                        NegatedCritic(artifact),
                        seed,
                        reward_spec,
                        feet,
                        args.horizon,
                        training_model=training_model,
                    )
                    results[f"{name}_random"] = rollout_contact(
                        env,
                        policy,
                        spec,
                        snapshot,
                        observations,
                        RandomCritic(),
                        seed,
                        reward_spec,
                        feet,
                        args.horizon,
                        training_model=training_model,
                    )
                results[f"{name}_maxq"] = rollout_contact(
                    env,
                    policy,
                    spec,
                    snapshot,
                    observations,
                    artifact,
                    seed,
                    reward_spec,
                    feet,
                    args.horizon,
                    training_model=training_model,
                )
                for threshold in args.thresholds:
                    results[f"{name}_reject{threshold:+.2f}"] = (
                        rollout_rejection_contact(
                            env,
                            policy,
                            spec,
                            snapshot,
                            observations,
                            artifact,
                            seed,
                            reward_spec,
                            feet,
                            threshold,
                            args.horizon,
                            training_model=training_model,
                        )
                    )
            # Per-state, per-lane arrays so a later summary can ask which initial
            # condition degraded and whether an intervention zeroed it. Coarse
            # means hide exactly that question.
            torch.save(
                {
                    "seed": seed,
                    "states": STATES,
                    "lanes": LANES,
                    "controllers": {
                        key: {
                            "returns": value["returns"].reshape(STATES, LANES),
                            "violation_steps": value["violation_steps"].reshape(
                                STATES, LANES
                            ),
                            "steps": value["steps"].reshape(STATES, LANES),
                            "survived": value["survived"].reshape(STATES, LANES),
                        }
                        for key, value in results.items()
                    },
                },
                output / f"block_{block:02d}.pt",
            )
            block_entry = {"block": block, "seed": seed, "controllers": {}}
            plain_by_artifact = {
                name: results[f"{name}_plain"] for name in args.artifact_names
            }
            for key, result in results.items():
                summary, by_state = summarize(result, STATES, LANES)
                base_name = key.split("_")[0]
                plain = plain_by_artifact[base_name]
                plain_state = plain["returns"].reshape(LANES, STATES).mean(0)
                summary["paired_gain_vs_plain"] = paired_summary(
                    by_state["contact_return"] - plain_state
                )
                summary["paired_violation_reduction"] = paired_summary(
                    plain["violation_steps"].reshape(STATES, LANES).mean(1)
                    - result["violation_steps"].reshape(STATES, LANES).mean(1)
                )
                block_entry["controllers"][key] = summary
            report["blocks"].append(block_entry)
            print(
                json.dumps(
                    {
                        "block": block,
                        **{
                            key: {
                                "return": round(value["contact_return"]["mean"], 3),
                                "viol": round(value["violation_rate_mean"], 4),
                                "gain": round(value["paired_gain_vs_plain"]["mean"], 3),
                                "train_r": round(value["training_reward_per_step"], 5),
                            }
                            for key, value in block_entry["controllers"].items()
                        },
                    }
                ),
                flush=True,
            )
        (output / "validation.json").write_text(json.dumps(validation, indent=2))
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"report={report_path}")
    finally:
        if hasattr(env, "_backend"):
            env._backend.close()


if __name__ == "__main__":
    main()
