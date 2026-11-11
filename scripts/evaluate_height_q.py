"""Validate a height Q artifact against fresh simulator continuations."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from gym.utils.helpers import class_to_dict
from gym.utils.task_registry import task_registry
from learning.offline.contracts import (
    EvaluationSpec,
    HeightRewardSpec,
    replay_config_fingerprint,
    sha256_file,
)
from learning.offline.evaluation import (
    evaluate_height_q,
    load_policy_from_artifact,
)
from learning.offline.q_artifact import HeightQ


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--experiment_name", default="go2trot")
    parser.add_argument("--load_run", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--original_cfg", action="store_true", default=False)
    parser.add_argument("--backend", choices=("mujoco", "vsim"), default="mujoco")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--repetitions", type=int, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, required=True)
    parser.add_argument("--candidate_actions", required=True)
    parser.add_argument("--output", required=True)
    return parser


def _load_candidates(path: str, action_dim: int) -> torch.Tensor:
    source = Path(path)
    if source.suffix == ".npy":
        candidates = torch.from_numpy(np.load(source)).float()
    elif source.suffix in (".pt", ".pth"):
        candidates = torch.load(source, map_location="cpu", weights_only=True).float()
    else:
        raise ValueError("candidate actions must be .npy or .pt")
    if candidates.ndim == 1:
        candidates = candidates.unsqueeze(0)
    if candidates.shape[1] != action_dim:
        raise ValueError("candidate file has the wrong action width")
    if not torch.isfinite(candidates).all():
        raise ValueError("candidate actions contain non-finite values")
    return candidates


def main() -> None:
    args = build_parser().parse_args()
    import gym.envs  # noqa: F401

    artifact = HeightQ(args.artifact)
    policy, policy_spec = load_policy_from_artifact(
        artifact,
        args.checkpoint,
        args.device,
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
    candidates = _load_candidates(args.candidate_actions, policy_spec.action_dim)
    spec = EvaluationSpec(
        trials=args.trials,
        repetitions=args.repetitions,
        horizon=args.horizon,
        thresholds=tuple(args.thresholds),
        seed=args.seed,
        device=args.device,
    )

    env_cfg, train_cfg = copy.deepcopy(
        task_registry.get_cfgs(
            args.experiment_name,
            original_cfg=args.original_cfg,
            experiment_name=args.experiment_name,
            load_run=args.load_run,
        )
    )
    env_cfg.env.num_envs = 1
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
    env_cfg.env.episode_length_s = max(
        env_cfg.env.episode_length_s,
        args.horizon * control_dt + 1.0,
    )
    if abs(env_cfg.control.ctrl_dt - policy_spec.control_dt) > 1e-12:
        raise SystemExit("environment control dt does not match the artifact")
    if policy_spec.n_policy_steps != 1:
        raise SystemExit("evaluation currently requires n_policy_steps == 1")

    def env_factory(seed: int):
        cfg = copy.deepcopy(env_cfg)
        cfg.seed = seed
        return task_registry.make_env(
            args.experiment_name,
            cfg,
            args.device,
            True,
            args.backend,
        )

    report = evaluate_height_q(
        artifact=artifact,
        policy=policy,
        policy_spec=policy_spec,
        env_factory=env_factory,
        candidates=candidates,
        spec=spec,
        reward_spec=reward_spec,
    )
    report["environment"] = {
        "experiment_name": args.experiment_name,
        "load_run": args.load_run,
        "backend": args.backend,
        "device": args.device,
        "env_config": class_to_dict(env_cfg),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    aggregate = report["aggregate"]
    print(
        json.dumps(
            {
                "output": str(output),
                "ranking_pair_accuracy": aggregate["ranking_pair_accuracy"],
                "candidates": len(aggregate["candidates"]),
            }
        )
    )


if __name__ == "__main__":
    main()
