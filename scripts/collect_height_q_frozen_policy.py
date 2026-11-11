"""Collect a replay snapshot by unrolling one frozen stochastic PPO actor.

This is deliberately not an evaluation rollout: it uses the normal training
environment, nominal Gaussian policy actions, administrative timeouts and
resets, and the existing replay recorder contract. The output is a replay
snapshot that ``scripts/train_height_q.py`` can relabel and fit.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from gym.utils.helpers import class_to_dict
from gym.utils.task_registry import task_registry
from learning.offline.contracts import (
    replay_config_fingerprint,
    sha256_file,
    utc_timestamp,
)
from learning.offline.evaluation import _policy_observation
from learning.offline.training import load_policy
from learning.storage.transition_recorder import TransitionRecorder

EXPECTED_CONFIG_FINGERPRINT = (
    "6412a8784c883ecb3b28c9601ad1871bec4acb97bfee0f5d7b13e77acc9938cb"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference_snapshot", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiment_name", default="go2trot")
    parser.add_argument("--load_run", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--transitions", type=int, default=1_000_000)
    parser.add_argument("--total_steps", type=int, default=2_500_000)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--run_id", default="fresh-frozen-model600-20261111")
    parser.add_argument("--snapshot_iteration", type=int, default=600)
    parser.add_argument("--behavior_iteration", type=int, default=600)
    parser.add_argument("--progress_every", type=int, default=50)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.transitions <= 0 or args.total_steps < args.transitions:
        raise ValueError("total_steps must be at least transitions")
    if args.num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if args.seed < 0 or args.behavior_iteration < 0:
        raise ValueError("seed and behavior_iteration must be nonnegative")
    import gym.envs  # noqa: F401

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    reference_manifest = json.loads(
        (Path(args.reference_snapshot) / "manifest.json").read_text()
    )
    checkpoint_sha256 = sha256_file(args.checkpoint)

    cfg, train = copy.deepcopy(
        task_registry.get_cfgs(
            args.experiment_name,
            original_cfg=False,
            experiment_name=args.experiment_name,
            load_run=args.load_run,
        )
    )
    cfg.env.num_envs = args.num_envs
    train.runner.device = args.device
    task_registry.convert_frequencies_to_params(cfg, train)
    fingerprint = replay_config_fingerprint(class_to_dict(cfg))
    if fingerprint != EXPECTED_CONFIG_FINGERPRINT:
        raise ValueError(
            f"environment fingerprint {fingerprint} does not match the frozen "
            f"training fingerprint {EXPECTED_CONFIG_FINGERPRINT}"
        )
    if float(cfg.env.episode_length_s) != 5.0:
        raise ValueError(
            "this collection requires the historical 5 s administrative timeout"
        )

    policy, spec = load_policy(reference_manifest, args.checkpoint, args.device)
    if spec.n_policy_steps != 1:
        raise ValueError("collection requires n_policy_steps == 1")
    if spec.checkpoint_sha256 != checkpoint_sha256:
        raise ValueError("policy spec checkpoint hash mismatch")

    cfg.seed = args.seed
    env = task_registry.make_env(
        args.experiment_name,
        cfg,
        args.device,
        True,
        "mujoco",
    )
    try:
        if env.num_envs != args.num_envs:
            raise RuntimeError("environment did not honor num_envs")
        expected_steps = int(env.max_episode_length)
        if expected_steps != 250:
            raise RuntimeError(
                f"expected 250 policy steps for the 5 s timeout, got {expected_steps}"
            )
        obs_dims = tuple(int(dim) for dim in reference_manifest["obs_dims"])
        action_dims = tuple(int(dim) for dim in reference_manifest["action_dims"])
        recorder = TransitionRecorder(
            num_envs=env.num_envs,
            obs_names=tuple(reference_manifest["obs_names"]),
            obs_dims=obs_dims,
            action_names=tuple(reference_manifest["action_names"]),
            action_dims=action_dims,
            noise=reference_manifest["noise"],
            scales=reference_manifest["scales"],
            device=args.device,
            output_dir=str(output),
            capacity=args.transitions,
            snapshot_interval=0,
            seed=args.seed,
            task_name=str(reference_manifest["task_name"]),
            backend="mujoco",
            source_revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            log_dir=None,
            actor_cfg=reference_manifest["actor_cfg"],
            env_cfg=class_to_dict(cfg),
            runner_cfg={
                "kind": "frozen-policy replay collection",
                "checkpoint": args.checkpoint,
                "checkpoint_sha256": checkpoint_sha256,
                "behavior_iteration": args.behavior_iteration,
            },
            episode_length_s=float(env.max_episode_length_s),
            n_policy_steps=spec.n_policy_steps,
            control_dt=float(env.dt),
        )
        # The recorder defaults to a timestamped run identity when no training
        # log_dir exists; pin the experiment identity explicitly.
        recorder.run_id = args.run_id

        obs_generator = torch.Generator(device=env.device).manual_seed(args.seed + 1)
        action_generator = torch.Generator(device=env.device).manual_seed(args.seed + 2)
        obs = _policy_observation(
            env,
            spec.obs_names,
            spec.obs_dims,
            spec.noise,
            spec.scales,
            obs_generator,
        )
        steps = 0
        started = time.perf_counter()
        while steps * env.num_envs < args.total_steps:
            actions = policy.sample_actions(obs, action_generator)
            env.set_states(spec.action_names, actions)
            env.step()

            pre_reset_clean_obs = env.get_states(list(spec.obs_names))
            successor_height = env.base_height.clone()
            terminated = env.terminated.clone()
            timed_out = env.timed_out.clone()

            # Match the training runner: reset only ended worlds, then choose
            # the next action from the new post-reset noisy observation.
            env.to_be_reset = terminated | timed_out
            env._reset_idx(env.to_be_reset)
            post_reset_obs = _policy_observation(
                env,
                spec.obs_names,
                spec.obs_dims,
                spec.noise,
                spec.scales,
                obs_generator,
            )
            recorder.append(
                actor_obs=obs,
                action=actions,
                post_reset_obs=post_reset_obs,
                pre_reset_clean_obs=pre_reset_clean_obs,
                successor_height_m=successor_height,
                terminated=terminated,
                timed_out=timed_out,
                behavior_iteration=args.behavior_iteration,
            )
            obs = post_reset_obs
            steps += 1
            if steps % args.progress_every == 0:
                elapsed = time.perf_counter() - started
                print(
                    json.dumps(
                        {
                            "steps": steps,
                            "transitions_appended": recorder.total_appended,
                            "valid_count": recorder.valid_count,
                            "elapsed_s": elapsed,
                            "env_steps_per_s": (
                                steps * env.num_envs / elapsed if elapsed else None
                            ),
                        }
                    ),
                    flush=True,
                )
        elapsed = time.perf_counter() - started
        snapshot = recorder.snapshot(args.snapshot_iteration)
        if snapshot is None:
            raise RuntimeError("recorder did not publish a snapshot")
        if recorder.valid_count != args.transitions:
            raise RuntimeError(
                f"retained {recorder.valid_count} transitions, expected "
                f"{args.transitions}"
            )
        data = torch.load(snapshot / "data.pt", map_location="cpu", weights_only=True)
        if int(data["terminated"].sum()) + int(data["timed_out"].sum()) == 0:
            raise RuntimeError("collection produced no episode boundaries")
        episode_pairs = torch.stack((data["env_id"], data["episode_id"]), dim=1).unique(
            dim=0
        )
        provenance = {
            "created_at": utc_timestamp(),
            "command": " ".join(["collect_height_q_frozen_policy.py", *sys.argv[1:]]),
            "snapshot": str(snapshot),
            "snapshot_sha256": {
                "data.pt": sha256_file(snapshot / "data.pt"),
                "manifest.json": sha256_file(snapshot / "manifest.json"),
            },
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": checkpoint_sha256,
            "environment_fingerprint": fingerprint,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "policy_steps": steps,
            "transitions_appended": recorder.total_appended,
            "retained_transitions": recorder.valid_count,
            "episode_length_s": float(env.max_episode_length_s),
            "max_policy_steps": int(env.max_episode_length),
            "control_dt": float(env.dt),
            "terminated_rows": int(data["terminated"].sum()),
            "timed_out_rows": int(data["timed_out"].sum()),
            "unique_episode_env_pairs": int(episode_pairs.shape[0]),
            "wall_seconds": elapsed,
            "policy_spec": reference_manifest,
            "code_sha256": {
                str(path): sha256_file(path)
                for path in (
                    Path(__file__),
                    Path("learning/storage/transition_recorder.py"),
                    Path("learning/offline/contracts.py"),
                    Path("learning/offline/evaluation.py"),
                    Path("learning/offline/training.py"),
                    Path("learning/offline/nominal_policy.py"),
                )
            },
        }
        (output / "collect_provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        )
        print(
            json.dumps(
                {
                    "snapshot": str(snapshot),
                    "retained_transitions": recorder.valid_count,
                    "transitions_appended": recorder.total_appended,
                    "policy_steps": steps,
                    "wall_seconds": elapsed,
                    "terminated_rows": int(data["terminated"].sum()),
                    "timed_out_rows": int(data["timed_out"].sum()),
                    "unique_episode_env_pairs": int(episode_pairs.shape[0]),
                }
            ),
            flush=True,
        )
    finally:
        env._backend.close()


if __name__ == "__main__":
    main()
