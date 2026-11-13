"""Persistent yaw-tracking intervention: plain, max-Q and rejection controllers."""

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from evaluate_height_q_variance import fidelity, physical
from gym.utils.helpers import class_to_dict
from gym.utils.task_registry import task_registry
from learning.offline.contracts import (
    ZERO_AFTER_TERMINAL,
    AngularVelocityRewardSpec,
    replay_config_fingerprint,
    sample_observation_noise,
    sha256_file,
)
from learning.offline.evaluation import load_policy_from_artifact
from learning.offline.fixed_state_variance import WarpInsertionSnapshot
from learning.offline.persistent_intervention import (
    rollout_angular_vel,
    rollout_rejection_angular_vel,
    sample_candidates,
)
from learning.offline.q_artifact import AngularVelocityQ
from learning.offline.rewards import (
    angular_velocity_reward,
    observation_field_slice,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--load_run", required=True)
    p.add_argument("--artifacts", nargs="+", required=True)
    p.add_argument(
        "--artifact_names",
        nargs="+",
        default=None,
        help="Controller names aligned with --artifacts; defaults to the "
        "artifact directory names.",
    )
    p.add_argument(
        "--rejection_artifact",
        type=int,
        default=0,
        help="Artifact index used for the rejection controllers.",
    )
    p.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[],
        help="Q thresholds; each adds a rejection controller.",
    )
    p.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Do not rerun the plain baseline; reuse blocks from another run.",
    )
    p.add_argument("--blocks", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--trace_steps",
        type=int,
        default=0,
        help="When positive, also record per-step yaw/command traces for the "
        "plain and first max-Q controllers.",
    )
    p.add_argument(
        "--trace_names",
        nargs="+",
        default=None,
        help="Max-Q controller names to trace (default: the first artifact).",
    )
    return p


def validate_artifacts(artifacts, saved, checkpoint, spec):
    expected = (0.25, ZERO_AFTER_TERMINAL, 0.99, 2.5, 0.0)
    for q in artifacts:
        if q.manifest["policy_spec"] != artifacts[0].manifest["policy_spec"]:
            raise ValueError("policy mismatch")
        r = q.manifest["reward_spec"]
        observed = (
            r["tracking_sigma"],
            r["terminal_continuation"],
            q.manifest["gamma"],
            r["error_scale"],
            q.manifest["failure_value"],
        )
        if observed != expected:
            raise ValueError(f"reward/discount mismatch: {observed} != {expected}")
        if q.manifest["return_convention"] != (
            "discounted_episodic_zero_after_terminal"
        ):
            raise ValueError("return convention mismatch")
        if q.obs_dim != 48 or q.action_dim != 12:
            raise ValueError("unexpected Q input dimensions")
    if (
        sha256_file(checkpoint) != spec.checkpoint_sha256
        or saved["checkpoint_sha256"] != spec.checkpoint_sha256
        or spec.n_policy_steps != 1
    ):
        raise ValueError("checkpoint/interval mismatch")


@torch.no_grad()
def live_reward_parity(env, policy, spec, reward_spec, steps=5):
    """Live r' must equal the task's own raw yaw term on the same states."""
    errors = []
    act_rng = torch.Generator(device=env.device).manual_seed(424243)
    for _ in range(steps):
        obs = env.get_states(spec.obs_names)
        action = policy.sample_actions(obs, act_rng)
        env.set_states(spec.action_names, action)
        env.step()
        reward = angular_velocity_reward(
            env.commands[:, 2], env.base_ang_vel[:, 2], reward_spec
        ) * (~env.terminated)
        reference = env._reward_tracking_ang_vel() * (~env.terminated)
        errors.append(float((reward - reference).abs().max()))
    return {
        "steps": steps,
        "max_absolute_error": max(errors),
        "task_reward_method": "Go2Trot._reward_tracking_ang_vel",
    }


def label_noise_check(env, policy, spec, reward_spec, seed=99000031, steps=50):
    """Quantify relabeled-vs-clean r' RMSE from the recorded observation noise.

    Recording noise comes from ``sample_observation_noise`` (the successor
    observation is ``clean + one policy-marginal draw``), so the relabeled label
    is noisier than the live reward that evaluation computes from clean task
    tensors. ``base_ang_vel`` carries ``noise/scales``; ``commands`` carry none.
    """
    contract = {"obs_names": spec.obs_names, "obs_dims": spec.obs_dims}
    command_slice = observation_field_slice(contract, "commands")
    ang_vel_slice = observation_field_slice(contract, "base_ang_vel")
    generator = torch.Generator(device=env.device).manual_seed(seed)
    action_generator = torch.Generator(device=env.device).manual_seed(seed + 1)
    clean_values = []
    noisy_values = []
    for _ in range(steps):
        obs = env.get_states(spec.obs_names)
        env.set_states(spec.action_names, policy.sample_actions(obs, action_generator))
        env.step()
        noise = sample_observation_noise(
            spec.obs_names,
            spec.obs_dims,
            spec.noise,
            spec.scales,
            env.num_envs,
            env.device,
            generator,
        )
        command = env.commands[:, 2]
        ang_vel = env.base_ang_vel[:, 2]
        clean = angular_velocity_reward(command, ang_vel, reward_spec)
        noisy = angular_velocity_reward(
            command + noise[:, command_slice][:, 2],
            ang_vel + noise[:, ang_vel_slice][:, 2],
            reward_spec,
        )
        live = ~env.terminated
        clean_values.append(clean[live])
        noisy_values.append(noisy[live])
    clean_all = torch.cat(clean_values)
    noisy_all = torch.cat(noisy_values)
    difference = noisy_all - clean_all
    return {
        "steps": steps,
        "rows": int(clean_all.numel()),
        "clean_mean": float(clean_all.mean()),
        "noisy_mean": float(noisy_all.mean()),
        "bias": float(difference.mean()),
        "rmse": float(difference.square().mean().sqrt()),
        "mean_absolute": float(difference.abs().mean()),
        "max_absolute": float(difference.abs().max()),
        "relative_rmse": float(
            difference.square().mean().sqrt() / clean_all.square().mean().sqrt()
        ),
        "fraction_difference_above_0.05": float(
            (difference.abs() > 0.05).double().mean()
        ),
        "observed_ang_vel_noise_std": float(
            torch.tensor(spec.noise["base_ang_vel"], dtype=torch.float32)
            / torch.tensor(spec.scales["base_ang_vel"], dtype=torch.float32)
        ),
    }


@torch.no_grad()
def main():
    args = build_parser().parse_args()
    if not 1 <= args.blocks <= 16:
        raise ValueError("need 1..16 blocks (eight calls/block)")
    if args.artifact_names is None:
        args.artifact_names = [Path(path).name for path in args.artifacts]
    if len(args.artifact_names) != len(args.artifacts):
        raise ValueError("--artifact_names must match --artifacts")
    if len(set(args.artifact_names)) != len(args.artifact_names):
        raise ValueError("--artifact_names must be unique")
    if "baseline" in args.artifact_names:
        raise ValueError("'baseline' is reserved for the plain controller")
    if not 0 <= args.rejection_artifact < len(args.artifacts):
        raise ValueError("--rejection_artifact is out of range")
    import gym.envs  # noqa: F401

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=args.resume)
    source = Path(args.source) / "insertion.pt"
    saved = torch.load(source, weights_only=True)
    artifacts = [AngularVelocityQ(path) for path in args.artifacts]
    policy, spec = load_policy_from_artifact(artifacts[0], args.checkpoint, "cuda:0")
    validate_artifacts(artifacts, saved, args.checkpoint, spec)
    reward_spec = artifacts[0].manifest["reward_spec"]
    reward_spec = AngularVelocityRewardSpec(**reward_spec)
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
        "skip_baseline": args.skip_baseline,
        "world_axes": ["lane8", "state128"],
        "candidate_axis": "neural only, eight IID samples",
        "seed_identity": (
            "(7100000+10*call,lane,state); env=seed, obs=seed+1, action=seed+2"
        ),
        "initial_observation": (
            "original noisy insertion observation, candidate0 world, "
            "replicated eight lanes"
        ),
        "reward_timing": (
            "post_step env.commands[:,2]/env.base_ang_vel[:,2], zero on termination"
        ),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    if args.resume:
        if json.loads((output / "provenance.json").read_text()) != provenance:
            raise ValueError("resume provenance mismatch")
    else:
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    with (output / "commands.txt").open("a") as f:
        f.write("uv run --frozen python " + " ".join(sys.argv) + "\n")
    (output / ("dirty_resume.txt" if args.resume else "dirty.txt")).write_text(
        subprocess.check_output(["git", "status", "--short"], text=True)
    )
    code_paths = [
        Path(__file__),
        Path("learning/offline/persistent_intervention.py"),
        Path("learning/offline/fixed_state_variance.py"),
        Path("learning/offline/q_artifact.py"),
        Path("learning/offline/rewards.py"),
    ]
    (output / ("code_resume.json" if args.resume else "code.json")).write_text(
        json.dumps({str(x): sha256_file(x) for x in code_paths}, indent=2)
    )
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
        cfg.env.episode_length_s,
        max(601, args.trace_steps + 101) * cfg.control.ctrl_dt + 1,
    )
    cfg.seed = saved["seed"]
    env = task_registry.make_env("go2trot", cfg, "cuda:0", True, "mujoco")
    try:
        original = WarpInsertionSnapshot(env)
        original.load_export(saved["snapshot"])
        original.restore()
        actions = saved["candidates"].flatten(0, 1).to(env.device)
        validation = {"source_fidelity": fidelity(env, actions, original)}
        torch.manual_seed(654321)
        env.set_states(spec.action_names, actions)
        env.step()
        reference = physical(env)
        ids = (torch.arange(128) * 8).repeat(8)
        snapshot = WarpInsertionSnapshot(env)
        snapshot.load_export(saved["snapshot"], ids)
        snapshot.restore()
        snapshot.assert_restored()
        for entry, (_, key, _, mapped) in zip(
            saved["snapshot"]["entries"], snapshot.entries
        ):
            if not torch.equal(
                mapped.cpu(),
                entry["value"].reshape(1024, -1)[ids].reshape(mapped.shape),
            ):
                raise RuntimeError(f"mapping mismatch {key}")
        torch.manual_seed(654321)
        env.set_states(spec.action_names, actions[ids.to(env.device)])
        env.step()
        errors = {
            k: float((physical(env)[k] - v[ids.to(env.device)]).abs().max())
            for k, v in reference.items()
        }
        if any(errors[k] > validation["source_fidelity"]["limits"][k] for k in errors):
            raise RuntimeError(f"replication failed {errors}")
        validation["replication"] = errors
        snapshot.restore()
        validation["lane_fidelity"] = fidelity(
            env, actions[ids.to(env.device)], snapshot
        )
        validation["live_reward_parity"] = live_reward_parity(
            env, policy, spec, reward_spec
        )
        validation["label_noise"] = label_noise_check(env, policy, spec, reward_spec)
        snapshot.restore()
        obs = saved["insertion_observations"][ids].to(env.device)
        candidates = sample_candidates(
            policy, obs, torch.Generator(device=env.device).manual_seed(99000011)
        )
        validation["cpu_gpu"] = []
        for q in artifacts:
            query = obs[:, None].expand(-1, 8, -1).flatten(0, 1)
            acts = candidates.flatten(0, 1)
            cpu = q.predict(query.cpu(), acts.cpu())
            q.q.to(env.device)
            q.obs_mean = q.obs_mean.to(env.device)
            q.obs_std = q.obs_std.to(env.device)
            gpu = q.predict(query, acts)
            split = torch.cat(
                [q.predict(o, a) for o, a in zip(query.split(257), acts.split(257))]
            )
            torch.testing.assert_close(cpu, gpu.cpu(), atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(gpu, split, atol=2e-5, rtol=2e-5)
            validation["cpu_gpu"].append(
                {
                    "max_error": float((cpu - gpu.cpu()).abs().max()),
                    "batch_error": float((gpu - split).abs().max()),
                    "argmax_disagreements": int(
                        (
                            cpu.reshape(-1, 8).argmax(-1)
                            != gpu.cpu().reshape(-1, 8).argmax(-1)
                        ).sum()
                    ),
                }
            )
        # Identical controller negative control. Warp restore is not bitwise: the
        # continuous yaw reward exposes ~1e-3 five-step replay roundoff that the
        # binary height reward hid, so discrete outputs must match exactly while
        # the continuous return needs an explicit small bound.
        a = rollout_angular_vel(
            env,
            policy,
            spec,
            snapshot,
            obs,
            None,
            99000021,
            reward_spec=reward_spec,
            horizon=5,
        )
        b = rollout_angular_vel(
            env,
            policy,
            spec,
            snapshot,
            obs,
            None,
            99000021,
            reward_spec=reward_spec,
            horizon=5,
        )
        for k in (
            "selection_counts",
            "survived",
            "steps",
            "termination_step",
            "timeout_seen",
            "first_terminated",
        ):
            torch.testing.assert_close(a[k], b[k], atol=0, rtol=0)
        torch.testing.assert_close(a["returns"], b["returns"], atol=1e-2, rtol=0)
        validation["negative_control_5steps"] = {
            "exact_fields": [
                "selection_counts",
                "survived",
                "steps",
                "termination_step",
                "timeout_seen",
                "first_terminated",
            ],
            "return_max_absolute_difference": float(
                (a["returns"] - b["returns"]).abs().max()
            ),
            "return_atol": 1e-2,
            "note": (
                "snapshot restore is not bitwise under Warp; the continuous reward "
                "exposes roundoff that a binary reward hides"
            ),
        }
        floor_a = rollout_angular_vel(
            env,
            policy,
            spec,
            snapshot,
            obs,
            None,
            99000022,
            reward_spec=reward_spec,
        )
        floor_b = rollout_angular_vel(
            env,
            policy,
            spec,
            snapshot,
            obs,
            None,
            99000022,
            reward_spec=reward_spec,
        )
        floor_diff = floor_b["returns"] - floor_a["returns"]
        validation["replay_noise_floor_h500"] = {
            "mean_return": float(floor_a["returns"].mean()),
            "mean_signed_difference": float(floor_diff.mean()),
            "mean_absolute_difference": float(floor_diff.abs().mean()),
            "std_difference": float(floor_diff.std()),
            "max_absolute_difference": float(floor_diff.abs().max()),
            "relative_mean_absolute_difference": float(
                floor_diff.abs().mean() / floor_a["returns"].abs().mean()
            ),
            "note": "two identical plain rollouts at the same seed, 1024 worlds",
        }
        validation["inventory_entries"] = len(snapshot.entries)
        (
            output / ("validation_resume.json" if args.resume else "validation.json")
        ).write_text(json.dumps(validation, indent=2))
        print(json.dumps({"validation": validation}), flush=True)
        controllers = list(zip(args.artifact_names, artifacts, strict=True))
        if args.thresholds:
            rejected = artifacts[args.rejection_artifact]
            for threshold in args.thresholds:
                controllers.append((f"rejection_c{threshold:g}", rejected))
        if not args.skip_baseline:
            controllers.insert(0, ("baseline", None))
        for block in range(args.blocks):
            for name, critic in controllers:
                path = output / f"{name}_{block:02d}.pt"
                if path.exists():
                    if not args.resume:
                        raise FileExistsError(path)
                    continue
                rejection = name.startswith("rejection_c")
                threshold = None
                if rejection:
                    threshold = float(name.removeprefix("rejection_c"))
                torch.cuda.synchronize()
                start = time.perf_counter()
                rows = []
                for call in range(8):
                    seed = 7100000 + 10 * (block * 8 + call)
                    if rejection:
                        row = rollout_rejection_angular_vel(
                            env,
                            policy,
                            spec,
                            snapshot,
                            obs,
                            critic,
                            seed,
                            threshold,
                            reward_spec=reward_spec,
                        )
                    else:
                        row = rollout_angular_vel(
                            env,
                            policy,
                            spec,
                            snapshot,
                            obs,
                            critic,
                            seed,
                            reward_spec=reward_spec,
                        )
                    rows.append(row)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                result = {
                    k: torch.stack(
                        [r[k].reshape(8, 128, *r[k].shape[1:]) for r in rows]
                    )
                    .flatten(0, 1)
                    .transpose(0, 1)
                    .contiguous()
                    for k in rows[0]
                }
                result.update(
                    seconds=seconds,
                    axes=["state128", "execution64", "optional_candidate8"],
                    seeds=[7100000 + 10 * (block * 8 + call) for call in range(8)],
                )
                torch.save(result, path)
                timing = {
                    "controller": name,
                    "block": block,
                    "seconds": seconds,
                    "projected_controller_seconds": seconds * 16,
                    "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                }
                print(json.dumps(timing), flush=True)
                if block == 0 and seconds * 16 > 720:
                    (output / "runtime_blocker.json").write_text(
                        json.dumps(timing, indent=2)
                    )
                    raise RuntimeError(
                        "12-minute/controller runtime gate exceeded; retained block, "
                        "approval required"
                    )
        if args.trace_steps > 0:
            if args.trace_names is None:
                traced = [("baseline", None), (args.artifact_names[0], artifacts[0])]
            else:
                by_name = dict(zip(args.artifact_names, artifacts, strict=True))
                missing = [name for name in args.trace_names if name not in by_name]
                if missing:
                    raise ValueError(f"unknown --trace_names: {missing}")
                traced = [("baseline", None)]
                traced += [(name, by_name[name]) for name in args.trace_names]
            for name, critic in traced:
                path = output / f"trace_{name}.pt"
                if path.exists():
                    if not args.resume:
                        raise FileExistsError(path)
                    continue
                torch.cuda.synchronize()
                start = time.perf_counter()
                row = rollout_angular_vel(
                    env,
                    policy,
                    spec,
                    snapshot,
                    obs,
                    critic,
                    7300000,
                    reward_spec=reward_spec,
                    horizon=args.trace_steps,
                    trace=True,
                )
                torch.cuda.synchronize()
                row.update(
                    seconds=time.perf_counter() - start,
                    steps=args.trace_steps,
                    control_dt=spec.control_dt,
                    axes=["lane8", "state128", "step"],
                )
                torch.save(row, path)
                print(
                    json.dumps(
                        {
                            "trace": name,
                            "steps": args.trace_steps,
                            "seconds": row["seconds"],
                        }
                    ),
                    flush=True,
                )
    finally:
        env._backend.close()


if __name__ == "__main__":
    main()
