"""Full threshold-controller sweep: rejection and closest-mean controllers.

Seven controllers share the original fair-100 height-Q artifact, the frozen
model_600 policy, the same 128 native snapshots and the same paired seeds:

* five stochastic rejection controllers (first of eight draws with Q >= c, else
  the max-Q draw), and
* two deterministic closest-to-mean controllers over the exact mean plus the
  same eight draws.

The plain and historical best-eight controllers are not rerun; their blocks are
reused from the persistent-intervention run. Each controller's first block is a
retained benchmark and is reused by the full run.
"""

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
from learning.offline.contracts import replay_config_fingerprint, sha256_file
from learning.offline.evaluation import load_policy_from_artifact
from learning.offline.fixed_state_variance import WarpInsertionSnapshot
from learning.offline.persistent_intervention import (
    rollout,
    rollout_mean,
    rollout_rejection,
    rollout_threshold,
    sample_rejection_candidates,
)
from learning.offline.q_artifact import HeightQ

CONTROLLERS = (
    ("reject_16", "rejection", -16.0),
    ("reject_4p866", "rejection", -4.866),
    ("reject_1p776", "rejection", -1.776),
    ("reject_0p956", "rejection", -0.956),
    ("reject_0p626", "rejection", -0.626),
    ("distance_4p866", "distance", -4.866),
    ("distance_1p776", "distance", -1.776),
)


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


@torch.no_grad()
def run_controller(
    env, policy, spec, snapshot, obs, critic, name, kind, threshold, seeds
):
    """Run eight calls for one block and return stacked [state, execution] tensors."""
    rows = []
    for call, seed in enumerate(seeds):
        if kind == "rejection":
            rows.append(
                rollout_rejection(
                    env,
                    policy,
                    spec,
                    snapshot,
                    obs,
                    critic,
                    seed,
                    threshold,
                )
            )
        elif kind == "distance":
            rows.append(
                rollout_threshold(
                    env,
                    policy,
                    spec,
                    snapshot,
                    obs,
                    critic,
                    seed,
                    threshold,
                )
            )
        else:
            raise ValueError(f"unknown controller kind {kind!r}")
    result = {}
    for key in rows[0]:
        stacked = torch.stack(
            [row[key].reshape(8, 128, *row[key].shape[1:]) for row in rows]
        )
        result[key] = stacked.flatten(0, 1).transpose(0, 1).contiguous()
    result.update(
        axes=["state128", "execution64", "optional_candidate"],
        seeds=list(seeds),
        controller=name,
        kind=kind,
        threshold=threshold,
    )
    return result


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--load_run", required=True)
    p.add_argument("--artifact", required=True)
    p.add_argument("--blocks", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--controllers",
        nargs="+",
        default=[name for name, _, _ in CONTROLLERS],
        help="Subset of controller names for smoke or a partial run.",
    )
    p.add_argument("--horizon", type=int, default=500)
    p.add_argument("--smoke_steps", type=int, default=8)
    p.add_argument("--max_projected_seconds", type=float, default=720.0)
    p.add_argument("--soft_target_seconds", type=float, default=480.0)
    args = p.parse_args()
    if not 1 <= args.blocks <= 16:
        raise ValueError("need 1..16 blocks (eight calls/block)")
    controllers = [row for row in CONTROLLERS if row[0] in args.controllers]
    if len(controllers) != len(args.controllers):
        raise ValueError("unknown controller name in --controllers")
    if len(set(args.controllers)) != len(args.controllers):
        raise ValueError("--controllers must be unique")
    if args.horizon <= 0 or args.smoke_steps <= 0:
        raise ValueError("horizons must be positive")
    import gym.envs  # noqa: F401

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=args.resume)
    source = Path(args.source) / "insertion.pt"
    saved = torch.load(source, weights_only=True)
    artifact = HeightQ(args.artifact)
    policy, spec = load_policy_from_artifact(artifact, args.checkpoint, "cuda:0")
    reward = artifact.manifest["reward_spec"]
    if (
        reward["reference_height_m"],
        reward["bound_m"],
        reward["inside_reward"],
        reward["outside_reward"],
        artifact.manifest["gamma"],
    ) != (0.32, 0.05, 0, -1, 0.99):
        raise ValueError("unexpected reward or gamma")
    checkpoint_sha = sha256_file(args.checkpoint)
    if (
        checkpoint_sha != spec.checkpoint_sha256
        or saved["checkpoint_sha256"] != spec.checkpoint_sha256
        or spec.n_policy_steps != 1
    ):
        raise ValueError("checkpoint/interval mismatch")
    artifact_sha = {
        name: sha256_file(Path(args.artifact) / name)
        for name in ("manifest.json", "q.pt")
    }
    provenance = {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": spec.checkpoint_sha256,
        "artifact": args.artifact,
        "artifact_sha256": artifact_sha,
        "reward_spec": reward,
        "gamma": artifact.manifest["gamma"],
        "controllers": [
            {"name": name, "kind": kind, "threshold": threshold}
            for name, kind, threshold in controllers
        ],
        "protocol": {
            "calls_per_block": 8,
            "lanes_per_call": 8,
            "target_executions_per_state": 1024,
            "horizon": args.horizon,
            "seed_identity": (
                "(7100000+10*call,lane,state); env=seed, obs=seed+1, action=seed+2"
            ),
            "world_axes": ["lane8", "state128"],
            "initial_observation": (
                "original noisy insertion observation, candidate0 world, "
                "replicated eight lanes"
            ),
            "rejection": (
                "eight stochastic draws in fixed order; first Q>=c; otherwise "
                "max-Q among eight; mean excluded"
            ),
            "distance": (
                "exact mean at index 0 plus the same eight draws; feasible Q>=c; "
                "minimum sum(((a-mu)/sigma)^2); otherwise max-Q among nine"
            ),
            "threshold_scope": "fixed for the whole rollout",
            "restore": (
                "native 82-field snapshot restore per call; no clamp, no prefix "
                "rerun, no timeout reset"
            ),
        },
        "runtime_model": {
            "provider": "opencode-go",
            "model": "deepseek-v4.1-flash",
            "note": "session configuration, not independently verifiable in-process",
        },
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    if args.resume:
        if json.loads((output / "provenance.json").read_text()) != provenance:
            raise ValueError("resume provenance mismatch")
    else:
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    with (output / "commands.txt").open("a") as f:
        f.write(
            f"{utc_now()} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen "
            + "python "
            + " ".join(sys.argv)
            + "\n"
        )
    (output / ("dirty_resume.txt" if args.resume else "dirty.txt")).write_text(
        subprocess.check_output(["git", "status", "--short"], text=True)
    )
    code_paths = [
        Path(__file__),
        Path("learning/offline/persistent_intervention.py"),
        Path("learning/offline/fixed_state_variance.py"),
        Path("learning/offline/q_artifact.py"),
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
        != artifact.manifest["dataset"]["config_fingerprint"]
    ):
        raise ValueError("environment fingerprint mismatch")
    cfg.env.episode_length_s = max(
        cfg.env.episode_length_s, (args.horizon + 101) * cfg.control.ctrl_dt + 1
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
        obs = saved["insertion_observations"][ids].to(env.device)
        diag, _ = sample_rejection_candidates(
            policy, obs, torch.Generator(device=env.device).manual_seed(99000011)
        )
        query = obs[:, None].expand(-1, 8, -1).flatten(0, 1)
        acts = diag.flatten(0, 1)
        cpu = artifact.predict(query.cpu(), acts.cpu())
        artifact.q.to(env.device)
        artifact.obs_mean = artifact.obs_mean.to(env.device)
        artifact.obs_std = artifact.obs_std.to(env.device)
        gpu = artifact.predict(query, acts)
        split = torch.cat(
            [artifact.predict(o, a) for o, a in zip(query.split(257), acts.split(257))]
        )
        torch.testing.assert_close(cpu, gpu.cpu(), atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(gpu, split, atol=2e-5, rtol=2e-5)
        validation["cpu_gpu"] = {
            "max_error": float((cpu - gpu.cpu()).abs().max()),
            "batch_error": float((gpu - split).abs().max()),
            "argmax_disagreements": int(
                (
                    cpu.reshape(-1, 8).argmax(-1) != gpu.cpu().reshape(-1, 8).argmax(-1)
                ).sum()
            ),
        }
        # Negative controls use the same seed and short horizon. Selector-level
        # infinities are exercised by unit tests; finite controls avoid any
        # accidental threshold-masking in the whole rollout path.
        plain = rollout(env, policy, spec, snapshot, obs, None, 99000021, horizon=5)
        permissive = rollout_rejection(
            env,
            policy,
            spec,
            snapshot,
            obs,
            artifact,
            99000021,
            -1e9,
            horizon=5,
        )
        for key in (
            "returns",
            "first_rewards",
            "survived",
            "selection_counts",
            "steps",
        ):
            torch.testing.assert_close(permissive[key], plain[key], atol=0, rtol=0)
        best8 = rollout(env, policy, spec, snapshot, obs, artifact, 99000021, horizon=5)
        restrictive = rollout_rejection(
            env,
            policy,
            spec,
            snapshot,
            obs,
            artifact,
            99000021,
            1e9,
            horizon=5,
        )
        for key in (
            "returns",
            "first_rewards",
            "survived",
            "selection_counts",
            "steps",
        ):
            torch.testing.assert_close(restrictive[key], best8[key], atol=0, rtol=0)
        mean = rollout_mean(env, policy, spec, snapshot, obs, 99000021, horizon=5)
        deterministic = rollout_threshold(
            env,
            policy,
            spec,
            snapshot,
            obs,
            artifact,
            99000021,
            -1e9,
            horizon=5,
        )
        for key in ("returns", "first_rewards", "survived", "steps"):
            torch.testing.assert_close(deterministic[key], mean[key], atol=0, rtol=0)
        validation["negative_controls_5steps"] = {
            "rejection_permissive_matches_plain": "exact",
            "rejection_restrictive_matches_best8": "exact",
            "distance_permissive_matches_mean": "exact",
        }
        # Actual CUDA controller smoke for every selected controller, not just
        # the nominal pilot path. It executes the real selection rule for a few
        # steps and checks liveness/accounting counters.
        smoke = {}
        for index, (name, kind, threshold) in enumerate(controllers):
            seed = 99000031 + index
            if kind == "rejection":
                result = rollout_rejection(
                    env,
                    policy,
                    spec,
                    snapshot,
                    obs,
                    artifact,
                    seed,
                    threshold,
                    horizon=args.smoke_steps,
                )
            else:
                result = rollout_threshold(
                    env,
                    policy,
                    spec,
                    snapshot,
                    obs,
                    artifact,
                    seed,
                    threshold,
                    horizon=args.smoke_steps,
                )
            steps = result["steps"].sum()
            if result["timeout_seen"].any():
                raise RuntimeError(f"{name} smoke saw a timeout")
            if not torch.isfinite(result["returns"]).all():
                raise RuntimeError(f"{name} smoke produced nonfinite returns")
            if result["selection_counts"].sum() != steps:
                raise RuntimeError(f"{name} smoke selection accounting failed")
            if kind == "rejection" and result["attempt_counts"].sum() != steps:
                raise RuntimeError(f"{name} smoke attempt accounting failed")
            if kind == "rejection" and result["fallback_sum"].sum() > steps:
                raise RuntimeError(f"{name} smoke fallback exceeded decisions")
            if kind == "distance" and result["mean_feasible_sum"].sum() > steps:
                raise RuntimeError(f"{name} smoke mean feasibility exceeded decisions")
            smoke[name] = {
                "kind": kind,
                "threshold": threshold,
                "horizon": args.smoke_steps,
                "steps": int(steps),
                "mean_return": float(result["returns"].mean()),
                "fallback_fraction": float(result["fallback_sum"].sum() / steps),
                "first_draw": float(result["selection_counts"][:, 0].sum() / steps),
                "terminations": int((~result["survived"]).sum()),
            }
        validation["controller_smoke"] = smoke
        validation["inventory_entries"] = len(snapshot.entries)
        (
            output / ("validation_resume.json" if args.resume else "validation.json")
        ).write_text(json.dumps(validation, indent=2))
        print(json.dumps({"validation": validation}), flush=True)

        seeds = [
            7100000 + 10 * (block * 8 + call)
            for block in range(16)
            for call in range(8)
        ]

        def write_controller_block(name, kind, threshold, block):
            block_seeds = seeds[block * 8 : block * 8 + 8]
            path = output / f"{name}_{block:02d}.pt"
            if path.exists():
                if not args.resume:
                    raise FileExistsError(path)
                return None
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = run_controller(
                env,
                policy,
                spec,
                snapshot,
                obs,
                artifact,
                name,
                kind,
                threshold,
                block_seeds,
            )
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            result["seconds"] = seconds
            torch.save(result, path)
            timing = {
                "timestamp": utc_now(),
                "controller": name,
                "kind": kind,
                "threshold": threshold,
                "block": block,
                "seconds": seconds,
                "projected_controller_seconds": seconds * 16,
                "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
            with (output / "timings.jsonl").open("a") as f:
                f.write(json.dumps(timing) + "\n")
            print(json.dumps(timing), flush=True)
            if block == 0 and seconds > args.soft_target_seconds / 16:
                print(
                    json.dumps(
                        {
                            "warning": "above eight-minute soft target",
                            "controller": name,
                            "projected_controller_seconds": seconds * 16,
                        }
                    ),
                    flush=True,
                )
            return seconds

        def write_block(block):
            for name, kind, threshold in controllers:
                write_controller_block(name, kind, threshold, block)

        if not args.resume:
            benchmark = {}
            for name, kind, threshold in controllers:
                benchmark[name] = write_controller_block(name, kind, threshold, 0)
            (output / "benchmark.json").write_text(json.dumps(benchmark, indent=2))
            blocked = {
                name: {"seconds": seconds, "projected": seconds * 16}
                for name, seconds in benchmark.items()
                if seconds * 16 > args.max_projected_seconds
            }
            if blocked:
                (output / "runtime_blocker.json").write_text(
                    json.dumps(blocked, indent=2)
                )
                raise RuntimeError(
                    "12-minute/controller runtime gate exceeded; benchmarks "
                    "retained, full run not launched"
                )
        for block in range(1 if not args.resume else 0, args.blocks):
            write_block(block)
    finally:
        env._backend.close()


if __name__ == "__main__":
    main()
