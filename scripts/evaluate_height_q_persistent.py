"""Persistent sample-eight intervention with retained benchmark/resume chunks."""

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
from learning.offline.persistent_intervention import rollout, sample_candidates
from learning.offline.q_artifact import HeightQ


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--load_run", required=True)
    p.add_argument("--artifacts", nargs="+", required=True)
    p.add_argument(
        "--artifact_names",
        nargs="+",
        default=["original", "evaldist"],
        help="Controller names aligned with --artifacts.",
    )
    p.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Do not rerun the plain baseline; reuse blocks from another run.",
    )
    p.add_argument("--blocks", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if not 1 <= args.blocks <= 16:
        raise ValueError("need 1..16 blocks (eight calls/block)")
    if len(args.artifact_names) != len(args.artifacts):
        raise ValueError("--artifact_names must match --artifacts")
    if len(set(args.artifact_names)) != len(args.artifact_names):
        raise ValueError("--artifact_names must be unique")
    if "baseline" in args.artifact_names:
        raise ValueError("'baseline' is reserved for the plain controller")
    import gym.envs  # noqa: F401

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=args.resume)
    source = Path(args.source) / "insertion.pt"
    saved = torch.load(source, weights_only=True)
    artifacts = [HeightQ(path) for path in args.artifacts]
    policy, spec = load_policy_from_artifact(artifacts[0], args.checkpoint, "cuda:0")
    for q in artifacts:
        if q.manifest["policy_spec"] != artifacts[0].manifest["policy_spec"]:
            raise ValueError("policy mismatch")
        r = q.manifest["reward_spec"]
        if (
            r["reference_height_m"],
            r["bound_m"],
            r["inside_reward"],
            r["outside_reward"],
            q.manifest["gamma"],
        ) != (0.32, 0.05, 0, -1, 0.99):
            raise ValueError("reward mismatch")
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
        f.write(
            "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen python "
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
        != artifacts[0].manifest["dataset"]["config_fingerprint"]
    ):
        raise ValueError("environment fingerprint mismatch")
    cfg.env.episode_length_s = max(
        cfg.env.episode_length_s, 601 * cfg.control.ctrl_dt + 1
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
        # Identical controller negative control is short: Warp roundoff may
        # diverge at long horizons even with exact state/RNG restoration.
        a = rollout(env, policy, spec, snapshot, obs, None, 99000021, horizon=5)
        b = rollout(env, policy, spec, snapshot, obs, None, 99000021, horizon=5)
        for k in ("returns", "first_rewards", "survived", "selection_counts", "steps"):
            torch.testing.assert_close(a[k], b[k], atol=0, rtol=0)
        validation["negative_control_5steps"] = (
            "exact rewards, returns, selection, masks"
        )
        validation["inventory_entries"] = len(snapshot.entries)
        (
            output / ("validation_resume.json" if args.resume else "validation.json")
        ).write_text(json.dumps(validation, indent=2))
        print(json.dumps({"validation": validation}), flush=True)
        controllers = list(zip(args.artifact_names, artifacts, strict=True))
        if not args.skip_baseline:
            controllers.insert(0, ("baseline", None))
        for block in range(args.blocks):
            for name, critic in controllers:
                path = output / f"{name}_{block:02d}.pt"
                if path.exists():
                    if not args.resume:
                        raise FileExistsError(path)
                    continue
                torch.cuda.synchronize()
                start = time.perf_counter()
                rows = [
                    rollout(
                        env,
                        policy,
                        spec,
                        snapshot,
                        obs,
                        critic,
                        7100000 + 10 * (block * 8 + call),
                    )
                    for call in range(8)
                ]
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
    finally:
        env._backend.close()


if __name__ == "__main__":
    main()
