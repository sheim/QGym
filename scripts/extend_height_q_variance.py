"""Random 16-state subset, eight parallel seed lanes, sixteen 64-return batches."""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

from evaluate_height_q_variance import fidelity, physical, run_continuation
from gym.utils.helpers import class_to_dict
from gym.utils.task_registry import task_registry
from learning.offline.contracts import replay_config_fingerprint, sha256_file
from learning.offline.evaluation import load_policy_from_artifact
from learning.offline.fixed_state_variance import (
    WarpInsertionSnapshot,
    install_paired_environment_noise,
    lane_source_ids,
)
from learning.offline.q_artifact import HeightQ


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--load_run", required=True)
    p.add_argument("--selection_seed", type=int, default=2026111064)
    p.add_argument("--new_seed", type=int, default=6400000)
    p.add_argument("--blocks", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if not 1 <= args.blocks <= 14:
        raise ValueError("blocks must be in [1,14]")
    import gym.envs  # noqa: F401

    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=args.resume)
    # Selection is made without inspecting any returns.
    indices = torch.randperm(
        128, generator=torch.Generator().manual_seed(args.selection_seed)
    )[:16]
    saved = torch.load(source / "insertion.pt", weights_only=True)
    old = torch.load(source / "raw_returns.pt", weights_only=True)
    old_seeds = torch.as_tensor(old["seeds"]).flatten().tolist()
    new_seeds = [args.new_seed + i * 10 for i in range(112)]
    if {s + j for s in old_seeds for j in range(3)} & {
        s + j for s in new_seeds for j in range(3)
    }:
        raise ValueError("old/new RNG stream collision")
    provenance = {
        "selection_seed": args.selection_seed,
        "state_indices": indices.tolist(),
        "source": str(source),
        "source_sha256": {
            name: sha256_file(source / name)
            for name in ("insertion.pt", "raw_returns.pt")
        },
        "new_call_seeds": new_seeds,
        "seed_identity": (
            "(call_seed, lane); independent vectorized RNG subsequences, "
            "lane-major 16-state groups; global env=seed, "
            "observation=seed+1, action=seed+2"
        ),
        "world_axes": ["lane8", "state16", "candidate8"],
        "repeat_order": "call-major then lane; 8 calls x 8 lanes per 64 batch",
        "old_seeds": old_seeds,
        "checkpoint_sha256": sha256_file(args.checkpoint),
    }
    if args.resume:
        if json.loads((output / "provenance.json").read_text()) != provenance:
            raise ValueError("resume provenance mismatch")
    else:
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    with (output / "commands.txt").open("a") as f:
        f.write(" ".join(sys.argv) + "\n")
    artifact = HeightQ(saved["artifact_paths"][0])
    policy, spec = load_policy_from_artifact(artifact, args.checkpoint, "cuda:0")
    if (
        spec.checkpoint_sha256 != provenance["checkpoint_sha256"]
        or spec.n_policy_steps != 1
    ):
        raise ValueError("policy identity mismatch")
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
        cfg.env.episode_length_s, 601 * cfg.control.ctrl_dt + 1
    )
    cfg.seed = saved["seed"]
    env = task_registry.make_env("go2trot", cfg, "cuda:0", True, "mujoco")
    try:
        original = WarpInsertionSnapshot(env)
        original.load_export(saved["snapshot"])
        original.restore()
        original.assert_restored()
        original_actions = saved["candidates"].flatten(0, 1).to(env.device)
        validation = fidelity(env, original_actions, original)
        torch.manual_seed(654321)
        env.set_states(spec.action_names, original_actions)
        env.step()
        reference = physical(env)
        ids = lane_source_ids(indices)
        snapshot = WarpInsertionSnapshot(env)
        snapshot.load_export(saved["snapshot"], ids)
        snapshot.restore()
        snapshot.assert_restored()
        actions = original_actions[ids.to(env.device)]
        torch.manual_seed(654321)
        env.set_states(spec.action_names, actions)
        env.step()
        errors = {
            k: float((physical(env)[k] - v[ids.to(env.device)]).abs().max())
            for k, v in reference.items()
        }
        validation["selected_lane_replication_error"] = errors
        if any(errors[k] > validation["limits"][k] for k in errors):
            raise RuntimeError(f"lane replication failed: {validation}")
        snapshot.restore()
        validation["lane_restore"] = fidelity(env, actions, snapshot)
        # Exact inventory audit includes flattened state, native model constants,
        # unconstrained native qpos, controller history and all episodic DR.
        validation["inventory_entries"] = len(snapshot.entries)
        for entry, (_, key, _, mapped) in zip(
            saved["snapshot"]["entries"], snapshot.entries
        ):
            expected = entry["value"].reshape(1024, -1)[ids].reshape(mapped.shape)
            if not torch.equal(mapped.cpu(), expected):
                raise RuntimeError(f"mapping audit failed: {key}")
        install_paired_environment_noise(env, 8)
        snapshot.restore()
        env._resample_commands(torch.ones(1024, dtype=torch.bool, device=env.device))
        command = env.commands.reshape(8, 16, 8, -1).clone()
        before = env.root_states.clone()
        env._push_robots()
        push = (env.root_states - before).reshape(8, 16, 8, -1)
        for name, value in (("commands", command), ("push", push)):
            if not torch.equal(value, value[:, :, :1].expand_as(value)) or torch.equal(
                value[0], value[1]
            ):
                raise RuntimeError(
                    f"candidate sharing/lane independence failed: {name}"
                )
        snapshot.restore()
        snapshot.assert_restored()
        validation["native_command_push_pairing"] = True
        (
            output / ("validation_resume.json" if args.resume else "validation.json")
        ).write_text(json.dumps(validation, indent=2))
        torch.save(
            {
                "snapshot": snapshot.export(),
                "state_indices": indices,
                "insertion_observations": saved["insertion_observations"].reshape(
                    128, 8, -1
                )[indices],
                "candidates": saved["candidates"][indices],
                "predictions": saved["predictions"][:, indices],
            },
            output / "insertion.pt",
        )
        print(
            json.dumps({"validation": validation, "state_indices": indices.tolist()}),
            flush=True,
        )
        for block in range(args.blocks):
            path = output / f"new_block_{block:02d}.pt"
            if path.exists():
                if not args.resume:
                    raise FileExistsError(path)
                continue
            start = time.perf_counter()
            rows = [
                run_continuation(
                    env, policy, spec, snapshot, actions, new_seeds[block * 8 + call]
                )
                for call in range(8)
            ]
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            result = {
                k: torch.stack([row[k].reshape(8, 16, 8) for row in rows]).reshape(
                    64, 16, 8
                )
                for k in rows[0]
            }
            result.update(
                seconds=seconds,
                seed_identities=[
                    (new_seeds[block * 8 + call], lane)
                    for call in range(8)
                    for lane in range(8)
                ],
            )
            torch.save(result, path)
            print(
                json.dumps(
                    {
                        "block": block,
                        "seconds": seconds,
                        "projected_new_seconds": seconds * 14,
                    }
                ),
                flush=True,
            )
            if block == 0 and seconds * 14 > 720:
                raise RuntimeError(
                    "runtime gate exceeds 12 minutes; retained first block, stopped"
                )
        if args.blocks == 14:
            blocks = [
                torch.load(output / f"new_block_{b:02d}.pt", weights_only=True)
                for b in range(14)
            ]
            raw = {
                k: torch.cat(
                    [
                        old[k].reshape(2, 64, 128, 8)[:, :, indices],
                        torch.stack([b[k] for b in blocks]),
                    ]
                )
                for k in ("returns", "first_rewards", "survived")
            }
            raw.update(
                axes=["batch64", "repeat64", "state16", "candidate8"],
                state_indices=indices,
                reused_seeds=old["seeds"],
                new_seed_identities=[b["seed_identities"] for b in blocks],
                new_block_seconds=[b["seconds"] for b in blocks],
            )
            torch.save(raw, output / "raw_returns.pt")
    finally:
        env._backend.close()


if __name__ == "__main__":
    main()
