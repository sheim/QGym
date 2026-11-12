"""H500 paired original-PPO reward versus original height-Q tradeoff.

Runs the plain candidate-0 controller and the original fair-100 height-Q
max-of-eight controller on the same 128 fixed insertion states, paired seeds and
1024 Warp worlds. Both objectives are recorded on each execution.
"""

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from tensordict import TensorDict

from evaluate_height_q_variance import fidelity, physical
from gym.utils.helpers import class_to_dict
from gym.utils.task_registry import task_registry
from learning.offline.contracts import replay_config_fingerprint, sha256_file
from learning.offline.evaluation import load_policy_from_artifact
from learning.offline.fixed_state_variance import WarpInsertionSnapshot
from learning.offline.persistent_intervention import rollout
from learning.offline.q_artifact import HeightQ
from learning.offline.reward_tradeoff import rollout_tradeoff
from learning.offline.training_reward import TrainingRewardModel, nonzero_weights
from learning.runners.on_policy_runner import OnPolicyRunner
from learning.utils import remove_zero_weighted_rewards

EXPECTED_WEIGHTS = {
    "tracking_lin_vel": 4.0,
    "tracking_ang_vel": 4.0,
    "ang_vel_xy": 0.01,
    "orientation": 1.0,
    "torques": 5.0e-6,
    "min_base_height": 0.5,
    "action_rate": 0.1,
    "action_rate2": 0.05,
    "trot_support": 3.0,
    "swing_contact": 3.0,
    "reference_traj": 3.0,
}
EXPECTED_TERMINATION = {"termination": 0.01}
SEED_BASE = 7_100_000
TRAINING_EPISODE_STEPS = 250


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--load_run", required=True)
    p.add_argument("--artifact", required=True)
    p.add_argument("--blocks", type=int, default=16)
    p.add_argument("--resume", action="store_true")
    return p


def reward_parity(env, train_cfg, reward_model):
    """Compare the shared dispatcher with the real runner method on this step."""
    runner = object.__new__(OnPolicyRunner)
    runner.env = env
    runner.device = env.device
    runner.setup_reward_functions()
    critic_cfg = copy.deepcopy(class_to_dict(train_cfg)["critic"])
    remove_zero_weighted_rewards(critic_cfg["reward"]["weights"])
    remove_zero_weighted_rewards(critic_cfg["reward"]["termination_weight"])
    runner.critic_cfg = critic_cfg
    rewards = TensorDict({}, batch_size=(1, env.num_envs), device=env.device)
    keys = list(critic_cfg["reward"]["termination_weight"]) + list(
        critic_cfg["reward"]["weights"]
    )
    for key in keys:
        rewards[key] = torch.zeros(1, env.num_envs, device=env.device)
    runner.update_rewards_dict(rewards, 0)
    terms = reward_model.step_terms(env)
    if set(terms) != set(keys):
        raise RuntimeError(
            f"dispatch key mismatch: helper={sorted(terms)} runner={sorted(keys)}"
        )
    for name in keys:
        if not torch.equal(terms[name], rewards[0][name]):
            raise RuntimeError(
                f"reward parity mismatch for {name}: "
                f"{float((terms[name] - rewards[0][name]).abs().max())}"
            )
    total = reward_model.total(terms)
    if not torch.isfinite(total).all():
        raise RuntimeError("nonfinite parity reward")
    return {
        "keys": sorted(keys),
        "exact_term_parity": True,
        "terminal_count": int(env.terminated.sum()),
        "total_mean": float(total.mean()),
        "total_min": float(total.min()),
        "total_max": float(total.max()),
    }


def stack_block(rows):
    """[call, world] results -> [state, execution] storage axes."""
    scalar = {}
    for key in rows[0]:
        if key in ("term_names", "actions"):
            continue
        value = rows[0][key]
        if not isinstance(value, torch.Tensor):
            continue
        if key == "original_terms":
            stacked = torch.stack([row[key].reshape(8, 128, -1) for row in rows])
            scalar[key] = stacked.flatten(0, 1).transpose(0, 1).contiguous()
        elif value.ndim == 1:
            scalar[key] = (
                torch.stack([row[key].reshape(8, 128) for row in rows])
                .flatten(0, 1)
                .transpose(0, 1)
                .contiguous()
            )
        else:
            scalar[key] = (
                torch.stack(
                    [row[key].reshape(8, 128, *value.shape[1:]) for row in rows]
                )
                .flatten(0, 1)
                .transpose(0, 1)
                .contiguous()
            )
    scalar["term_names"] = rows[0]["term_names"]
    if not torch.equal(scalar["original_terms"].sum(-1), scalar["original_returns"]):
        raise RuntimeError("stored per-term sums do not equal stored original return")
    if scalar["timeout_seen"].any():
        raise RuntimeError("timeout in stored block")
    return scalar


def block_path(output, name, block):
    return output / f"{name}_{block:02d}.pt"


@torch.no_grad()
def main():
    args = parser().parse_args()
    if not 1 <= args.blocks <= 16:
        raise ValueError("blocks must be in [1, 16]")
    import gym.envs  # noqa: F401

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=args.resume)
    source = Path(args.source) / "insertion.pt"
    saved = torch.load(source, weights_only=True)
    artifact = HeightQ(args.artifact)
    policy, spec = load_policy_from_artifact(artifact, args.checkpoint, "cuda:0")
    if len(saved["snapshot"]["entries"]) != 82:
        raise ValueError("source snapshot inventory is not 82 entries")
    reward_spec = artifact.manifest["reward_spec"]
    if (
        reward_spec["reference_height_m"],
        reward_spec["bound_m"],
        reward_spec["inside_reward"],
        reward_spec["outside_reward"],
    ) != (0.32, 0.05, 0, -1):
        raise ValueError("height reward spec mismatch")
    if (
        sha256_file(args.checkpoint) != spec.checkpoint_sha256
        or saved["checkpoint_sha256"] != spec.checkpoint_sha256
        or spec.n_policy_steps != 1
    ):
        raise ValueError("checkpoint or policy interval mismatch")

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
    fingerprint = replay_config_fingerprint(class_to_dict(cfg))
    if fingerprint != artifact.manifest["dataset"]["config_fingerprint"]:
        raise ValueError("environment fingerprint differs from original critic")
    reward_cfg = class_to_dict(train.critic.reward)
    weights = dict(reward_cfg["weights"])
    termination_weights = dict(reward_cfg["termination_weight"])
    gamma = float(train.algorithm.gamma)
    effective_weights = nonzero_weights(weights)
    effective_termination = nonzero_weights(termination_weights)
    if effective_weights != EXPECTED_WEIGHTS:
        raise ValueError(f"training reward weights changed: {effective_weights}")
    if effective_termination != EXPECTED_TERMINATION:
        raise ValueError(
            f"training termination weights changed: {effective_termination}"
        )
    if gamma != 0.99:
        raise ValueError(f"training gamma changed: {gamma}")
    if artifact.manifest["gamma"] != gamma:
        raise ValueError("height-Q gamma differs from training gamma")
    n_policy_steps = int((1.0 / cfg.control.ctrl_dt) / train.actor.frequency)
    if n_policy_steps != 1:
        raise ValueError(f"n_policy_steps must be 1, got {n_policy_steps}")
    cfg.env.episode_length_s = max(
        cfg.env.episode_length_s, 601 * cfg.control.ctrl_dt + 1
    )
    cfg.seed = saved["seed"]

    code_paths = [
        Path(__file__),
        Path("learning/offline/reward_tradeoff.py"),
        Path("learning/offline/training_reward.py"),
        Path("learning/offline/tradeoff_analysis.py"),
        Path("learning/offline/fixed_state_variance.py"),
        Path("learning/offline/persistent_intervention.py"),
        Path("learning/offline/q_artifact.py"),
        Path("learning/offline/evaluation.py"),
        Path("learning/runners/BaseRunner.py"),
    ]
    code = {str(path): sha256_file(path) for path in code_paths}
    provenance = {
        "experiment": "original_reward_tradeoff",
        "source": str(source),
        "source_sha256": sha256_file(source),
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": spec.checkpoint_sha256,
        "artifact": args.artifact,
        "artifact_sha256": {
            "manifest.json": sha256_file(Path(args.artifact) / "manifest.json"),
            "q.pt": sha256_file(Path(args.artifact) / "q.pt"),
        },
        "height_reward_spec": reward_spec,
        "height_gamma": float(artifact.manifest["gamma"]),
        "training_gamma": gamma,
        "training_dt": float(cfg.control.ctrl_dt),
        "control_frequency": float(cfg.control.ctrl_frequency),
        "desired_sim_frequency": float(cfg.control.desired_sim_frequency),
        "decimation": int(cfg.control.decimation),
        "actor_frequency": float(train.actor.frequency),
        "n_policy_steps": n_policy_steps,
        "rollout_size": int(train.algorithm.rollout_size),
        "batch_size": int(train.algorithm.batch_size),
        "training_episode_steps": TRAINING_EPISODE_STEPS,
        "horizon": 500,
        "training_reward": {
            "weights": effective_weights,
            "termination_weights": effective_termination,
            "zero_weight_keys_removed": sorted(set(weights) - set(effective_weights)),
            "dispatch": "mask * ((weight * dt) * reward_function())",
            "ordinary_mask": "~env.terminated",
            "termination_mask": "env.terminated",
            "after_terminal": "zero, no absorbing tail",
        },
        "env_config_fingerprint": fingerprint,
        "env_config": class_to_dict(cfg),
        "training_config": class_to_dict(train),
        "seed_base": SEED_BASE,
        "seed_identity": (
            "(7100000+10*call,lane,state); env=seed, obs=seed+1, action=seed+2"
        ),
        "world_axes": ["lane8", "state128"],
        "candidate_axis": (
            "eight shared IID draws; plain=candidate0, original=Q argmax"
        ),
        "initial_observation": (
            "saved original noisy insertion observation, replicated lanes"
        ),
        "episode_length_s": float(cfg.env.episode_length_s),
        "max_episode_length": int(
            torch.ceil(
                torch.tensor(cfg.env.episode_length_s / cfg.control.ctrl_dt)
            ).item()
        ),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "code_sha256": code,
    }
    if args.resume:
        if json.loads((output / "provenance.json").read_text()) != provenance:
            raise ValueError("resume provenance mismatch")
    else:
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    with (output / "commands.txt").open("a") as handle:
        handle.write(
            "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen python "
            + " ".join(sys.argv)
            + "\n"
        )
    (output / ("dirty_resume.txt" if args.resume else "dirty.txt")).write_text(
        subprocess.check_output(["git", "status", "--short"], text=True)
    )
    (output / ("code_resume.json" if args.resume else "code.json")).write_text(
        json.dumps(code, indent=2)
    )

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
            key: float((physical(env)[key] - value[ids.to(env.device)]).abs().max())
            for key, value in reference.items()
        }
        if any(
            errors[key] > validation["source_fidelity"]["limits"][key] for key in errors
        ):
            raise RuntimeError(f"replication failed {errors}")
        validation["replication"] = errors
        snapshot.restore()
        validation["lane_fidelity"] = fidelity(
            env, actions[ids.to(env.device)], snapshot
        )
        obs = saved["insertion_observations"][ids].to(env.device)

        validation["cpu_gpu"] = []
        query = obs[:, None].expand(-1, 8, -1).flatten(0, 1)
        from learning.offline.persistent_intervention import sample_candidates

        candidates = sample_candidates(
            policy, obs, torch.Generator(device=env.device).manual_seed(99000011)
        )
        acts = candidates.flatten(0, 1)
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

        reward_model = TrainingRewardModel(
            env,
            weights,
            termination_weights,
            gamma=gamma,
        )
        # One real post-step comparison against OnPolicyRunner itself.
        snapshot.restore()
        torch.manual_seed(123456)
        env.set_states(spec.action_names, actions[ids.to(env.device)])
        env.step()
        validation["reward_parity"] = reward_parity(env, train, reward_model)
        validation["inventory_entries"] = len(snapshot.entries)

        # Existing persistent-intervention rollout does not call the training
        # reward functions. Exact H5 agreement therefore also checks that those
        # read-only calls leave the trajectory untouched.
        validation["height_parity"] = {}
        validation["negative_control_differences"] = {}
        for name, critic in (("plain", None), ("original", artifact)):
            theirs = rollout(
                env,
                policy,
                spec,
                snapshot,
                obs,
                critic,
                99000021,
                horizon=5,
            )
            ours = rollout_tradeoff(
                env,
                policy,
                spec,
                snapshot,
                obs,
                critic,
                99000021,
                reward_model=reward_model,
                reference_height=float(reward_spec["reference_height_m"]),
                height_bound=float(reward_spec["bound_m"]),
                horizon=5,
                trace=True,
            )
            checks = {
                "height_returns": (theirs["returns"], ours["height_returns"]),
                "steps": (theirs["steps"], ours["steps"]),
                "survived": (theirs["survived"], ours["survived"]),
                "selection_counts": (
                    theirs["selection_counts"],
                    ours["selection_counts"],
                ),
                "first_height_reward": (
                    theirs["first_rewards"],
                    ours["first_height_reward"],
                ),
            }
            for key, (left, right) in checks.items():
                if not torch.equal(left, right):
                    raise RuntimeError(f"height parity mismatch for {key}")
            torch.testing.assert_close(
                theirs["q_gain_sum"], ours["q_gain_sum"], atol=1e-2, rtol=1e-4
            )
            again = rollout_tradeoff(
                env,
                policy,
                spec,
                snapshot,
                obs,
                critic,
                99000021,
                reward_model=reward_model,
                reference_height=float(reward_spec["reference_height_m"]),
                height_bound=float(reward_spec["bound_m"]),
                horizon=5,
                trace=True,
            )
            for key, value in ours.items():
                if not isinstance(value, torch.Tensor):
                    continue
                if value.dtype == torch.bool:
                    difference = float((value != again[key]).sum())
                else:
                    difference = float((value - again[key]).abs().max())
                if key in {
                    "selection_counts",
                    "steps",
                    "survived",
                    "first_terminated",
                    "termination_step",
                    "timeout_seen",
                }:
                    if not torch.equal(value, again[key]):
                        raise RuntimeError(
                            f"negative control discrete mismatch for {key}: "
                            f"maxdiff={difference}"
                        )
                elif key == "actions":
                    # Restore does not preserve solver scratch, so a short
                    # repeated rollout can differ numerically even with an
                    # identical controller. This is a side-effect check, not a
                    # bitwise trajectory claim.
                    torch.testing.assert_close(value, again[key], atol=0.1, rtol=1e-4)
                else:
                    torch.testing.assert_close(value, again[key], atol=5e-3, rtol=1e-4)
                validation["negative_control_differences"][key] = difference
            validation["height_parity"][name] = {
                "exact_height_return_steps_selection_gain": True,
                "repeat_within_numeric_tolerance": True,
            }
        (
            output / ("validation_resume.json" if args.resume else "validation.json")
        ).write_text(json.dumps(validation, indent=2))
        print(json.dumps({"validation": validation}), flush=True)

        controllers = [("plain", None), ("original", artifact)]
        order = [(name, 0, critic) for name, critic in controllers]
        order += [
            (name, block, critic)
            for block in range(1, args.blocks)
            for name, critic in controllers
        ]
        for name, block, critic in order:
            path = block_path(output, name, block)
            if path.exists():
                if not args.resume:
                    raise FileExistsError(path)
                continue
            torch.cuda.synchronize()
            start = time.perf_counter()
            rows = []
            seeds = []
            for call in range(8):
                seed = SEED_BASE + 10 * (block * 8 + call)
                rows.append(
                    rollout_tradeoff(
                        env,
                        policy,
                        spec,
                        snapshot,
                        obs,
                        critic,
                        seed,
                        reward_model=reward_model,
                        reference_height=float(reward_spec["reference_height_m"]),
                        height_bound=float(reward_spec["bound_m"]),
                        horizon=500,
                    )
                )
                seeds.append(seed)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            result = stack_block(rows)
            result.update(
                {
                    "axes": ["state128", "execution64", "optional_candidate8"],
                    "seeds": seeds,
                    "block": block,
                    "controller": name,
                    "seconds": seconds,
                    "provenance_sha256": sha256_file(output / "provenance.json"),
                }
            )
            torch.save(result, path)
            timing = {
                "controller": name,
                "block": block,
                "seconds": seconds,
                "projected_controller_seconds": seconds * 16,
                "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
            with (output / "timings.jsonl").open("a") as handle:
                handle.write(json.dumps(timing) + "\n")
            print(json.dumps(timing), flush=True)
            if block == 0 and seconds * 16 > 720:
                (output / "runtime_blocker.json").write_text(
                    json.dumps(timing, indent=2)
                )
                raise RuntimeError(
                    f"{name} projected controller time exceeds 12 minutes; "
                    "pilot block retained, approval required"
                )
        counts = {
            name: len(list(output.glob(f"{name}_*.pt"))) for name, _ in controllers
        }
        completion = {
            "requested_blocks": args.blocks,
            "complete_blocks": counts,
            "expected_blocks": {name: args.blocks for name, _ in controllers},
            "all_requested_present": all(
                counts[name] == args.blocks for name, _ in controllers
            ),
            "timings": [
                json.loads(line)
                for line in (output / "timings.jsonl").read_text().splitlines()
            ],
            "height_reward_spec": reward_spec,
            "height_gamma": float(artifact.manifest["gamma"]),
            "training_gamma": gamma,
            "horizon": 500,
            "training_episode_steps": TRAINING_EPISODE_STEPS,
        }
        (output / "completion.json").write_text(json.dumps(completion, indent=2))
        print(json.dumps({"completion": completion}), flush=True)
    finally:
        env._backend.close()


if __name__ == "__main__":
    main()
