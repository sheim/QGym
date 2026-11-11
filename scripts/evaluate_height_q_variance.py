"""Fixed k=100 states, paired candidates, repeated 16-return batch estimators."""

import argparse
import copy
import itertools
import json
import time
from pathlib import Path

import torch

from gym.utils.helpers import class_to_dict
from gym.utils.task_registry import task_registry
from learning.offline.contracts import (
    replay_config_fingerprint,
    sample_observation_noise,
    sha256_file,
)
from learning.offline.evaluation import _policy_observation, load_policy_from_artifact
from learning.offline.fixed_state_variance import (
    WarpInsertionSnapshot,
    accumulate_return,
    expand_states,
    install_paired_environment_noise,
    summarize_returns,
)
from learning.offline.q_artifact import HeightQ


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts", nargs=2, required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--load_run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=240_000)
    p.add_argument(
        "--batches", type=int, default=8, help="Total batches, including saved"
    )
    p.add_argument("--resume", action="store_true")
    return p


def physical(env):
    return {
        k: getattr(env, k).clone()
        for k in (
            "root_states",
            "dof_pos",
            "dof_vel",
            "base_ang_vel",
            "contact_forces",
        )
    }


def fidelity(env, actions, snapshot):
    """Compare uninterrupted next step and two restored next steps."""
    results = []
    for repetition in range(3):
        if repetition:
            snapshot.restore()
            snapshot.assert_restored()
        torch.manual_seed(123456)
        env.set_states(("dof_pos_target",), actions)
        env.step()
        results.append(physical(env))
    errors = {
        key: max(float((results[i][key] - results[0][key]).abs().max()) for i in (1, 2))
        for key in results[0]
    }
    # One-step numerical fidelity, not long-horizon bitwise identity. These
    # bounds are fixed before the run, and substantially below prior clamp drift.
    limits = {
        "root_states": 2e-3,
        "dof_pos": 1e-4,
        "dof_vel": 1e-2,
        "base_ang_vel": 2e-3,
        "contact_forces": 1.0,
    }
    snapshot.restore()
    if any(errors[key] > limits[key] for key in limits):
        raise RuntimeError(f"restore fidelity gate failed: {errors}; limits={limits}")
    return {"max_absolute_error": errors, "limits": limits}


def run_continuation(env, policy, spec, snapshot, actions, seed):
    snapshot.restore()
    snapshot.assert_restored()
    torch.manual_seed(seed)
    obs_rng = torch.Generator(device=env.device).manual_seed(seed + 1)
    act_rng = torch.Generator(device=env.device).manual_seed(seed + 2)
    alive = torch.ones(1024, dtype=torch.bool, device=env.device)
    total = torch.zeros(1024, dtype=torch.float64, device=env.device)
    first_reward = None
    env.set_states(spec.action_names, actions)
    for step in range(500):
        env.step()
        reward = -((env.base_height[:, 0] - 0.32).abs() > 0.05).double()
        if step == 0:
            first_reward = reward.clone()
        total, alive = accumulate_return(
            total, alive, reward, env.terminated, step, 0.99
        )
        if step != 499:
            noise = sample_observation_noise(
                spec.obs_names,
                spec.obs_dims,
                spec.noise,
                spec.scales,
                128,
                env.device,
                obs_rng,
            )
            obs = env.get_states(spec.obs_names) + expand_states(noise, 8)
            mean = policy.mean_actions(obs)
            eps = torch.randn(
                128, spec.action_dim, device=env.device, generator=act_rng
            )
            next_actions = mean + policy.actor.std.detach() * expand_states(eps, 8)
            env.set_states(spec.action_names, next_actions)
    if not torch.isfinite(total).all():
        raise RuntimeError("nonfinite returns")
    return {
        "returns": total.reshape(128, 8).cpu(),
        "first_rewards": first_reward.reshape(128, 8).cpu(),
        "survived": alive.reshape(128, 8).cpu(),
    }


@torch.no_grad()
def main():
    args = parser().parse_args()
    if not 1 <= args.batches <= 8:
        raise ValueError("batches must be in [1,8]")
    import gym.envs  # noqa: F401

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=args.resume)
    artifacts = [HeightQ(path) for path in args.artifacts]
    policy, spec = load_policy_from_artifact(artifacts[0], args.checkpoint, args.device)
    for artifact in artifacts:
        if artifact.manifest["policy_spec"] != artifacts[0].manifest["policy_spec"]:
            raise ValueError("critics have different policy contracts")
        r = artifact.manifest["reward_spec"]
        if (
            r["reference_height_m"],
            r["bound_m"],
            r["inside_reward"],
            r["outside_reward"],
            artifact.manifest["gamma"],
        ) != (0.32, 0.05, 0, -1, 0.99):
            raise ValueError("critic reward/discount mismatch")
    if (
        sha256_file(args.checkpoint) != spec.checkpoint_sha256
        or spec.n_policy_steps != 1
    ):
        raise ValueError("policy identity or interval mismatch")
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
    fingerprint = replay_config_fingerprint(class_to_dict(cfg))
    if fingerprint != artifacts[0].manifest["dataset"]["config_fingerprint"]:
        raise ValueError("environment fingerprint differs from original critic")
    cfg.env.episode_length_s = max(
        cfg.env.episode_length_s, 601 * cfg.control.ctrl_dt + 1
    )
    cfg.seed = args.seed
    env = task_registry.make_env("go2trot", cfg, args.device, True, "mujoco")
    try:
        if args.resume:
            saved = torch.load(output / "insertion.pt", weights_only=True)
            if (
                saved["seed"] != args.seed
                or saved["checkpoint_sha256"] != spec.checkpoint_sha256
            ):
                raise ValueError("resume identity mismatch")
            snapshot = WarpInsertionSnapshot(env)
            exported = saved["snapshot"]
            if len(snapshot.entries) != len(exported["entries"]):
                raise ValueError("snapshot inventory changed")
            restored = []
            for (owner, key, native, value), entry in zip(
                snapshot.entries, exported["entries"]
            ):
                if (type(owner).__name__, key, native) != (
                    entry["owner"],
                    entry["field"],
                    entry["native"],
                ):
                    raise ValueError("snapshot inventory changed")
                restored.append((owner, key, native, entry["value"].to(value.device)))
            snapshot.entries = restored
            snapshot.counter = exported["common_step_counter"]
            snapshot.dr_rng = exported["dr_rng"]
            snapshot.restore()
            snapshot.assert_restored()
        else:
            prefix_rng = torch.Generator(device=env.device).manual_seed(args.seed + 3)
            action_rng = torch.Generator(device=env.device).manual_seed(args.seed + 4)
            alive = ~env.terminated.clone()
            for _ in range(100):
                obs = _policy_observation(
                    env,
                    spec.obs_names,
                    spec.obs_dims,
                    spec.noise,
                    spec.scales,
                    prefix_rng,
                )
                env.set_states(
                    spec.action_names, policy.sample_actions(obs, action_rng)
                )
                env.step()
                alive &= ~env.terminated
            selected = alive.nonzero().flatten()[:128]
            if selected.numel() != 128:
                raise RuntimeError("fewer than 128 surviving insertion states")
            native = WarpInsertionSnapshot(env)
            obs = _policy_observation(
                env, spec.obs_names, spec.obs_dims, spec.noise, spec.scales, prefix_rng
            )
            mean = policy.mean_actions(obs)
            validation = fidelity(env, mean, native)
            snapshot = WarpInsertionSnapshot(env, expand_states(selected, 8))
            snapshot.restore()
            snapshot.assert_restored()
            # Initial noisy observation and candidate set are frozen once.
            insertion_obs = expand_states(obs[selected], 8)
            means = mean[selected]
            gen = torch.Generator(device=env.device).manual_seed(args.seed + 5)
            eps = torch.randn(128, 8, spec.action_dim, device=env.device, generator=gen)
            eps[:, 0] = 0
            candidates = means[:, None] + policy.actor.std.detach() * eps
            # Replicated next-state check against source uninterrupted step.
            native.restore()
            env.set_states(spec.action_names, mean)
            torch.manual_seed(654321)
            env.step()
            reference = physical(env)
            snapshot.restore()
            torch.manual_seed(654321)
            env.set_states(spec.action_names, expand_states(means, 8))
            env.step()
            replication_errors = {
                k: float((physical(env)[k] - expand_states(v[selected], 8)).abs().max())
                for k, v in reference.items()
            }
            validation["replication_max_absolute_error"] = replication_errors
            if any(
                replication_errors[k] > validation["limits"][k]
                for k in replication_errors
            ):
                raise RuntimeError(f"replication fidelity gate failed: {validation}")
            snapshot.restore()
            predictions = torch.stack(
                [
                    q.predict(insertion_obs, candidates.flatten(0, 1))
                    .cpu()
                    .reshape(128, 8)
                    for q in artifacts
                ]
            )
            saved = {
                "snapshot": snapshot.export(),
                "insertion_observations": insertion_obs.cpu(),
                "candidates": candidates.cpu(),
                "predictions": predictions,
                "prefix_alive": alive.cpu(),
                "selected_world_ids": selected.cpu(),
                "seed": args.seed,
                "checkpoint_sha256": spec.checkpoint_sha256,
                "validation": validation,
                "env_config": json.loads(json.dumps(class_to_dict(cfg))),
                "artifact_paths": args.artifacts,
            }
            torch.save(saved, output / "insertion.pt")
            (output / "validation.json").write_text(json.dumps(validation, indent=2))
            print(json.dumps({"validation": validation}), flush=True)
        install_paired_environment_noise(env, 8)
        actions = saved["candidates"].to(env.device).flatten(0, 1)
        for batch in range(args.batches):
            path = output / f"batch_{batch}.pt"
            if path.exists():
                if not args.resume:
                    raise FileExistsError(path)
                continue
            start = time.perf_counter()
            rows = []
            seeds = []
            for repeat in range(16):
                seed = args.seed + 1_000_000 + (batch * 16 + repeat) * 10
                rows.append(
                    run_continuation(env, policy, spec, snapshot, actions, seed)
                )
                seeds.append(seed)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            result = {key: torch.stack([row[key] for row in rows]) for key in rows[0]}
            result.update(
                {
                    "axes": ["repeat", "state", "candidate"],
                    "seeds": seeds,
                    "batch": batch,
                    "seconds": seconds,
                }
            )
            torch.save(result, path)
            print(
                json.dumps({"batch": batch, "seconds": seconds, "path": str(path)}),
                flush=True,
            )
        if args.batches == 8:
            batches = [
                torch.load(output / f"batch_{b}.pt", weights_only=True)
                for b in range(8)
            ]
            raw = {
                k: torch.stack([b[k] for b in batches])
                for k in ("returns", "first_rewards", "survived")
            }
            raw["axes"] = ["batch", "repeat", "state", "candidate"]
            raw["seeds"] = [b["seeds"] for b in batches]
            torch.save(raw, output / "raw_returns.pt")
            stats = summarize_returns(raw["returns"])
            torch.save(stats, output / "statistics.pt")
            report = {
                k: float(v.mean()) for k, v in stats.items() if k != "batch_means"
            }
            report["first_reward_negative_rate"] = float(
                (raw["first_rewards"] == -1).double().mean()
            )
            report["terminated_fraction"] = float((~raw["survived"]).double().mean())
            report["survivor_tail_width"] = 0.99**500 / 0.01
            report["batch_seconds"] = [b["seconds"] for b in batches]
            target = raw["returns"].mean((0, 1))
            pairs = list(itertools.combinations(range(8), 2))
            report["critics"] = []
            for prediction in saved["predictions"]:
                agreement = torch.stack(
                    [
                        torch.sign(prediction[:, a] - prediction[:, b])
                        * torch.sign(target[:, a] - target[:, b])
                        for a, b in pairs
                    ]
                )
                report["critics"].append(
                    {
                        "mae_to_finite_estimate": float(
                            (prediction - target).abs().mean()
                        ),
                        "pair_accuracy_to_noisy_estimate": float(
                            ((agreement + 1) / 2).mean()
                        ),
                        "within_state_prediction_range": float(
                            (prediction.amax(-1) - prediction.amin(-1)).mean()
                        ),
                    }
                )
            (output / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False)
            )
            print(json.dumps(report), flush=True)
    finally:
        env._backend.close()


if __name__ == "__main__":
    main()
