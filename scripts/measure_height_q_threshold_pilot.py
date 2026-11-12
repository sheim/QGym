"""Measure the nominal-trajectory height-Q distribution for threshold calibration.

No candidate intervention: every step executes the first of the same eight
stochastic policy draws (the historical baseline candidate0, now index 1).
The exact policy mean is added at index 0 and all nine candidates are scored
with the original fair-100 PPO-data critic.
"""

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
from learning.offline.contracts import replay_config_fingerprint, sha256_file
from learning.offline.evaluation import _policy_observation, load_policy_from_artifact
from learning.offline.fixed_state_variance import (
    WarpInsertionSnapshot,
    accumulate_return,
)
from learning.offline.persistent_intervention import (
    sample_candidates_with_mean,
    score_candidates,
)
from learning.offline.q_artifact import HeightQ


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        default="/tmp/opencode/height-q-fixed-variance-20261110-v1",
    )
    p.add_argument(
        "--output",
        default="/tmp/opencode/height-q-threshold-pilot-20261111-v1",
    )
    p.add_argument(
        "--checkpoint",
        default="logs/go2trot/Nov10_10-36-22_gpu-final-600-trot-hist-20261110/model_600.pt",
    )
    p.add_argument(
        "--load_run",
        default="Nov10_10-36-22_gpu-final-600-trot-hist-20261110",
    )
    p.add_argument(
        "--artifact",
        default="/tmp/opencode/height-q-exp-20261110-final-0p05-fair-v1/100pct",
    )
    p.add_argument(
        "--calls",
        type=int,
        default=8,
        help="Eight lanes per call; calls*8 executions per state (default 64).",
    )
    p.add_argument("--horizon", type=int, default=500)
    p.add_argument("--seed_base", type=int, default=7100000)
    p.add_argument("--max_projected_seconds", type=float, default=240.0)
    args = p.parse_args()
    if not 1 <= args.calls <= 16:
        raise ValueError("--calls must be 1..16 (16 calls = 128 executions/state)")
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")

    import gym.envs  # noqa: F401

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
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

    provenance = {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": spec.checkpoint_sha256,
        "artifact": args.artifact,
        "artifact_sha256": {
            name: sha256_file(Path(args.artifact) / name)
            for name in ("manifest.json", "q.pt")
        },
        "reward_spec": reward,
        "gamma": artifact.manifest["gamma"],
        "protocol": {
            "calls": args.calls,
            "lanes_per_call": 8,
            "executions_per_state": args.calls * 8,
            "horizon": args.horizon,
            "seed_base": args.seed_base,
            "seeds": [args.seed_base + 10 * call for call in range(args.calls)],
            "world_axes": ["lane8", "state128"],
            "state_selection": (
                "first 128 prefix survivors from the fixed-variance insertion snapshot"
            ),
            "initial_observation": (
                "candidate0-world noisy insertion observation replicated across lanes"
            ),
            "nominal_action": (
                "new index 1 = first of the same eight Gaussian draws; this is the "
                "historical baseline candidate0"
            ),
            "q_scoring": (
                "all nine candidates (exact policy mean at index 0 plus eight "
                "stochastic draws at indices 1..8) scored at every decision"
            ),
            "interventions": "none; the executed action is the nominal first draw",
            "masking": "Q is recorded before the step for alive worlds only",
            "restore": (
                "native snapshot restore per call; no clamp, no prefix rerun, no "
                "timeout reset"
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
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    with (output / "commands.txt").open("a") as f:
        f.write(
            "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen python "
            + " ".join(sys.argv)
            + "\n"
        )
    (output / "dirty.txt").write_text(
        subprocess.check_output(["git", "status", "--short"], text=True)
    )
    code_paths = [
        Path(__file__),
        Path("learning/offline/persistent_intervention.py"),
        Path("learning/offline/fixed_state_variance.py"),
        Path("learning/offline/q_artifact.py"),
    ]
    (output / "code.json").write_text(
        json.dumps({str(path): sha256_file(path) for path in code_paths}, indent=2)
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
        original.assert_restored()

        ids = (torch.arange(128) * 8).repeat(8)
        snapshot = WarpInsertionSnapshot(env)
        snapshot.load_export(saved["snapshot"], ids)
        snapshot.restore()
        snapshot.assert_restored()
        initial_obs = saved["insertion_observations"][ids].to(env.device)
        if initial_obs.shape != (1024, artifact.obs_dim):
            raise ValueError("unexpected insertion observation shape")

        diag_candidates = sample_candidates_with_mean(
            policy,
            initial_obs,
            torch.Generator(device=env.device).manual_seed(99000011),
        )
        query = initial_obs[:, None].expand(-1, 9, -1).flatten(0, 1)
        actions = diag_candidates.flatten(0, 1)
        cpu = artifact.predict(query.cpu(), actions.cpu())
        artifact.q.to(env.device)
        artifact.obs_mean = artifact.obs_mean.to(env.device)
        artifact.obs_std = artifact.obs_std.to(env.device)
        gpu = artifact.predict(query, actions)
        split = torch.cat(
            [
                artifact.predict(obs_chunk, act_chunk)
                for obs_chunk, act_chunk in zip(query.split(257), actions.split(257))
            ]
        )
        torch.testing.assert_close(cpu, gpu.cpu(), atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(gpu, split, atol=2e-5, rtol=2e-5)
        validation = {
            "inventory_entries": len(snapshot.entries),
            "source_validation": saved["validation"],
            "initial_observation_shape": list(initial_obs.shape),
            "initial_observation_mean": float(initial_obs.mean()),
            "initial_observation_std": float(initial_obs.std()),
            "cpu_gpu_q": {
                "max_error": float((cpu - gpu.cpu()).abs().max()),
                "batch_error": float((gpu - split).abs().max()),
                "argmax_disagreements": int(
                    (
                        cpu.reshape(-1, 9).argmax(-1)
                        != gpu.cpu().reshape(-1, 9).argmax(-1)
                    ).sum()
                ),
            },
        }
        (output / "validation.json").write_text(json.dumps(validation, indent=2))
        print(json.dumps({"validation": validation}), flush=True)

        timings = []
        for call in range(args.calls):
            snapshot.restore()
            snapshot.assert_restored()
            seed = args.seed_base + 10 * call
            torch.manual_seed(seed)
            obs_rng = torch.Generator(device=env.device).manual_seed(seed + 1)
            act_rng = torch.Generator(device=env.device).manual_seed(seed + 2)
            obs = initial_obs.clone()
            alive = torch.ones(1024, dtype=torch.bool, device=env.device)
            q_hist = torch.empty(
                (args.horizon, 1024, 9), dtype=torch.float32, device=env.device
            )
            alive_hist = torch.empty(
                (args.horizon, 1024), dtype=torch.bool, device=env.device
            )
            returns = torch.zeros(1024, dtype=torch.float64, device=env.device)
            steps = torch.zeros(1024, dtype=torch.float64, device=env.device)
            oob = torch.zeros(1024, dtype=torch.float64, device=env.device)
            xy_error_sq = torch.zeros(1024, dtype=torch.float64, device=env.device)
            yaw_error_sq = torch.zeros(1024, dtype=torch.float64, device=env.device)
            speed_sum = torch.zeros(1024, dtype=torch.float64, device=env.device)
            command_speed_sum = torch.zeros(
                1024, dtype=torch.float64, device=env.device
            )
            first_reward = None
            first_terminated = None
            timeout_seen = torch.zeros(1024, dtype=torch.bool, device=env.device)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for step in range(args.horizon):
                candidates = sample_candidates_with_mean(policy, obs, act_rng)
                scores = score_candidates(artifact, obs, candidates)
                q_hist[step] = scores
                alive_hist[step] = alive
                action = candidates[:, 1]
                command = env.commands.clone()
                env.set_states(spec.action_names, action)
                env.step()
                reward = -((env.base_height[:, 0] - 0.32).abs() > 0.05).double()
                if step == 0:
                    first_reward = reward.clone()
                    first_terminated = env.terminated.clone()
                timeout_seen |= env.timed_out
                steps += alive
                oob += -reward * alive
                returns, alive = accumulate_return(
                    returns, alive, reward, env.terminated, step, 0.99
                )
                xy_error_sq += (env.base_lin_vel[:, :2] - command[:, :2]).square().sum(
                    -1
                ) * alive
                yaw_error_sq += (
                    env.base_ang_vel[:, 2] - command[:, 2]
                ).square() * alive
                speed_sum += env.base_lin_vel[:, :2].norm(dim=-1) * alive
                command_speed_sum += command[:, :2].norm(dim=-1) * alive
                if step != args.horizon - 1:
                    obs = _policy_observation(
                        env,
                        spec.obs_names,
                        spec.obs_dims,
                        spec.noise,
                        spec.scales,
                        obs_rng,
                    )
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            if timeout_seen.any():
                raise RuntimeError("administrative timeout during pilot rollout")
            result = {
                "q": q_hist.cpu(),
                "alive": alive_hist.cpu(),
                "returns": returns.cpu(),
                "steps": steps.cpu(),
                "oob": oob.cpu(),
                "xy_error_sq": xy_error_sq.cpu(),
                "yaw_error_sq": yaw_error_sq.cpu(),
                "speed_sum": speed_sum.cpu(),
                "command_speed_sum": command_speed_sum.cpu(),
                "first_rewards": first_reward.cpu(),
                "first_terminated": first_terminated.cpu(),
                "survived": alive.cpu(),
                "timeout_seen": timeout_seen.cpu(),
                "seed": seed,
                "seconds": seconds,
            }
            torch.save(result, output / f"call_{call:02d}.pt")
            timing = {
                "call": call,
                "seed": seed,
                "seconds": seconds,
                "projected_total_seconds": seconds * args.calls,
                "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
            timings.append(timing)
            print(json.dumps(timing), flush=True)
            if call == 0 and seconds * args.calls > args.max_projected_seconds:
                (output / "runtime_blocker.json").write_text(
                    json.dumps({"timings": timings}, indent=2)
                )
                raise RuntimeError(
                    "projected pilot runtime exceeds approval gate; first call retained"
                )
        (output / "benchmark.json").write_text(json.dumps(timings, indent=2))
    finally:
        env._backend.close()


if __name__ == "__main__":
    main()
