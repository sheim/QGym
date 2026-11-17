"""Warmed-latency, memory and component measurements.

The measured boundary is ``controller.action(observation_on_gpu) -> action_on_gpu``.
Artifact loading, environment construction and Warp kernel compilation are
outside it; timings synchronise only at that boundary.

Run:
  PYTHONPATH=<repo>:scripts python measure_cost.py \
      --samples 256 --horizon 8 --policy-sequences 8 --patch-warp-sync
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess

import torch

from gmppi_common import add_artifact_args, load_artifacts, make_eval_task
from gmppi_controller import GPUMPPIController


def gpu_memory_mib(device_index: int = 0) -> float:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            str(device_index),
        ]
    )
    return float(out.decode().strip())


def time_call(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_artifact_args(parser, __file__)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--policy-sequences", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--patch-warp-sync", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.patch_warp_sync:
        from gmppi_patch_warp_sync import install

        install()

    torch.cuda.reset_peak_memory_stats()
    baseline_used = gpu_memory_mib()
    artifact, policy, _ = load_artifacts(args)
    evaluate = make_eval_task(policy, args.device)
    controller = GPUMPPIController(
        artifact,
        policy,
        evaluate,
        num_samples=args.samples,
        horizon=args.horizon,
        num_policy_sequences=args.policy_sequences,
        seed=0,
    )
    observation = evaluate.observation()

    print(
        f"K={args.samples} T={args.horizon} M_policy={args.policy_sequences} "
        f"worlds: candidates={args.samples} policy={args.policy_sequences} "
        f"evaluation=1"
    )
    print(f"gpu memory before build {baseline_used:.0f} MiB")
    print(f"gpu memory after build  {gpu_memory_mib():.0f} MiB")

    decision = time_call(
        lambda: controller.action(observation, evaluate.env),
        args.warmup,
        args.repeats,
    )

    # Component measurements run the same work in isolation.
    nominal = controller.mppi.U[:1].expand(args.samples, -1)

    def physics_only():
        for _ in range(args.horizon):
            controller.pred.set_action_targets(nominal)
            controller.pred.step()

    def physics_and_obs():
        for _ in range(args.horizon):
            controller.pred.set_action_targets(nominal)
            controller.pred.step()
            controller.pred.observation()

    def policy_rollout():
        controller.policy_task.scatter(controller.sampler._pending_state)
        for _ in range(args.horizon):
            action = policy.sample_actions(
                controller.policy_task.observation(), controller.generator
            )
            controller.policy_task.set_action_targets(action)
            controller.policy_task.step()

    def policy_rollout_obs():
        controller.policy_task.scatter(controller.sampler._pending_state)
        for _ in range(args.horizon):
            observation_ = controller.policy_task.observation()
            action = policy.sample_actions(observation_, controller.generator)
            controller.policy_task.set_action_targets(action)
            controller.policy_task.step()

    def terminal_policy_q():
        observation_ = controller.pred.observation()
        action = policy.mean_actions(observation_)
        controller.q((observation_ - controller.q_mean) / controller.q_std, action)

    def scatter_only():
        from gmppi_adapter import capture_state

        controller.pred.scatter(capture_state(evaluate.env))

    components = {
        "scatter (K worlds)": time_call(scatter_only, 3, args.repeats),
        "physics only (T steps)": time_call(physics_only, 3, args.repeats),
        "physics+obs (T steps)": time_call(physics_and_obs, 3, args.repeats),
        "policy rollout (M,T)": time_call(policy_rollout, 3, args.repeats),
        "policy rollout+obs (M,T)": time_call(policy_rollout_obs, 3, args.repeats),
        "terminal policy+Q (K)": time_call(terminal_policy_q, 3, args.repeats),
    }

    physics_steps = args.horizon * args.samples
    result = {
        "num_samples": args.samples,
        "horizon": args.horizon,
        "num_policy_sequences": args.policy_sequences,
        "patched_sync": bool(args.patch_warp_sync),
        "physics_control_steps_per_decision": physics_steps,
        "policy_rollout_steps_per_decision": args.horizon * args.policy_sequences,
        "q_rows_per_decision": args.samples,
        "decision": decision,
        "components": components,
        "gpu_memory_before_mib": baseline_used,
        "gpu_memory_after_mib": gpu_memory_mib(),
        "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "decisions_per_second": 1000.0 / decision["median_ms"],
        "physics_steps_per_second": physics_steps * 1000.0 / decision["median_ms"],
    }

    print("\ncomponent timings (median ms)")
    for name, value in components.items():
        print(
            f"  {name:<28} {value['median_ms']:.3f} "
            f"(min {value['min_ms']:.3f} max {value['max_ms']:.3f})"
        )
    print(
        f"\ndecision median {decision['median_ms']:.3f} ms "
        f"({result['decisions_per_second']:.1f} decisions/s)"
    )
    print(
        f"physics control steps/decision {physics_steps} -> "
        f"{result['physics_steps_per_second']:.0f} steps/s"
    )
    print(
        f"tensor peak allocated {result['torch_peak_allocated_mib']:.0f} MiB, "
        f"reserved {result['torch_peak_reserved_mib']:.0f} MiB"
    )
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
