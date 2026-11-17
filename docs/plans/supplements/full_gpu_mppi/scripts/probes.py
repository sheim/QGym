"""Supporting probes used by the report.

  --probe step        one Q2 Go2Trot control step vs world count (GPU vs host)
  --probe launch      how much of a control step is host launch time
  --probe weighting   MPPI importance-weight concentration vs temperature
  --probe multi-env   two evaluation environments, two controllers

Run:
  PYTHONPATH=<repo>:scripts python probes.py --probe step
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import warp as wp

from gmppi_adapter import PredictionTask
from gmppi_common import add_artifact_args, load_artifacts, make_eval_task
from gmppi_controller import GPUMPPIController


def events(fn, warmup=5, repeats=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def probe_step(args):
    """One control step: GPU time scales with worlds, host time does not."""
    for worlds in (1, 8, 64, 256, 1024):
        task = PredictionTask(worlds, seed=3, device=args.device)
        action = torch.zeros(worlds, 12, device=args.device)
        task.set_action_targets(action)
        task.step()
        gpu = events(task.step)
        print(
            f"K={worlds:<5} step={gpu:.3f} ms  steps/s={1000 / gpu:.0f}  "
            f"per-world-us={gpu * 1000 / worlds:.2f}"
        )


def probe_launch(args):
    """Compare wall time with the GPU busy time reported by torch.profiler."""
    from torch.profiler import ProfilerActivity, profile

    worlds = 256
    task = PredictionTask(worlds, seed=3, device=args.device)
    action = torch.zeros(worlds, 12, device=args.device)
    task.set_action_targets(action)
    for _ in range(5):
        task.step()
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = time.perf_counter()
    repeats = 20
    for _ in range(repeats):
        task.step()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - start) / repeats * 1000

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            task.step()
        torch.cuda.synchronize()
    events_ = prof.key_averages()
    busy = sum(getattr(e, "self_device_time_total", 0) for e in events_) / 10 / 1000
    print(
        f"K={worlds} wall={wall:.3f} ms/step  gpu_busy={busy:.3f} ms/step "
        f"({100 * busy / wall:.0f}% GPU)"
    )
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=8))


def probe_weighting(args):
    """Effective number of contributing candidates under the upstream update."""
    artifact, policy, _ = load_artifacts(args)
    evaluate = make_eval_task(policy, args.device)
    observation = evaluate.observation()
    for lambda_ in (0.1, 0.5, 2.0, 10.0):
        controller = GPUMPPIController(
            artifact,
            policy,
            evaluate,
            num_samples=256,
            horizon=4,
            num_policy_sequences=8,
            lambda_=lambda_,
            seed=0,
        )
        controller.action(observation, evaluate.env)
        torch.cuda.synchronize()
        weight = controller.mppi.omega
        effective = (1.0 / (weight**2).sum()).item()
        print(
            f"lambda={lambda_:<5} omega_max={weight.max().item():.4f} "
            f"effective_candidates={effective:.1f} "
            f"cost=({controller.mppi.cost_total.min().item():.2f},"
            f"{controller.mppi.cost_total.max().item():.2f})"
        )
        del controller


def probe_multi_env(args):
    """Candidate batching does not batch controllers."""
    artifact, policy, _ = load_artifacts(args)
    evaluate = make_eval_task(policy, args.device)
    observation = evaluate.observation()
    first = GPUMPPIController(
        artifact,
        policy,
        evaluate,
        num_samples=256,
        horizon=4,
        num_policy_sequences=8,
        seed=0,
    )
    second_env = PredictionTask(1, seed=17, device=args.device)
    second_env.copy_physics_from(evaluate)
    second = GPUMPPIController(
        artifact,
        policy,
        second_env,
        num_samples=256,
        horizon=4,
        num_policy_sequences=8,
        seed=5,
    )
    other_observation = second_env.observation()

    for _ in range(3):
        first.action(observation, evaluate.env)
        second.action(other_observation, second_env.env)
    torch.cuda.synchronize()

    one = events(lambda: first.action(observation, evaluate.env), 0, 10)
    both = events(
        lambda: (
            first.action(observation, evaluate.env),
            second.action(other_observation, second_env.env),
        ),
        0,
        10,
    )
    print(f"1 controller, 1 env : {one:.3f} ms/decision")
    print(f"2 controllers, 2 env: {both:.3f} ms/decision ({both / one:.2f}x)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_artifact_args(parser, __file__)
    parser.add_argument(
        "--probe",
        choices=("step", "launch", "weighting", "multi-env"),
        required=True,
    )
    args = parser.parse_args()
    wp.init()
    {
        "step": probe_step,
        "launch": probe_launch,
        "weighting": probe_weighting,
        "multi-env": probe_multi_env,
    }[args.probe](args)


if __name__ == "__main__":
    main()
