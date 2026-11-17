"""Nsight Systems target: a warmed, isolated controller hot window.

Everything (environment build, Warp compilation, artifact load) happens before
``torch.cuda.profiler.start()``, and the range ends with an explicit
synchronise, so every transfer in the report belongs to the decision boundary.

Run:
  PYTHONPATH=<repo>:scripts GMPPI_NVTX=1 nsys profile \
      -o hot --trace=cuda,nvtx -c cudaProfilerApi --capture-range-end=stop-shutdown \
      python profile_hot_path.py --samples 256 --horizon 8 --patch-warp-sync

  nsys stats --report \
      nvtx_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum hot.nsys-rep

NVTX child ranges label candidate generation, physics, observation building,
policy rollout, Q evaluation and the MPPI update.
"""

from __future__ import annotations

import argparse

import torch

from gmppi_common import add_artifact_args, load_artifacts, make_eval_task
from gmppi_controller import GPUMPPIController


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_artifact_args(parser, __file__)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--policy-sequences", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--decisions", type=int, default=6)
    parser.add_argument("--patch-warp-sync", action="store_true")
    args = parser.parse_args()

    if args.patch_warp_sync:
        from gmppi_patch_warp_sync import install

        install()

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

    for _ in range(args.warmup):
        controller.action(observation, evaluate.env)
    torch.cuda.synchronize()

    torch.cuda.profiler.start()
    for _ in range(args.decisions):
        controller.action(observation, evaluate.env)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print("hot window done")


if __name__ == "__main__":
    main()
