"""Shared setup for the full-GPU MPPI reproduction scripts.

Paths to the two external artifacts are CLI options (or environment variables)
because neither is committed:

  * the PPO checkpoint lives under ``logs/`` in the product checkout, which is
    git-ignored;
  * the fixed-policy height Q artifact was produced by the offline replay
    investigation and lives in a temporary directory.

See ``README.md`` in this directory for the exact values used by the report.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

from learning.offline.evaluation import load_policy_from_artifact
from learning.offline.q_artifact import HeightQ

from gmppi_adapter import PredictionTask

DEFAULT_ARTIFACT = "/tmp/opencode/height-q-exp-20261110-final-0p05-fair-v1/100pct"


def default_checkpoint(repo_root: str) -> str:
    return os.path.join(
        repo_root,
        "logs/go2trot/Nov10_10-36-22_gpu-final-600-trot-hist-20261110/model_600.pt",
    )


def repo_root_from(script_file: str) -> str:
    """Product checkout root: <repo>/docs/plans/supplements/full_gpu_mppi/scripts."""
    return os.path.abspath(os.path.join(os.path.dirname(script_file), *[os.pardir] * 5))


def add_artifact_args(parser: argparse.ArgumentParser, script_file: str) -> None:
    root = repo_root_from(script_file)
    parser.add_argument(
        "--artifact",
        default=os.environ.get("GMPPI_ARTIFACT", DEFAULT_ARTIFACT),
        help="height-Q artifact directory (manifest.json + q.pt)",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("GMPPI_CHECKPOINT", default_checkpoint(root)),
        help="frozen PPO checkpoint recorded in the artifact",
    )
    parser.add_argument("--device", default="cuda:0")


def load_artifacts(args):
    artifact = HeightQ(args.artifact)
    policy, spec = load_policy_from_artifact(artifact, args.checkpoint, args.device)
    artifact.q.to(args.device)
    return artifact, policy, spec


def make_eval_task(policy, device: str = "cuda:0", seed: int = 7, warmup: int = 40):
    """One evaluation world warmed into a trotting state by the real policy."""
    task = PredictionTask(1, seed=seed, device=device)
    for _ in range(warmup):
        task.set_action_targets(policy.mean_actions(task.observation()))
        task.step()
    torch.cuda.synchronize()
    return task


def require_cuda() -> None:
    if not torch.cuda.is_available():
        print("this investigation requires a CUDA device", file=sys.stderr)
        raise SystemExit(2)
