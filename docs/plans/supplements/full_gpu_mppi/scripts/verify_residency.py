"""Residency gates.

  7  the sync-free all-worlds commit reproduces SimBackend.reset_state exactly
  8  no host synchronisation anywhere inside controller.action()

``torch.cuda.set_sync_debug_mode("error")`` turns every synchronising call made
through PyTorch into a RuntimeError, so gate 8 is enforced rather than observed.

Gate 8 passes only with ``--patch-warp-sync``, which replaces Q2's assembled
state refresh with the bit-identical slice-only version. Without the patch the
unmodified backend synchronises in two places; the script reports both.

Run:
  PYTHONPATH=<repo>:scripts python verify_residency.py --patch-warp-sync
"""

from __future__ import annotations

import argparse
import contextlib

import torch

from gmppi_adapter import PredictionTask, capture_state
from gmppi_common import add_artifact_args, load_artifacts, make_eval_task
from gmppi_controller import GPUMPPIController

from gmppi_patch_warp_sync import install, uninstall


def commit_parity(args):
    artifact, policy, _ = load_artifacts(args)
    generator = torch.Generator(device=args.device).manual_seed(5)
    evaluate = make_eval_task(policy, args.device)
    pred = PredictionTask(32, seed=11, device=args.device)
    pred.copy_physics_from(evaluate)

    state = capture_state(evaluate.env)
    action = policy.sample_actions(evaluate.observation(), generator)
    fields = ("dof_pos", "dof_vel", "root_states", "contact_forces")

    pred.scatter(state, sync_free=False)
    committed = {name: getattr(pred.env, name).clone() for name in fields}
    pred.scatter(state, sync_free=True)
    print("  committed state, both commit paths:")
    for name in fields:
        value, other = committed[name], getattr(pred.env, name)
        delta = (value.double() - other.double()).abs().max().item()
        print(
            f"    {name:<16} identical={torch.equal(value, other)} maxdiff={delta:.3e}"
        )

    after = {}
    for tag in ("reference", "sync_free"):
        pred.scatter(state, sync_free=tag == "sync_free")
        pred.set_action_targets(action.expand(32, -1))
        pred.step()
        after[tag] = {name: getattr(pred.env, name).clone() for name in fields}
    print("  after one control step (engine reproducibility floor, not the commit):")
    for name in fields:
        a, b = after["reference"][name], after["sync_free"][name]
        delta = (a.double() - b.double()).abs().max().item()
        print(f"    {name:<16} identical={torch.equal(a, b)} maxdiff={delta:.3e}")


@contextlib.contextmanager
def sync_debug():
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode("default")


def sync_gate(args):
    artifact, policy, _ = load_artifacts(args)
    evaluate = make_eval_task(policy, args.device)
    controller = GPUMPPIController(
        artifact,
        policy,
        evaluate,
        num_samples=64,
        horizon=4,
        num_policy_sequences=4,
        seed=0,
    )
    observation = evaluate.observation()
    for _ in range(3):
        controller.action(observation, evaluate.env)
    torch.cuda.synchronize()

    failures = 0
    with sync_debug():
        for _ in range(3):
            try:
                controller.action(observation, evaluate.env)
            except RuntimeError as exc:
                failures += 1
                print(f"  SYNCHRONISATION: {exc}")
    torch.cuda.synchronize()
    print(f"  decisions with host synchronisation: {failures}/3")

    # The unmodified backend, for contrast.
    uninstall()
    task = PredictionTask(8, seed=3, device=args.device)
    mask = torch.ones(8, dtype=torch.bool, device=args.device)
    task.backend.reset_state(mask)
    torch.cuda.synchronize()
    for name, call in (
        ("SimBackend.reset_state", lambda: task.backend.reset_state(mask)),
        ("SimBackend._sync_assembled_states", task.backend._sync_assembled_states),
    ):
        with sync_debug():
            try:
                call()
                print(f"  {name}: no synchronisation")
            except RuntimeError:
                print(f"  {name}: SYNCHRONISES")
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_artifact_args(parser, __file__)
    parser.add_argument(
        "--patch-warp-sync",
        action="store_true",
        help="install the slice-only assembled-state refresh",
    )
    args = parser.parse_args()
    if args.patch_warp_sync:
        install()
    try:
        print("== 7  sync-free commit vs SimBackend.reset_state ==")
        commit_parity(args)
        print("== 8  host synchronisation inside the decision boundary ==")
        sync_gate(args)
    finally:
        uninstall()


if __name__ == "__main__":
    main()
