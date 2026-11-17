"""Prediction and controller correctness gates (run before any timing).

  1  scatter + step parity against the ordinary Q2 task step
  2  observation parity against Q2's own observation construction
  3  frozen-policy and Q inference parity against the artifact paths
  4  a deliberately different candidate changes the predicted trajectory
  5  controller rollouts do not touch the evaluation environment
  6  height cost / terminal conventions / seeded reproducibility

Exit status is non-zero when one-step prediction deviates from the ordinary task
step by more than 1e-3 in the observation or root state.

Run:
  PYTHONPATH=<repo>:scripts python verify_prediction_parity.py \
      --artifact <dir> --checkpoint <model.pt>
"""

from __future__ import annotations

import argparse
import sys

import torch

from gmppi_adapter import PredictionTask, capture_state
from gmppi_common import add_artifact_args, load_artifacts, make_eval_task

K = 4


def stats(tag, a, b):
    d = (a.double() - b.double()).abs()
    print(f"  {tag:<38} max={d.max().item():.3e} mean={d.mean().item():.3e}")


def report(name, value):
    print(f"  {name:<38} {value}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_artifact_args(parser, __file__)
    args = parser.parse_args()

    torch.manual_seed(0)
    generator = torch.Generator(device=args.device).manual_seed(1234)
    artifact, policy, spec = load_artifacts(args)
    height_spec = artifact.manifest["reward_spec"]
    gamma = float(artifact.manifest["gamma"])
    failure_value = float(artifact.manifest["failure_value"])

    eval_task = make_eval_task(policy, args.device)
    evaluate = eval_task  # one evaluation world, advanced by executed actions
    pred_task = PredictionTask(K, seed=11, device=args.device)
    pred_task.copy_physics_from(eval_task)

    print("== 1/2  one-step parity against the ordinary Q2 step ==")
    worst = {"obs": 0.0, "root": 0.0}
    pred_obs = None
    for trial in range(5):
        state = capture_state(evaluate.env)
        action = policy.sample_actions(evaluate.observation(), generator)

        evaluate.set_action_targets(action)
        evaluate.step()
        evaluate_obs = evaluate.observation().clone()
        evaluate_dof = evaluate.env.dof_pos.clone()
        evaluate_root = evaluate.env.root_states.clone()

        pred_task.scatter(state)
        pred_task.set_action_targets(action.expand(K, -1))
        pred_task.step()
        pred_obs = pred_task.observation()
        pred_dof = pred_task.env.dof_pos.clone()
        pred_root = pred_task.env.root_states.clone()

        stats(f"trial {trial} observation", pred_obs, evaluate_obs.expand(K, -1))
        stats(f"trial {trial} dof_pos", pred_dof, evaluate_dof.expand(K, -1))
        stats(f"trial {trial} root_states", pred_root, evaluate_root.expand(K, -1))
        worst["obs"] = max(
            worst["obs"],
            (pred_obs[0].double() - evaluate_obs[0].double()).abs().max().item(),
        )
        worst["root"] = max(
            worst["root"],
            (pred_root[0].double() - evaluate_root[0].double()).abs().max().item(),
        )
    spread = (pred_obs.double() - pred_obs[0].double()).abs().max().item()
    report("candidate worlds spread", f"{spread:.3e}")

    print("== 3  frozen-policy and Q parity against artifact paths ==")
    obs = evaluate.observation()
    batched = policy.mean_actions(obs.expand(K, -1))
    single = policy.mean_actions(obs)
    stats("policy batched vs single row", batched, single.expand(K, -1))

    mean_q = artifact.obs_mean.to(args.device)
    std_q = artifact.obs_std.to(args.device)
    candidates = policy.sample_actions(obs.expand(K, -1), generator)
    q_fast = artifact.q((obs.expand(K, -1) - mean_q) / std_q, candidates)
    q_artifact = artifact.predict(obs.expand(K, -1), candidates)
    stats("Q resident path vs HeightQ.predict", q_fast, q_artifact)
    report("Q values", [round(v, 6) for v in q_fast.tolist()])

    from learning.offline.q_artifact import HeightQ

    q_cpu = HeightQ(args.artifact).predict(obs.expand(K, -1).cpu(), candidates.cpu())
    stats("Q cuda vs Q cpu", q_fast, q_cpu.to(args.device))

    print("== 4  a different candidate changes the predicted outcome ==")
    state = capture_state(evaluate.env)
    action_a = policy.mean_actions(evaluate.observation())
    action_b = action_a + 0.5
    outcomes = {}
    for tag, action in (("a", action_a), ("b", action_b)):
        pred_task.scatter(state)
        pred_task.set_action_targets(action.expand(K, -1))
        pred_task.step()
        outcomes[tag] = (
            pred_task.observation().clone(),
            pred_task.base_height().clone(),
        )
    stats(
        "obs delta for a 0.5-unit action change",
        outcomes["a"][0],
        outcomes["b"][0],
    )
    report(
        "height a / b",
        f"{outcomes['a'][1][0].item():.6f} / {outcomes['b'][1][1].item():.6f}",
    )

    print("== 5  evaluator isolation ==")
    before = {
        name: getattr(evaluate.env, name).clone()
        for name in ("dof_pos", "dof_vel", "root_states")
    }
    pred_task.scatter(capture_state(evaluate.env))
    for _ in range(3):
        pred_task.set_action_targets(
            policy.sample_actions(pred_task.observation(), generator)
        )
        pred_task.step()
    for name, value in before.items():
        report(
            f"evaluator {name} unchanged",
            bool(torch.equal(value, getattr(evaluate.env, name))),
        )

    print("== 6  cost conventions ==")
    pred_task.scatter(capture_state(evaluate.env))
    height = pred_task.base_height()
    inside = (height - float(height_spec["reference_height_m"])).abs() <= float(
        height_spec["bound_m"]
    )
    reward = torch.where(
        inside,
        torch.full_like(height, float(height_spec["inside_reward"])),
        torch.full_like(height, float(height_spec["outside_reward"])),
    )
    report("height", [round(v, 4) for v in height.tolist()])
    report("height reward", reward.tolist())
    report("gamma / failure value", f"{gamma} / {failure_value:.3f}")

    seed_a = torch.Generator(device=args.device).manual_seed(99)
    seed_b = torch.Generator(device=args.device).manual_seed(99)
    a = policy.sample_actions(pred_task.observation(), seed_a)
    b = policy.sample_actions(pred_task.observation(), seed_b)
    report("seeded policy draws reproducible", bool(torch.equal(a, b)))

    print("\nworst-case one-step deviations (candidate 0 vs evaluation env)")
    for name, value in worst.items():
        print(f"  {name:<10} {value:.3e}")
    if worst["obs"] > 1e-3 or worst["root"] > 1e-3:
        print("FAIL: prediction deviates from the ordinary task step")
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
