"""Semantics check for the foot contact-force extraction.

Two probes on the same frozen checkpoint:

1. **Policy standing** — zero all velocity commands, let the policy hold a
   stance, and verify that the four foot forces sum to the robot's bodyweight
   (and that no other body carries load).
2. **Open-loop default pose** — command the default joint configuration, which
   the low-gain PD controller cannot hold; the robot settles onto its shins.
   This shows that ``contact_forces`` is a *per-body net contact force*: the
   foot-only sum then falls short of bodyweight exactly by the load carried by
   the other bodies, and the all-body sum still equals bodyweight.

Run from the product repo root:

    uv run --frozen python \
      docs/plans/supplements/foot_contact_study/check_force_semantics.py
"""

import json
import pathlib

import torch

from collect_foot_contacts import parse_args, setup

HERE = pathlib.Path(__file__).resolve().parent
G = 9.81


def measure(env, tag, body_names, feet, samples=100):
    """Average the support ratios over ``samples`` control steps."""
    mass = env._backend.link_mass.sum(dim=1)
    weight = mass * G
    feet_ratio, all_ratio, per_foot = [], [], []
    per_body = torch.zeros(env.num_bodies, device=env.device)
    vz = []
    with torch.no_grad():
        for _ in range(samples):
            env.step()
            forces = env.contact_forces
            feet_ratio.append(forces.index_select(1, feet)[..., 2].sum(dim=1) / weight)
            all_ratio.append(forces[..., 2].sum(dim=1) / weight)
            per_foot.append(forces.index_select(1, feet)[..., 2] / weight[:, None])
            per_body = torch.maximum(per_body, forces.norm(dim=-1).max(dim=0).values)
            vz.append(env.base_lin_vel[:, 2].abs())
    feet_ratio = torch.cat(feet_ratio)
    all_ratio = torch.cat(all_ratio)
    per_foot = torch.cat(per_foot)
    touching = [
        body_names[i] for i in range(len(body_names)) if float(per_body[i]) > 1.0
    ]
    return {
        "probe": tag,
        "note": "Fz summed over all bodies is the net external contact force; "
        "it must equal bodyweight in a time average",
        "samples": int(feet_ratio.numel()),
        "sum_fz_feet_over_bodyweight": float(feet_ratio.mean()),
        "sum_fz_all_bodies_over_bodyweight": float(all_ratio.mean()),
        "sum_fz_all_bodies_over_bodyweight_std": float(all_ratio.std()),
        "per_foot_share_of_bodyweight": {
            body_names[int(i)]: float(per_foot[:, k].mean()) for k, i in enumerate(feet)
        },
        "bodies_with_contact": touching,
        "base_vz_abs_mean": float(torch.cat(vz).mean()),
    }


def main():
    args = parse_args([])
    args.device = "cuda:0"
    args.num_envs = 8
    args.tag = "static"
    results = {}
    with torch.no_grad():
        _, _, env, runner = setup(args)
        body_names = list(env.robot_layout.body_names)
        feet = env.feet_indices

        # 1) policy stance with zero velocity command
        for _ in range(300):
            env.commands[:] = 0.0
            runner.set_actions(
                runner.actor_cfg["actions"],
                runner.get_inference_actions(),
                runner.actor_cfg["disable_actions"],
            )
            env.step()
        results["policy_zero_command"] = measure(
            env, "policy_zero_command", body_names, feet
        )

        # 2) open-loop default pose, which the gains cannot hold
        for _ in range(120):
            env.dof_pos_target[:] = env.default_dof_pos
            env.step()
        results["default_pose_hold"] = measure(
            env, "default_pose_hold", body_names, feet
        )

    (HERE / "artifacts").mkdir(exist_ok=True)
    (HERE / "artifacts" / "static_check.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
