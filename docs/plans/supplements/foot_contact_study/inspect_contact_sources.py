"""Decompose the foot contact force into the MuJoCo contacts that produce it.

``contact_forces`` is one net 3-vector per body. This script reaches into the
MuJoCo CPU world, enumerates ``data.contact``, classifies each contact as
foot/ground, foot/self (another robot body), or other/other, sums the contact
forces per body, and compares that sum with the published tensor.

Answers: is the published foot force only the ground reaction, or does it also
contain self-collision / other external sources?

Run from the product repo root:

    uv run --frozen python \
      docs/plans/supplements/foot_contact_study/inspect_contact_sources.py
"""

import json
import pathlib

import mujoco
import numpy as np
import torch

from collect_foot_contacts import parse_args, setup

HERE = pathlib.Path(__file__).resolve().parent


def contact_wrench(model, data, index, transpose=False):
    """Contact force in world coordinates, acting on body2 (negative on body1)."""
    force = np.zeros(6)
    mujoco.mj_contactForce(model, data, index, force)
    # mj_contactForce fills force(3) then torque(3) in the contact frame; a
    # point contact has zero torque. ``frame`` is the contact-frame-to-world
    # rotation, so world = frame @ force.
    frame = np.asarray(data.contact[index].frame).reshape(3, 3)
    if transpose:
        return frame.T @ force[0:3]
    return frame @ force[0:3]


def main():
    args = parse_args([])
    args.device = "cpu"
    args.num_envs = 4
    args.tag = "contacts"
    body_names = None
    rows = []
    with torch.no_grad():
        env_cfg, _, env, runner = setup(args)
        backend = env._backend
        body_names = list(env.robot_layout.body_names)
        feet = {int(i) for i in env.feet_indices}
        # MuJoCo body ids differ from canonical ids; map by name.
        for _ in range(200):
            runner.set_actions(
                runner.actor_cfg["actions"],
                runner.get_inference_actions(),
                runner.actor_cfg["disable_actions"],
            )
            env.step()
            published = env.contact_forces.detach().cpu().numpy()
            for env_id in range(env.num_envs):
                model = backend._model_for_env(env_id)
                data = backend._datas[env_id]
                sums = np.zeros((env.num_bodies, 3))
                sums_t = np.zeros((env.num_bodies, 3))
                kinds = {
                    "foot_ground": 0,
                    "foot_self": 0,
                    "other_ground": 0,
                    "self_self": 0,
                }
                nonfoot_bodies = set()
                pairs = {}
                for c in range(data.ncon):
                    contact = data.contact[c]
                    b1 = int(model.geom_bodyid[contact.geom1])
                    b2 = int(model.geom_bodyid[contact.geom2])
                    world = contact_wrench(model, data, c)
                    world_t = contact_wrench(model, data, c, transpose=True)
                    for body, sign in ((b1, -1.0), (b2, 1.0)):
                        if body == 0:  # world body carries no force
                            continue
                        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
                        canonical = body_names.index(name)
                        sums[canonical] += sign * world
                        sums_t[canonical] += sign * world_t
                    n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1)
                    n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2)
                    f1 = n1 != "world" and body_names.index(n1) in feet
                    f2 = n2 != "world" and body_names.index(n2) in feet
                    world_ground = n1 == "world" or n2 == "world"
                    if (f1 or f2) and world_ground:
                        kinds["foot_ground"] += 1
                    elif f1 or f2:
                        kinds["foot_self"] += 1
                        nonfoot_bodies.add(n2 if f1 else n1)
                    elif world_ground:
                        kinds["other_ground"] += 1
                        nonfoot_bodies.add(n1 if n2 == "world" else n2)
                    else:
                        kinds["self_self"] += 1
                        if n1 != "world":
                            nonfoot_bodies.add(n1)
                        if n2 != "world":
                            nonfoot_bodies.add(n2)
                        key = " | ".join(sorted((n1, n2)))
                        pairs[key] = pairs.get(key, 0) + 1
                rows.append(
                    {
                        "env": env_id,
                        "ncon": int(data.ncon),
                        "kinds": kinds,
                        "nonfoot_bodies": sorted(nonfoot_bodies),
                        "self_pairs": pairs,
                        "feet_published": published[env_id, list(env.feet_indices)],
                        "feet_contact_sum": sums[list(env.feet_indices)],
                        "error_frame": np.abs(
                            published[env_id] - sums.astype(np.float32)
                        ).max(),
                        "error_frame_transposed": np.abs(
                            published[env_id] - sums_t.astype(np.float32)
                        ).max(),
                    }
                )

    err = np.array([r["error_frame"] for r in rows])
    err_t = np.array([r["error_frame_transposed"] for r in rows])
    total = {k: sum(r["kinds"][k] for r in rows) for k in rows[0]["kinds"]}
    touching = {}
    for r in rows:
        for name in r["nonfoot_bodies"]:
            touching[name] = touching.get(name, 0) + 1
    pairs = {}
    for r in rows:
        for key, count in r["self_pairs"].items():
            pairs[key] = pairs.get(key, 0) + count
    summary = {
        "steps": 200,
        "num_envs": int(args.num_envs),
        "samples": len(rows),
        "contact_kinds": total,
        "world_force_convention": "contact_force_world = contact.frame.T @ "
        "mj_contactForce(...)[0:3] (force is the first three entries, "
        "not the torque)",
        "max_abs_error_tensor_vs_contact_sum_n": float(err_t.max()),
        "max_abs_error_with_other_rotation_convention_n": float(err.max()),
        "samples_with_foot_self_contact": int(
            sum(1 for r in rows if r["kinds"]["foot_self"] > 0)
        ),
        "samples_with_other_ground_contact": int(
            sum(1 for r in rows if r["kinds"]["other_ground"] > 0)
        ),
        "samples_with_self_self_contact": int(
            sum(1 for r in rows if r["kinds"]["self_self"] > 0)
        ),
        "nonfoot_bodies_in_contact": touching,
        "self_contact_pairs": dict(sorted(pairs.items(), key=lambda item: -item[1])),
    }
    (HERE / "artifacts").mkdir(exist_ok=True)
    (HERE / "artifacts" / "contact_sources.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
