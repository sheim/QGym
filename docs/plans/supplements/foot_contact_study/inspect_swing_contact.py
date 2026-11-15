"""Where does foot/leg contact force come from during scheduled swing?

Replays the frozen policy on the MuJoCo CPU backend and records every
individual contact involving a robot body, with its force, penetration and the
leg phase of that body's foot. Contacts are then split by the task's own
schedule indicator ``sin(leg_phase) > 0`` (scheduled stance) and, for swing,
into "late lift-off" (phase in (pi, 1.5pi)) and "early touchdown" (phase in
(1.5pi, 2pi)).

Answers: the swing-phase force is ground contact, and it comes from the real
stance being longer than the clock's 50 % window, not from inertia, joint
reaction forces, or self-collision.

Run from the product repo root:

    uv run --frozen python \
      docs/plans/supplements/foot_contact_study/inspect_swing_contact.py
"""

import json
import pathlib

import mujoco
import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from collect_foot_contacts import parse_args, setup

HERE = pathlib.Path(__file__).resolve().parent
PHASE_OFFSET_CYCLES = {
    "FL_foot": 0.0,
    "FR_foot": 0.5,
    "RL_foot": 0.5,
    "RR_foot": 0.0,
}


def contact_force_world(model, data, index):
    """World-frame contact force on body2 (negative on body1).

    ``mj_contactForce`` puts force(3) then torque(3) in the contact frame;
    ``frame.T @ force`` matches ``cfrc_ext[3:6]`` exactly (see
    ``inspect_contact_sources.py``).
    """
    force = np.zeros(6)
    mujoco.mj_contactForce(model, data, index, force)
    frame = np.asarray(data.contact[index].frame).reshape(3, 3)
    return frame.T @ force[0:3]


def classify(phase):
    if np.sin(phase) > 0.0:
        return "stance"
    return "swing_late_liftoff" if phase < 1.5 * np.pi else "swing_early_touchdown"


def main():
    args = parse_args([])
    args.device = "cpu"
    args.num_envs = 4
    args.tag = "swing"
    records = []
    with torch.no_grad():
        _, _, env, runner = setup(args)
        backend = env._backend
        body_names = list(env.robot_layout.body_names)
        feet = [int(i) for i in env.feet_indices]
        foot_names = [body_names[i] for i in feet]
        for _ in range(200):
            runner.set_actions(
                runner.actor_cfg["actions"],
                runner.get_inference_actions(),
                runner.actor_cfg["disable_actions"],
            )
            env.step()
            phases = env.phase.detach().cpu().numpy().reshape(-1)
            for env_id in range(env.num_envs):
                model = backend._model_for_env(env_id)
                data = backend._datas[env_id]
                leg = {
                    name: float(
                        np.mod(
                            phases[env_id] + 2.0 * np.pi * PHASE_OFFSET_CYCLES[name],
                            2.0 * np.pi,
                        )
                    )
                    for name in foot_names
                }
                for c in range(data.ncon):
                    contact = data.contact[c]
                    b1 = int(model.geom_bodyid[contact.geom1])
                    b2 = int(model.geom_bodyid[contact.geom2])
                    n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1)
                    n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2)
                    world = contact_force_world(model, data, c)
                    for body, sign in ((b1, -1.0), (b2, 1.0)):
                        if body == 0:
                            continue
                        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
                        other = n2 if name == n1 else n1
                        prefix = name.split("_")[0] + "_foot"
                        foot_phase = leg.get(name) or leg.get(prefix)
                        records.append(
                            {
                                "env": env_id,
                                "body": name,
                                "other": other,
                                "is_foot": name in leg,
                                "force_n": float(np.linalg.norm(world)),
                                "fz_n": float(world[2]),
                                "penetration_m": float(-contact.dist),
                                "pos_z_m": float(contact.pos[2]),
                                "leg_phase": foot_phase,
                                "class": (
                                    classify(foot_phase)
                                    if foot_phase is not None
                                    else "no_leg"
                                ),
                            }
                        )
    feet_records = [r for r in records if r["is_foot"]]
    other_records = [r for r in records if not r["is_foot"]]
    summary = {
        "probes": {
            "steps": 200,
            "num_envs": int(args.num_envs),
            "contacts": len(records),
            "foot_contacts": len(feet_records),
            "non_foot_contacts": len(other_records),
        },
        "by_class": {},
        "foot_by_class": {},
        "non_foot_by_body": {},
        "swing_geometry": {},
        "interpretation": {
            "foot_swing_source": "ground plane contact (see other='world')",
            "not_sources": "no self-collision, no applied external force; "
            "joint reaction and inertial loads are not part of contact_forces",
        },
    }
    for klass in ("stance", "swing_late_liftoff", "swing_early_touchdown"):
        subset = [r for r in feet_records if r["class"] == klass]
        arr = np.array([r["force_n"] for r in subset]) if subset else np.array([0.0])
        summary["foot_by_class"][klass] = {
            "contacts": len(subset),
            "share_of_foot_contacts": len(subset) / max(len(feet_records), 1),
            "force_n_mean": float(arr.mean()),
            "force_n_p50": float(np.percentile(arr, 50)),
            "force_n_p95": float(np.percentile(arr, 95)),
            "force_n_max": float(arr.max()),
            "fraction_force_above_10n": float((arr > 10.0).mean()),
            "other_bodies": sorted({r["other"] for r in subset}),
        }
    for r in records:
        if r["is_foot"]:
            continue
        entry = summary["non_foot_by_body"].setdefault(
            r["body"], {"contacts": 0, "by_class": {}, "max_force_n": 0.0}
        )
        entry["contacts"] += 1
        entry["by_class"][r["class"]] = entry["by_class"].get(r["class"], 0) + 1
        entry["max_force_n"] = max(entry["max_force_n"], r["force_n"])
    swing = [r for r in feet_records if r["class"].startswith("swing")]
    if swing:
        pen = np.array([r["penetration_m"] for r in swing])
        posz = np.array([r["pos_z_m"] for r in swing])
        summary["swing_geometry"] = {
            "contacts": len(swing),
            "penetration_m_p50": float(np.percentile(pen, 50)),
            "penetration_m_p95": float(np.percentile(pen, 95)),
            "contact_z_m_p50": float(np.percentile(posz, 50)),
            "contact_z_m_p05": float(np.percentile(posz, 5)),
            "contact_z_m_p95": float(np.percentile(posz, 95)),
            "other_bodies": sorted({r["other"] for r in swing}),
        }

    ART = HERE / "artifacts"
    ART.mkdir(exist_ok=True)
    (ART / "swing_contact.json").write_text(json.dumps(summary, indent=2) + "\n")

    # ── figure ─────────────────────────────────────────────────────────────
    plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.25})
    colors = {
        "stance": "tab:blue",
        "swing_late_liftoff": "tab:orange",
        "swing_early_touchdown": "tab:red",
    }
    fig, (ax, axb) = plt.subplots(
        1, 2, figsize=(11.5, 4), gridspec_kw={"width_ratios": [3, 2]}
    )
    for klass, color in colors.items():
        subset = [r for r in feet_records if r["class"] == klass]
        if not subset:
            continue
        ax.scatter(
            [r["leg_phase"] for r in subset],
            [r["force_n"] for r in subset],
            s=6,
            alpha=0.35,
            color=color,
            label=f"{klass} ({len(subset)})",
        )
    for edge in (0.0, np.pi):
        ax.axvspan(edge, edge + np.pi, color="tab:green", alpha=0.05)
    ax.set_xlabel("leg phase of the contacting foot (rad)")
    ax.set_ylabel("contact force on that foot (N)")
    ax.set_title("Every foot contact, coloured by the task's schedule")
    ax.set_xlim(0, 2 * np.pi)
    ax.legend(fontsize=7, markerscale=2)
    labels = list(summary["foot_by_class"])
    axb.bar(
        range(len(labels)),
        [summary["foot_by_class"][k]["contacts"] for k in labels],
        color=[colors[k] for k in labels],
    )
    axb.set_xticks(range(len(labels)))
    axb.set_xticklabels([k.replace("_", "\n") for k in labels], fontsize=7)
    axb.set_ylabel("contact count")
    axb.set_title("Foot contacts by schedule class")
    fig.tight_layout()
    fig.savefig(
        HERE / "figures" / "fig10_swing_contact_source.png",
        dpi=100,
        pil_kwargs={"optimize": True},
    )
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
