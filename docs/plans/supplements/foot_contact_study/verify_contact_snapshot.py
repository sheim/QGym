"""Verify a contact-labelled replay snapshot and re-derive the r' thresholds.

Loads a snapshot through the product loader, checks the contact/bodyweight
fields, rebuilds the per-foot load series, detects touchdowns, and reports the
per-leg p90 thresholds and the resulting `-1 / 0` label rates. The numbers are
compared against the study's simulator-side measurements (10 % of touchdowns,
~2.8 % of control steps at p90).

Run from the product repo root:

    uv run --frozen python \
      docs/plans/supplements/foot_contact_study/verify_contact_snapshot.py \
      --snapshot logs/go2trot/<run>/replay/replay_iter_0000616
"""

import argparse
import json
import pathlib

import numpy as np
import torch

from learning.offline.dataset import load_snapshot
from learning.offline.rewards import observation_field_slice

HERE = pathlib.Path(__file__).resolve().parent
G = 9.81
PHASE_OFFSET_CYCLES = {
    "FL_foot": 0.0,
    "FR_foot": 0.5,
    "RL_foot": 0.5,
    "RR_foot": 0.0,
}
CONTACT_FRACTION_BW = 0.05


def run_runs(mask_1d):
    padded = np.concatenate([[False], mask_1d, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[0::2], edges[1::2]))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--percentile", type=float, default=90.0)
    parser.add_argument(
        "--out", default=str(HERE / "artifacts" / "snapshot_check.json")
    )
    return parser.parse_args()


def main():
    args = parse_args()
    snap = load_snapshot(args.snapshot)
    manifest = snap.manifest
    data = snap.data
    body_names = list(manifest["field_metadata"]["body_names"])
    feet = [body_names.index(name) for name in PHASE_OFFSET_CYCLES]
    num_envs = int(manifest["num_envs"])
    n_rows = len(snap)
    if n_rows % num_envs:
        raise RuntimeError(f"{n_rows} rows are not a multiple of {num_envs} envs")
    n_steps = n_rows // num_envs

    contact = data["next_contact_forces"].reshape(n_steps, num_envs, len(body_names), 3)
    contact = contact.numpy()
    bodyweight = data["bodyweight"].reshape(n_steps, num_envs, 1).numpy()
    obs = data["next_actor_obs"].numpy()

    # row order is step-major over environments; verify before reshaping
    env_ids = data["env_id"].reshape(n_steps, num_envs)
    expected = torch.arange(num_envs).expand(n_steps, num_envs)
    assert torch.equal(env_ids, expected), "snapshot rows are not step-major"

    phase_slice = observation_field_slice(manifest, "phase_obs")
    sin_phase = obs[:, phase_slice][..., 0].reshape(n_steps, num_envs)
    cos_phase = obs[:, phase_slice][..., 1].reshape(n_steps, num_envs)
    phase = np.mod(np.arctan2(sin_phase, cos_phase), 2.0 * np.pi)

    fz = contact[:, :, feet, 2]
    load = fz / bodyweight
    # A terminated transition gets reward 0 by the zero-after-terminal
    # convention, and its force describes a fall, not a gait event.
    terminated = data["terminated"].reshape(n_steps, num_envs).numpy().astype(bool)
    terminated_fraction = float(terminated.mean())
    bw = bodyweight.reshape(-1)
    bw_is_constant = bool(np.allclose(bodyweight, bodyweight[:1], rtol=1e-6))

    peaks = {}
    peak_records = []
    for env in range(num_envs):
        for k, name in enumerate(PHASE_OFFSET_CYCLES):
            contact_mask = (load[:, env, k] > CONTACT_FRACTION_BW) & ~terminated[:, env]
            for start, end in run_runs(contact_mask):
                if end - start < 5:
                    continue
                peak = float(load[start:end, env, k].max())
                if peak < 0.30:
                    continue
                peaks.setdefault(name, []).append(peak)
                peak_records.append(
                    {
                        "env": env,
                        "foot": name,
                        "peak": peak,
                        "peak_step": start + int(np.argmax(load[start:end, env, k])),
                    }
                )

    limits = {
        name: float(np.percentile(np.asarray(peaks[name]), args.percentile))
        for name in PHASE_OFFSET_CYCLES
    }
    exceed = np.stack(
        [load[:, :, k] > limits[name] for k, name in enumerate(PHASE_OFFSET_CYCLES)],
        axis=-1,
    )
    any_exceed = exceed.any(axis=-1) & ~terminated
    flagged = sum(
        1 for record in peak_records if record["peak"] > limits[record["foot"]]
    )
    per_leg = {
        name: float(exceed[:, :, k].mean())
        for k, name in enumerate(PHASE_OFFSET_CYCLES)
    }
    phase_offsets = np.stack(
        [
            np.mod(phase + 2.0 * np.pi * PHASE_OFFSET_CYCLES[name], 2.0 * np.pi)
            for name in PHASE_OFFSET_CYCLES
        ],
        axis=-1,
    )
    swing = ~(np.sin(phase_offsets) > 0.0)
    result = {
        "snapshot": args.snapshot,
        "valid_rows": n_rows,
        "num_envs": num_envs,
        "steps_per_env": n_steps,
        "format": manifest["format"],
        "extra_fields": manifest["extra_fields"],
        "field_dims": {k: manifest["field_dims"][k] for k in manifest["extra_fields"]},
        "checkpoint": manifest.get("run_id"),
        "bodyweight_n": {
            "mean": float(bw.mean()),
            "min": float(bw.min()),
            "max": float(bw.max()),
            "std": float(bw.std()),
            "constant_within_env": bw_is_constant,
        },
        "feet": {
            name: body_names[index] for name, index in zip(PHASE_OFFSET_CYCLES, feet)
        },
        "terminated_fraction_percent": 100.0 * terminated_fraction,
        "touchdowns": len(peak_records),
        "thresholds_bw": limits,
        "thresholds_n_at_mean_bodyweight": {
            name: limits[name] * float(bw.mean()) for name in limits
        },
        "label": {
            "percentile": args.percentile,
            "steps_flagged_percent": 100.0 * float(any_exceed.mean()),
            "steps_flagged_per_leg_percent": {
                name: 100.0 * value for name, value in per_leg.items()
            },
            "flag_rate_inside_scheduled_swing_percent": 100.0
            * float((any_exceed[..., None] & swing).sum() / max(swing.sum(), 1)),
            "touchdowns_flagged_percent": 100.0 * flagged / max(len(peak_records), 1),
        },
        "study_reference": {
            "steps_flagged_percent": 2.82,
            "touchdowns_flagged_percent": 10.0,
            "thresholds_bw": {
                "FL_foot": 0.900,
                "FR_foot": 0.968,
                "RL_foot": 0.787,
                "RR_foot": 0.778,
            },
            "note": "from 128 Warp worlds x 600 steps of model_600, "
            "pre-action force; this file uses post-action force, so small "
            "differences are expected",
        },
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
