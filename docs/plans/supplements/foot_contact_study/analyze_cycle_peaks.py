"""Per-stance and per-cycle peak foot force from the recorded rollouts.

Segments each environment's recorded trajectory into stance events (contiguous
runs of foot contact) and gait cycles (phase wraps), then reports the peak
vertical force, impulse and timing of each.

Run from the product repo root:

    uv run --frozen python \
      docs/plans/supplements/foot_contact_study/analyze_cycle_peaks.py
"""

import json
import pathlib

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
ART = HERE / "artifacts"
FIG = HERE / "figures"
G = 9.81

PHASE_OFFSET_CYCLES = {
    "FL_foot": 0.0,
    "FR_foot": 0.5,
    "RL_foot": 0.5,
    "RR_foot": 0.0,
}
CONTACT_FRACTION_BW = 0.05


def quantiles(values, points=(5, 50, 90, 95, 99)):
    values = np.asarray(values, dtype=np.float64)
    return {f"p{point}": float(np.percentile(values, point)) for point in points} | {
        "mean": float(values.mean()),
        "max": float(values.max()),
    }


def run_runs(mask_1d):
    """Return (start, end) index pairs of contiguous True runs."""
    padded = np.concatenate([[False], mask_1d, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[0::2], edges[1::2]))


def main(tag="gpu"):
    data = np.load(DATA / f"{tag}.npz")
    meta = json.loads((DATA / f"{tag}.json").read_text())
    feet = meta["feet_names"]
    dt = float(data["dt"])
    bodyweight = data["total_mass"] * G
    load = data["foot_forces"][..., 2] / bodyweight[None, :, None]
    fz = data["foot_forces"][..., 2]
    fnorm = np.linalg.norm(data["foot_forces"], axis=-1)
    phase = data["phase"].astype(np.float64)
    phase_frequency = data["phase_frequency"].astype(np.float64)
    commands = data["commands"].astype(np.float64)
    n_steps, n_envs = fz.shape[:2]
    contact = load > CONTACT_FRACTION_BW
    leg_phase = np.stack(
        [
            np.mod(phase + 2.0 * np.pi * PHASE_OFFSET_CYCLES[name], 2.0 * np.pi)
            for name in feet
        ],
        axis=-1,
    )
    scheduled_stance = np.sin(leg_phase) > 0.0

    # cycle index per env from phase wraps (no resets in these recordings)
    wraps = np.zeros_like(phase, dtype=np.int64)
    wraps[1:] = np.diff(phase, axis=0) < -np.pi
    cycle_id = np.cumsum(wraps, axis=0)
    cycle_length = np.zeros_like(phase)
    for env in range(n_envs):
        ids, counts = np.unique(cycle_id[:, env], return_counts=True)
        for value, count in zip(ids, counts):
            cycle_length[cycle_id[:, env] == value, env] = count

    events = []
    for env in range(n_envs):
        for k, name in enumerate(feet):
            for start, end in run_runs(contact[:, env, k]):
                if end - start < 2:  # ignore single-step solver blips
                    continue
                window = slice(start, end)
                peak_rel = int(np.argmax(fz[window, env, k]))
                peak_step = start + peak_rel
                events.append(
                    {
                        "env": env,
                        "foot": name,
                        "cycle": int(cycle_id[peak_step, env]),
                        "start": start,
                        "end": end,
                        "steps": end - start,
                        "duration_s": (end - start) * dt,
                        "peak_fz_n": float(fz[peak_step, env, k]),
                        "peak_norm_n": float(fnorm[peak_step, env, k]),
                        "peak_load_bw": float(load[peak_step, env, k]),
                        "peak_phase_in_stance": float(leg_phase[peak_step, env, k]),
                        "impulse_ns": float(fz[window, env, k].sum() * dt),
                        "mean_load_bw": float(load[window, env, k].mean()),
                        "stance_fraction_in_scheduled_swing": float(
                            (~scheduled_stance[window, env, k]).mean()
                        ),
                        "vx": float(commands[peak_step, env, 0]),
                        "frequency_hz": float(phase_frequency[peak_step, env]),
                        "cycle_steps": int(cycle_length[peak_step, env]),
                    }
                )

    # A contact run can be a full stance or a short light touch (the front feet
    # graze during scheduled swing). Keep both, but only the loaded runs are
    # "touchdowns" for peak-force statistics.
    for event in events:
        event["loaded"] = event["steps"] >= 5 and event["peak_load_bw"] >= 0.30

    per_cycle = []
    for env in range(n_envs):
        cycles = np.unique(cycle_id[:, env])
        # the first and last cycle of a recording are cut by the recording
        # window, so only analyse complete interior cycles
        for cycle in cycles[1:-1]:
            in_cycle = cycle_id[:, env] == cycle
            for k, name in enumerate(feet):
                per_cycle.append(
                    {
                        "env": env,
                        "foot": name,
                        "cycle": int(cycle),
                        "steps": int(in_cycle.sum()),
                        "period_s": float(in_cycle.sum() * dt),
                        "max_load_bw": float(load[in_cycle, env, k].max()),
                        "mean_load_bw": float(load[in_cycle, env, k].mean()),
                        "impulse_ns": float(fz[in_cycle, env, k].sum() * dt),
                        "frequency_hz": float(phase_frequency[in_cycle, env].mean()),
                        "vx": float(commands[in_cycle, env, 0].mean()),
                        "airborne_any": bool(
                            (load[in_cycle, env, :] < CONTACT_FRACTION_BW).all(1).any()
                        ),
                    }
                )

    summary = {
        "source": {
            "tag": tag,
            "checkpoint_sha256": meta["checkpoint_sha256"],
            "num_envs": meta["num_envs"],
            "steps": meta["steps"],
            "dt_s": dt,
            "contact_threshold_bw": CONTACT_FRACTION_BW,
            "gait_period_s_mean": float(np.median(cycle_length[cycle_length > 0]) * dt),
        },
        "counts": {
            "stance_events": len(events),
            "cycles": len({(c["env"], c["cycle"]) for c in per_cycle}),
            "per_foot_stance_events": {
                name: sum(1 for e in events if e["foot"] == name) for name in feet
            },
            "per_foot_loaded_events": {
                name: sum(1 for e in events if e["foot"] == name and e["loaded"])
                for name in feet
            },
            "loaded_definition": (
                "duration >= 5 control steps (0.1 s) and peak >= 0.30 bw"
            ),
        },
        "impulse_check": {},
        "per_foot_stance_peak": {},
        "per_foot_cycle_peak": {},
        "peak_by_speed": {},
        "peak_by_frequency": {},
        "peak_timing": {},
    }

    # impulse identity: a foot's cycle-mean force is bodyweight/4, so its
    # impulse per cycle should be bodyweight * period / 4.
    ratios = []
    for entry in per_cycle:
        expected = bodyweight[entry["env"]] * entry["period_s"] / 4.0
        ratios.append(entry["impulse_ns"] / expected)
    summary["impulse_check"] = {
        "per_cycle_impulse_over_bodyweight_period_quarter": quantiles(ratios),
        "note": "1.0 means each foot carries exactly a quarter of bodyweight on "
        "average over the cycle",
    }

    for name in feet:
        stance = [e for e in events if e["foot"] == name]
        loaded = [e for e in stance if e["loaded"]]
        cycle = [c for c in per_cycle if c["foot"] == name]
        peaks = np.array([e["peak_load_bw"] for e in stance])
        cycle_peaks = np.array([c["max_load_bw"] for c in cycle])
        loaded_peaks = np.array([e["peak_load_bw"] for e in loaded])
        summary["per_foot_stance_peak"][name] = {
            "count_all_runs": len(stance),
            "count_loaded": len(loaded),
            "peak_load_bw": quantiles(loaded_peaks),
            "peak_fz_n": quantiles([e["peak_fz_n"] for e in loaded]),
            "peak_norm_n": quantiles([e["peak_norm_n"] for e in loaded]),
            "duration_s": quantiles([e["duration_s"] for e in loaded]),
            "impulse_ns": quantiles([e["impulse_ns"] for e in loaded]),
            "fraction_above_0.8_bw": float((loaded_peaks > 0.8).mean()),
            "fraction_above_1.0_bw": float((loaded_peaks > 1.0).mean()),
            "fraction_above_1.2_bw": float((loaded_peaks > 1.2).mean()),
            "grazing_runs_peak_bw": quantiles(
                [e["peak_load_bw"] for e in stance if not e["loaded"]]
            ),
        }
        summary["per_foot_cycle_peak"][name] = {
            "count": len(cycle),
            "max_load_bw": quantiles(cycle_peaks),
            "mean_load_bw": quantiles([c["mean_load_bw"] for c in cycle]),
            "period_s": quantiles([c["period_s"] for c in cycle]),
            "within_env_std_bw": float(
                np.median(
                    [
                        np.std([c["max_load_bw"] for c in cycle if c["env"] == env])
                        for env in {c["env"] for c in cycle}
                        if sum(1 for c in cycle if c["env"] == env) > 2
                    ]
                )
            ),
            "fraction_cycles_with_any_airborne": float(
                np.mean([c["airborne_any"] for c in cycle])
            ),
        }

    def binned(key, edges, labels, values_key, limit=30):
        out = {}
        for lo, hi, label in zip(edges[:-1], edges[1:], labels):
            sel = [e for e in events if e["loaded"] and lo <= e[key] < hi]
            if len(sel) < limit:
                continue
            out[label] = {
                "stances": len(sel),
                "peak_load_bw_p95": float(
                    np.percentile([e[values_key] for e in sel], 95)
                ),
                "peak_load_bw_max": float(max(e[values_key] for e in sel)),
                "peak_load_bw_mean": float(np.mean([e[values_key] for e in sel])),
                "duration_s_mean": float(np.mean([e["duration_s"] for e in sel])),
                "vx_mean": float(np.mean([e["vx"] for e in sel])),
            }
        return out

    summary["peak_by_speed"] = binned(
        "vx",
        [-0.6, 0.1, 0.6, 1.2, 2.1],
        ["<0.1", "0.1-0.6", "0.6-1.2", ">1.2"],
        "peak_load_bw",
    )
    summary["peak_by_frequency"] = binned(
        "frequency_hz",
        [0.9, 1.5, 2.0, 2.5, 3.1],
        ["1.0-1.5", "1.5-2.0", "2.0-2.5", "2.5-3.0"],
        "peak_load_bw",
    )
    summary["peak_timing"] = {
        name: {
            "peak_phase_rad": quantiles(
                [
                    e["peak_phase_in_stance"]
                    for e in events
                    if e["foot"] == name and e["loaded"]
                ]
            ),
            "fraction_of_peak_in_second_half_of_stance": float(
                np.mean(
                    [
                        e["peak_phase_in_stance"] > np.pi / 2
                        for e in events
                        if e["foot"] == name and e["loaded"]
                    ]
                )
            ),
            "stance_fraction_outside_scheduled_stance": quantiles(
                [
                    e["stance_fraction_in_scheduled_swing"]
                    for e in events
                    if e["foot"] == name and e["loaded"]
                ]
            ),
        }
        for name in feet
    }

    (ART / "cycle_peaks.json").write_text(json.dumps(summary, indent=2) + "\n")

    # ── figures ────────────────────────────────────────────────────────────
    plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.25})

    def save(fig, name, dpi=100):
        fig.tight_layout()
        fig.savefig(FIG / name, dpi=dpi, pil_kwargs={"optimize": True})
        plt.close(fig)

    fig, (ax, axb) = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(0.0, 1.8, 73)
    for name in feet:
        peaks = [e["peak_load_bw"] for e in events if e["foot"] == name and e["loaded"]]
        ax.hist(peaks, bins=bins, histtype="step", lw=1.6, density=True, label=name)
    ax.set_xlabel("per-stance peak Fz / bodyweight")
    ax.set_ylabel("density")
    ax.set_title("Peak force per touchdown")
    ax.legend(fontsize=7)
    labels = feet
    p95 = [summary["per_foot_stance_peak"][n]["peak_load_bw"]["p95"] for n in labels]
    vmax = [summary["per_foot_stance_peak"][n]["peak_load_bw"]["max"] for n in labels]
    xpos = np.arange(len(labels))
    axb.bar(xpos - 0.18, p95, 0.36, label="p95")
    axb.bar(xpos + 0.18, vmax, 0.36, label="max")
    axb.set_xticks(xpos)
    axb.set_xticklabels(labels)
    axb.set_ylabel("peak Fz / bodyweight")
    axb.set_title("Touchdown peak: p95 and maximum")
    axb.legend(fontsize=7)
    save(fig, "fig8_touchdown_peak_force.png")

    fig, (ax, axb) = plt.subplots(1, 2, figsize=(11, 4))
    labels = list(summary["peak_by_speed"])
    ax.plot(
        range(len(labels)),
        [summary["peak_by_speed"][label]["peak_load_bw_p95"] for label in labels],
        "o-",
        label="p95",
    )
    ax.plot(
        range(len(labels)),
        [summary["peak_by_speed"][label]["peak_load_bw_mean"] for label in labels],
        "s--",
        label="mean",
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_xlabel("commanded vx (m/s)")
    ax.set_ylabel("peak Fz / bodyweight")
    ax.set_title("Touchdown peak vs speed")
    ax.legend(fontsize=7)
    flabels = list(summary["peak_by_frequency"])
    axb.plot(
        range(len(flabels)),
        [summary["peak_by_frequency"][label]["peak_load_bw_p95"] for label in flabels],
        "o-",
        label="p95",
    )
    axb.plot(
        range(len(flabels)),
        [summary["peak_by_frequency"][label]["duration_s_mean"] for label in flabels],
        "s--",
        label="stance duration (s)",
    )
    axb.set_xticks(range(len(flabels)))
    axb.set_xticklabels(flabels, rotation=15)
    axb.set_xlabel("gait frequency (Hz)")
    axb.set_ylabel("peak bw / duration s")
    axb.set_title("Touchdown peak and stance duration vs frequency")
    axb.legend(fontsize=7)
    save(fig, "fig9_peak_vs_speed_frequency.png")

    print(json.dumps(summary["counts"], indent=2))
    for name in feet:
        entry = summary["per_foot_stance_peak"][name]
        print(
            f"{name}: {entry['count_loaded']} loaded / "
            f"{entry['count_all_runs']} contact runs, peak bw mean "
            f"{entry['peak_load_bw']['mean']:.3f} p95 "
            f"{entry['peak_load_bw']['p95']:.3f} max "
            f"{entry['peak_load_bw']['max']:.3f}, >1.0 bw "
            f"{entry['fraction_above_1.0_bw']:.3f}"
        )
    print(
        "impulse/expected:",
        summary["impulse_check"]["per_cycle_impulse_over_bodyweight_period_quarter"],
    )


if __name__ == "__main__":
    main()
