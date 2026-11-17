"""Characterise go2trot foot contact forces over the locomotion cycle.

Consumes ``data/<tag>.npz`` written by ``collect_foot_contacts.py`` and writes
figures + ``artifacts/summary.json`` / ``artifacts/summary.md`` next to this
file.

Run from the product repo root:

    uv run --frozen python \
      docs/plans/supplements/foot_contact_study/analyze_foot_contacts.py
"""

import json
import pathlib

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent

from figure_io import save_figure  # noqa: E402

DATA = HERE / "data"
ART = HERE / "artifacts"
FIG = HERE / "figures"
G = 9.81

# go2trot ``control.gait_phase_offsets`` in cycles; see
# gym/envs/go2/go2trot_config.py. Leg phase = remainder(phase + 2*pi*offset).
PHASE_OFFSET_CYCLES = {
    "FL_foot": 0.0,
    "FR_foot": 0.5,
    "RL_foot": 0.5,
    "RR_foot": 0.0,
}
PHASE_BINS = 25
# Contact is defined relative to bodyweight, not in Newtons: a 1 N threshold
# counts the ~15 N foot "graze" during swing as contact.
CONTACT_FRACTION_BW = 0.05
TOUCH_FRACTION_BW = 0.005


def load_dataset(tag):
    data = np.load(DATA / f"{tag}.npz")
    meta = json.loads((DATA / f"{tag}.json").read_text())
    return data, meta


def shifted_phase(phase, lag):
    """Shift ``phase`` by ``lag`` rows.

    ``step()`` applies torques chosen from the previous control step's
    observation and advances the phase after the physics substep, so row t's
    force nominally belongs to row t-1's phase. Lag is a free parameter here
    because the mechanical contact lags the clock by a fraction of a step.
    """
    if lag == 0:
        return phase
    if lag > 0:
        return np.concatenate([phase[lag:], np.repeat(phase[-1:], lag, axis=0)])
    raise ValueError("lag must be >= 0")


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def bodyweight(data):
    return data["total_mass"] * G


def clean_mask(data):
    """Samples with no non-foot body in contact and no termination."""
    return (data["nonfoot_contact_norm"] < 1.0) & (data["terminated"] < 0.5)


def phase_profile(values, leg_phase, mask, bins=PHASE_BINS):
    """Mean/std/quantiles/count of ``values`` binned by leg phase."""
    edges = np.linspace(0.0, 2.0 * np.pi, bins + 1)
    idx = np.clip(np.digitize(leg_phase, edges) - 1, 0, bins - 1)
    valid = mask & np.isfinite(values)
    out = {
        "center": 0.5 * (edges[1:] + edges[:-1]),
        "mean": np.full(bins, np.nan),
        "std": np.full(bins, np.nan),
        "p10": np.full(bins, np.nan),
        "p90": np.full(bins, np.nan),
        "count": np.zeros(bins, dtype=int),
    }
    for b in range(bins):
        sel = valid & (idx == b)
        sample = values[sel]
        out["count"][b] = sample.size
        if sample.size == 0:
            continue
        out["mean"][b] = sample.mean()
        out["std"][b] = sample.std()
        out["p10"][b] = np.percentile(sample, 10)
        out["p90"][b] = np.percentile(sample, 90)
    return out


def ridge_fit_predict(x_train, y_train, x_test, lam):
    """Standardise, centre and solve ridge in closed form (numpy only)."""
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-8] = 1.0
    xs = (x_train - mean) / std
    xt = (x_test - mean) / std
    y_mean = y_train.mean(axis=0)
    gram = xs.T @ xs
    gram[np.diag_indices_from(gram)] += lam * len(xs)
    coef = np.linalg.solve(gram, xs.T @ (y_train - y_mean))
    return xt @ coef + y_mean


def random_features(x, units, seed):
    """Fixed tanh random projection for a cheap nonlinear probe."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 1.0 / np.sqrt(x.shape[1]), size=(x.shape[1], units))
    b = rng.normal(0.0, 1.0, size=units)
    return np.tanh(x @ w + b)


def r2_score(y_true, y_pred):
    resid = np.square(y_true - y_pred).sum(axis=0)
    var = np.square(y_true - y_true.mean(axis=0)).sum(axis=0)
    return 1.0 - resid / var


def main():
    ART.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)

    data, meta = load_dataset("gpu")
    if meta["resets"]:
        raise RuntimeError("recollect without mid-rollout resets for phase analysis")
    feet = meta["feet_names"]
    masses = bodyweight(data)
    dt = float(data["dt"])
    forces = data["foot_forces"].astype(np.float64)
    fz = forces[..., 2]
    load = fz / masses[None, :, None]
    phase = data["phase"].astype(np.float64)
    clean = clean_mask(data)
    freq = data["phase_frequency"].astype(np.float64)
    vx = data["commands"][..., 0].astype(np.float64)
    n_steps, n_envs = fz.shape[:2]
    n_feet = len(feet)

    def leg_phases(lag):
        shifted = shifted_phase(phase, lag)
        return np.stack(
            [
                np.mod(shifted + 2.0 * np.pi * PHASE_OFFSET_CYCLES[name], 2.0 * np.pi)
                for name in feet
            ],
            axis=-1,
        )

    # ── phase alignment: sweep whole control steps ─────────────────────────
    loaded = load > CONTACT_FRACTION_BW
    alignment = {}
    for lag in range(0, 4):
        lp = leg_phases(lag)
        stance = np.sin(lp) > 0.0
        clean4 = clean[..., None]
        alignment[str(lag)] = {
            "contact_given_scheduled_stance": float(loaded[stance & clean4].mean()),
            "contact_given_scheduled_swing": float(loaded[(~stance) & clean4].mean()),
            "separation": float(
                loaded[stance & clean4].mean() - loaded[(~stance) & clean4].mean()
            ),
        }
    best_lag = max(alignment, key=lambda k: alignment[k]["separation"])

    # Sub-step phase offset: rotate the leg-phase clock in radians and find the
    # offset that best separates scheduled stance from scheduled swing.
    clean4 = clean[..., None]
    shifts = np.linspace(-np.pi, np.pi, 49)
    sweep = []
    for shift in shifts:
        lp = np.mod(leg_phases(0) + shift, 2.0 * np.pi)
        stance = np.sin(lp) > 0.0
        separation = float(
            loaded[stance & clean4].mean() - loaded[(~stance) & clean4].mean()
        )
        sweep.append(separation)
    sweep = np.asarray(sweep)
    best_shift = float(shifts[int(np.argmax(sweep))])
    cycle_seconds = 1.0 / float(freq[clean].mean())
    clock_offset = {
        "best_phase_shift_rad": best_shift,
        "best_separation": float(sweep.max()),
        "separation_at_zero": float(sweep[np.argmin(np.abs(shifts))]),
        "equivalent_time_ms": float(best_shift / (2 * np.pi) * cycle_seconds * 1e3),
        "mean_cycle_s": cycle_seconds,
        "sweep_rad": shifts.tolist(),
        "sweep_separation": sweep.tolist(),
    }
    LAG = 0  # causal convention: the force of row t follows row t-1's command
    leg_phase = leg_phases(LAG)
    scheduled_stance = np.sin(leg_phase) > 0.0

    summary = {
        "source": {
            "tag": meta["tag"],
            "backend": meta["backend"],
            "device": meta["device"],
            "checkpoint": meta["checkpoint_path"],
            "checkpoint_sha256": meta["checkpoint_sha256"],
            "num_envs": meta["num_envs"],
            "steps": meta["steps"],
            "dt_s": dt,
            "control_hz": 1.0 / dt,
            "episode_length_s": meta["episode_length_s"],
            "command_resampling_s": meta["command_resampling_s"],
            "push_robots": meta["push_robots"],
            "feet": feet,
            "body_mass_kg": {
                "mean": float(data["total_mass"].mean()),
                "std": float(data["total_mass"].std()),
            },
            "bodyweight_n": float(masses.mean()),
            "contact_threshold_bw": CONTACT_FRACTION_BW,
            "contact_threshold_n_mean": float(CONTACT_FRACTION_BW * masses.mean()),
        },
        "samples": {
            "total": int(n_steps * n_envs),
            "clean_fraction": float(clean.mean()),
            "nonfoot_contact_fraction": float(
                (data["nonfoot_contact_norm"] >= 1.0).mean()
            ),
        },
        "phase_alignment": {
            "causal_lag_used": LAG,
            "best_separation_lag": int(best_lag),
            "by_lag": alignment,
            "sub_step_offset": clock_offset,
            "note": "lag is in control steps (20 ms); best-separation lag can be "
            "offset by mechanical contact lag rather than reversed causality",
        },
        "force_frame_check": {},
        "per_foot": {},
        "contact_schedule": {},
        "support": {},
        "reward_terms": {},
        "command_dependence": {},
        "frequency_dependence": {},
        "candidate_reward": {},
        "observability": {},
    }

    # ── force vector / vertical-fraction checks ────────────────────────────
    fnorm = np.linalg.norm(forces, axis=-1)
    fy_frac = (np.abs(forces[..., 2]) / np.maximum(fnorm, 1e-9))[fnorm > 1.0]
    summary["force_frame_check"] = {
        "vertical_fraction_of_force_norm": {
            "mean": float(fy_frac.mean()),
            "p05": float(np.percentile(fy_frac, 5)),
            "p50": float(np.percentile(fy_frac, 50)),
        },
        "max_horizontal_over_vertical": float(
            (np.linalg.norm(forces[..., :2], axis=-1) / np.maximum(np.abs(fz), 1e-6))[
                np.abs(fz) > 10.0
            ].max()
        ),
        "negative_fz_samples": int((fz < -0.5).sum()),
        "total_fz_over_bodyweight_mean": float(
            (fz.sum(-1) / masses[None, :])[clean].mean()
        ),
    }

    # ── per-foot statistics ────────────────────────────────────────────────
    for i, name in enumerate(feet):
        stance = scheduled_stance[:, :, i]
        contact = loaded[:, :, i]
        stance_load = load[:, :, i][stance & clean]
        swing_load = load[:, :, i][(~stance) & clean]
        summary["per_foot"][name] = {
            "fz_n": {
                "mean": float(fz[:, :, i][clean].mean()),
                "p95": float(np.percentile(fz[:, :, i][clean], 95)),
                "p99": float(np.percentile(fz[:, :, i][clean], 99)),
                "max": float(fz[:, :, i][clean].max()),
            },
            "load_bw_scheduled_stance": {
                "mean": float(stance_load.mean()),
                "p10": float(np.percentile(stance_load, 10)),
                "p50": float(np.percentile(stance_load, 50)),
                "p90": float(np.percentile(stance_load, 90)),
                "p99": float(np.percentile(stance_load, 99)),
            },
            "load_bw_scheduled_swing": {
                "mean": float(swing_load.mean()),
                "p90": float(np.percentile(swing_load, 90)),
                "p95": float(np.percentile(swing_load, 95)),
                "max": float(swing_load.max()),
            },
            "touch_fraction": float((load[:, :, i] > TOUCH_FRACTION_BW)[clean].mean()),
            "duty_factor": float(contact[clean].mean()),
            "scheduled_duty": float(stance[clean].mean()),
            "contact_given_scheduled_stance": float(contact[stance & clean].mean()),
            "contact_given_scheduled_swing": float(contact[(~stance) & clean].mean()),
        }

    # ── contact schedule as a function of leg phase ────────────────────────
    profile_edges = np.linspace(0.0, 2.0 * np.pi, PHASE_BINS + 1)
    for i, name in enumerate(feet):
        contact_prob = phase_profile(
            loaded[:, :, i].astype(np.float64), leg_phase[:, :, i], clean
        )
        strength = phase_profile(smoothstep(load[:, :, i]), leg_phase[:, :, i], clean)
        mean_load = phase_profile(load[:, :, i], leg_phase[:, :, i], clean)
        summary["contact_schedule"][name] = {
            "phase_edges": profile_edges.tolist(),
            "phase_centers": contact_prob["center"].tolist(),
            "contact_probability": contact_prob["mean"].tolist(),
            "load_mean_bw": mean_load["mean"].tolist(),
            "load_p10_bw": mean_load["p10"].tolist(),
            "load_p90_bw": mean_load["p90"].tolist(),
            "contact_strength_mean": strength["mean"].tolist(),
            "cycle_fraction_mean_load_above_0.1bw": float(
                np.nanmean(mean_load["mean"] >= 0.1)
            ),
            "phase_of_peak_mean_load": float(
                mean_load["center"][int(np.nanargmax(mean_load["mean"]))]
            ),
        }

    # ── total support and sharing ──────────────────────────────────────────
    support = load.sum(axis=-1)
    diag_a = load[:, :, 0] + load[:, :, 3]  # FL + RR
    diag_b = load[:, :, 1] + load[:, :, 2]  # FR + RL
    front = load[:, :, 0] + load[:, :, 1]
    rear = load[:, :, 2] + load[:, :, 3]
    airborne = (load < CONTACT_FRACTION_BW).all(axis=-1)
    summary["support"] = {
        "total_load_bw": {
            "mean": float(support[clean].mean()),
            "std": float(support[clean].std()),
            "p05": float(np.percentile(support[clean], 5)),
            "p50": float(np.percentile(support[clean], 50)),
            "p95": float(np.percentile(support[clean], 95)),
            "fraction_below_0.8": float((support[clean] < 0.8).mean()),
            "fraction_above_1.2": float((support[clean] > 1.2).mean()),
        },
        "airborne_fraction_all_feet_below_threshold": float(airborne[clean].mean()),
        "mean_fz_over_bodyweight_per_foot": {
            name: float(load[:, :, i][clean].mean()) for i, name in enumerate(feet)
        },
        "stance_mean_fz_over_bodyweight_per_foot": {
            name: summary["per_foot"][name]["load_bw_scheduled_stance"]["mean"]
            for name in feet
        },
        "front_pair_mean_bw": float(front[clean].mean()),
        "rear_pair_mean_bw": float(rear[clean].mean()),
        "diagonal_FL_RR_mean_bw": float(diag_a[clean].mean()),
        "diagonal_FR_RL_mean_bw": float(diag_b[clean].mean()),
        "diagonal_instantaneous_abs_difference_bw": float(
            np.abs(diag_a - diag_b)[clean].mean()
        ),
        "diagonal_instantaneous_abs_difference_note": (
            "a trot alternates diagonals, so the instantaneous difference is "
            "large by design; the two cycle means above are the sharing check"
        ),
        "corr_total_load_minus_1_with_base_vz_change": float(
            np.corrcoef(
                (support[:-1] - 1.0).ravel(),
                (np.diff(data["base_lin_vel"][..., 2], axis=0) / dt).ravel(),
            )[0, 1]
        ),
    }

    # ── the task's own contact terms, recomputed from recorded force ───────
    strength_all = smoothstep(load)
    stance_strength = np.where(scheduled_stance, strength_all, 1.0)
    paired = np.clip(2.0 * np.sqrt(np.prod(stance_strength, axis=-1)), 0.0, 1.0)
    trot_support = np.maximum(np.sin(leg_phase), 0.0).max(axis=-1) * paired
    swing_contact = -np.mean(
        np.maximum(-np.sin(leg_phase), 0.0) * strength_all, axis=-1
    )
    summary["reward_terms"] = {
        "trot_support": {
            "mean": float(trot_support[clean].mean()),
            "p10": float(np.percentile(trot_support[clean], 10)),
            "p50": float(np.percentile(trot_support[clean], 50)),
            "p90": float(np.percentile(trot_support[clean], 90)),
        },
        "swing_contact": {
            "mean": float(swing_contact[clean].mean()),
            "p10": float(np.percentile(swing_contact[clean], 10)),
            "p50": float(np.percentile(swing_contact[clean], 50)),
            "p90": float(np.percentile(swing_contact[clean], 90)),
        },
        "weighted_per_step": {
            "trot_support_w3": float(3.0 * dt * trot_support[clean].mean()),
            "swing_contact_w3": float(3.0 * dt * swing_contact[clean].mean()),
        },
    }

    # ── stance compliance / swing violation rates ──────────────────────────
    compliance_levels = [0.05, 0.1, 0.2, 0.3, 0.5]
    summary["constraint_rates"] = {
        "stance_load_above": {
            f"{level:.2f}_bw": float(
                (load[scheduled_stance & clean[..., None]] >= level).mean()
            )
            for level in compliance_levels
        },
        "swing_load_above": {
            f"{level:.2f}_bw": float(
                (load[(~scheduled_stance) & clean[..., None]] >= level).mean()
            )
            for level in compliance_levels
        },
        "per_foot_stance_load_above_0.2_bw": {
            name: float(
                (load[:, :, i][scheduled_stance[:, :, i] & clean] >= 0.2).mean()
            )
            for i, name in enumerate(feet)
        },
        "per_foot_swing_load_above_0.1_bw": {
            name: float(
                (load[:, :, i][(~scheduled_stance[:, :, i]) & clean] >= 0.1).mean()
            )
            for i, name in enumerate(feet)
        },
    }

    # ── stance-window width: how much foot impulse falls outside the clock ─
    window_sweep = {}
    all_feet_clean = np.broadcast_to(clean[..., None], fz.shape)
    for width in (0.50, 0.55, 0.60, 0.65, 0.70):
        delta = np.abs(np.mod(leg_phase - np.pi / 2.0 + np.pi, 2.0 * np.pi) - np.pi)
        inside = (delta <= width * np.pi) & clean[..., None]
        window_sweep[f"{width:.2f}"] = {
            "samples_inside_fraction": float(inside.mean()),
            "contact_precision_inside": float(loaded[inside].mean()),
            "impulse_inside_fraction": float(
                fz[inside].sum() / fz[all_feet_clean].sum()
            ),
            "contact_impulse_outside_fraction": float(
                fz[loaded & all_feet_clean & ~inside].sum() / fz[all_feet_clean].sum()
            ),
        }
    summary["schedule_window_sweep"] = {
        "definition": "stance window centred on the clock's stance half with "
        "total duty `width`, measured as the mean over the four legs",
        "by_duty": window_sweep,
    }

    # ── dependence on commanded speed and gait frequency ───────────────────
    speed_edges = [-0.6, 0.1, 0.6, 1.2, 2.1]
    speed_labels = ["<0.1", "0.1-0.6", "0.6-1.2", ">1.2"]
    for lo, hi, label in zip(speed_edges[:-1], speed_edges[1:], speed_labels):
        sel = clean & (vx >= lo) & (vx < hi)
        if sel.sum() < 100:
            continue
        summary["command_dependence"][label] = {
            "samples": int(sel.sum()),
            "total_load_bw_mean": float(support[sel].mean()),
            "per_foot_load_mean": {
                name: float(load[:, :, i][sel].mean()) for i, name in enumerate(feet)
            },
            "per_foot_load_p95": {
                name: float(np.percentile(load[:, :, i][sel], 95))
                for i, name in enumerate(feet)
            },
            "duty_factor_mean": float(loaded[sel].mean()),
            "airborne_fraction": float(airborne[sel].mean()),
            "gait_frequency_hz_mean": float(freq[sel].mean()),
        }

    freq_edges = [0.9, 1.5, 2.0, 2.5, 3.1]
    freq_labels = ["1.0-1.5", "1.5-2.0", "2.0-2.5", "2.5-3.0"]
    for lo, hi, label in zip(freq_edges[:-1], freq_edges[1:], freq_labels):
        sel = clean & (freq >= lo) & (freq < hi)
        if sel.sum() < 100:
            continue
        summary["frequency_dependence"][label] = {
            "samples": int(sel.sum()),
            "duty_factor_mean": float(loaded[sel].mean()),
            "per_foot_load_p95_mean": float(
                np.mean([np.percentile(load[:, :, i][sel], 95) for i in range(n_feet)])
            ),
            "total_load_bw_mean": float(support[sel].mean()),
            "airborne_fraction": float(airborne[sel].mean()),
            "steps_per_cycle": float(1.0 / (freq[sel].mean() * dt)),
        }

    # ── candidate contact-constraint reward ────────────────────────────────
    # Bounded to [0, 1]; schedule-aware because the phase clock is an actor
    # observation, so a Q at deployment can evaluate the same expression.
    stance_score = np.mean(scheduled_stance * smoothstep(load / 0.25), axis=-1)
    swing_score = np.mean((~scheduled_stance) * smoothstep(load / 0.10), axis=-1)
    r_prime = 0.5 * stance_score + 0.5 * (1.0 - swing_score)
    grid = {}
    for target in (0.15, 0.25, 0.35):
        ss = np.mean(scheduled_stance * smoothstep(load / target), axis=-1)
        for tol in (0.05, 0.10, 0.20):
            sw = np.mean((~scheduled_stance) * smoothstep(load / tol), axis=-1)
            rp = 0.5 * ss + 0.5 * (1.0 - sw)
            grid[f"target={target},tol={tol}"] = {
                "reward_mean": float(rp[clean].mean()),
                "reward_std": float(rp[clean].std()),
                "reward_p10": float(np.percentile(rp[clean], 10)),
                "reward_p50": float(np.percentile(rp[clean], 50)),
                "reward_p90": float(np.percentile(rp[clean], 90)),
                "fraction_below_0.5": float((rp[clean] < 0.5).mean()),
                "fraction_above_0.9": float((rp[clean] > 0.9).mean()),
            }
    variant_target, variant_tol = 0.20, 0.10
    stance_deficit = np.mean(
        scheduled_stance * np.maximum(variant_target - load, 0.0), axis=-1
    )
    swing_overload = np.mean(
        (~scheduled_stance) * np.maximum(load - variant_tol, 0.0), axis=-1
    )
    variants = {
        "linear_penalty": 1.0
        - np.clip(stance_deficit / 0.25 + swing_overload / 0.10, 0.0, 1.0),
        "exp_penalty": np.exp(
            -np.square(stance_deficit / 0.25) - np.square(swing_overload / 0.10)
        ),
        "schedule_smoothstep": r_prime,
    }
    variant_stats = {
        name: {
            "mean": float(value[clean].mean()),
            "std": float(value[clean].std()),
            "p05": float(np.percentile(value[clean], 5)),
            "p50": float(np.percentile(value[clean], 50)),
            "p95": float(np.percentile(value[clean], 95)),
        }
        for name, value in variants.items()
    }
    summary["candidate_reward"] = {
        "variants": {
            "params": {"stance_target_bw": variant_target, "swing_tol_bw": variant_tol},
            "definitions": {
                "linear_penalty": (
                    "1 - clip(stance_deficit/0.25 + swing_overload/0.10, 0, 1)"
                ),
                "exp_penalty": (
                    "exp(-(stance_deficit/0.25)^2 - (swing_overload/0.10)^2)"
                ),
                "schedule_smoothstep": (
                    "0.5*stance_load_score + 0.5*(1 - swing_load_score)"
                ),
            },
            "stats": variant_stats,
        },
        "form": (
            "r' = 0.5*mean_legs[stance * smoothstep(load/target)] "
            "+ 0.5*(1 - mean_legs[swing * smoothstep(load/tol)])"
        ),
        "selected": {"target_bw": 0.25, "tol_bw": 0.10},
        "selected_stats": {
            "mean": float(r_prime[clean].mean()),
            "std": float(r_prime[clean].std()),
            "p05": float(np.percentile(r_prime[clean], 5)),
            "p50": float(np.percentile(r_prime[clean], 50)),
            "p95": float(np.percentile(r_prime[clean], 95)),
            "fraction_below_0.3": float((r_prime[clean] < 0.3).mean()),
        },
        "grid": grid,
    }

    # ── observability probe: can the actor observation explain the force? ──
    obs = data["actor_obs"].astype(np.float64)
    action = data["dof_pos_target"].astype(np.float64)
    shift = lambda a: np.concatenate([a[1:], a[-1:]], axis=0)  # noqa: E731
    target_force = shift(load)
    target_contact = shift(loaded.astype(np.float64))
    target_support = shift(support).reshape(n_steps, n_envs, 1)
    target_reward = shift(r_prime).reshape(n_steps, n_envs, 1)
    win = clean & shift(clean)
    flat = lambda a: a.reshape(-1, *a.shape[2:])  # noqa: E731
    full_state = np.concatenate(
        [
            data["dof_pos"],
            data["dof_vel"],
            data["torques"],
            data["root_states"],
            np.stack([np.sin(phase), np.cos(phase)], axis=-1),
            freq[..., None],
            data["commands"],
        ],
        axis=-1,
    ).astype(np.float64)
    feature_sets = {
        "actor_obs": flat(obs),
        "actor_obs_plus_action": flat(np.concatenate([obs, action], axis=-1)),
        "full_simulator_state": flat(full_state),
    }
    targets = {
        "foot_load_bw": flat(target_force),
        "foot_contact": flat(target_contact),
        "total_load_bw": flat(target_support),
        "candidate_reward": flat(target_reward),
    }
    train_env = np.arange(n_envs) < int(round(0.8 * n_envs))
    env_ids = np.repeat(np.arange(n_envs), n_steps)
    train_rows = win.reshape(-1) & np.isin(env_ids, np.flatnonzero(train_env))
    test_rows = win.reshape(-1) & ~np.isin(env_ids, np.flatnonzero(train_env))

    probe = {}
    for fname, features in feature_sets.items():
        lin = {}
        nonlin = {}
        for tname, y in targets.items():
            lin[tname] = float(
                r2_score(
                    y[test_rows],
                    ridge_fit_predict(
                        features[train_rows], y[train_rows], features[test_rows], 1e-3
                    ),
                ).mean()
            )
            nonlin[tname] = float(
                r2_score(
                    y[test_rows],
                    ridge_fit_predict(
                        random_features(features[train_rows], 256, seed=0),
                        y[train_rows],
                        random_features(features[test_rows], 256, seed=0),
                        1e-2,
                    ),
                ).mean()
            )
        probe[fname] = {"linear_r2": lin, "random_feature_r2": nonlin}

    # phase-resolved observability of the per-foot load (first foot)
    best = random_features(feature_sets["actor_obs_plus_action"], 256, seed=0)
    best_pred = ridge_fit_predict(
        best[train_rows], targets["foot_load_bw"][train_rows], best[test_rows], 1e-2
    )
    phase_test = leg_phase[:, :, 0].reshape(-1)[test_rows]
    phase_r2 = {}
    edges = np.linspace(0, 2 * np.pi, 9)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (phase_test >= lo) & (phase_test < hi)
        if sel.sum() < 50:
            continue
        phase_r2[f"{lo:.2f}-{hi:.2f}"] = float(
            r2_score(
                targets["foot_load_bw"][test_rows][sel][:, :1],
                best_pred[sel][:, :1],
            )[0]
        )
    variant_r2 = {}
    for name, value in variants.items():
        y_variant = shift(value).reshape(-1, 1)
        inference = random_features(feature_sets["actor_obs_plus_action"], 256, seed=0)
        pred_variant = ridge_fit_predict(
            inference[train_rows], y_variant[train_rows], inference[test_rows], 1e-2
        )
        variant_r2[name] = {
            "random_feature_r2": float(r2_score(y_variant[test_rows], pred_variant)[0]),
            "random_feature_mae": float(
                np.abs(y_variant[test_rows] - pred_variant).mean()
            ),
        }

    summary["observability"] = {
        "split": "80/20 by env_id (held-out worlds)",
        "train_rows": int(train_rows.sum()),
        "test_rows": int(test_rows.sum()),
        "target_alignment": "row t obs/action predict row t+1 force (post-action)",
        "probe": probe,
        "candidate_reward_variants": variant_r2,
        "fl_load_r2_by_phase_bin": phase_r2,
    }

    (ART / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # ── figures ────────────────────────────────────────────────────────────
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 100,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )

    def save(fig, name, dpi=200):
        # High-dpi render, then shrink only if the file exceeds the commit
        # limit (figure_io prints the effective resolution).
        save_figure(fig, FIG / name, dpi=dpi)

    tick_labels = ["0", "pi/2", "pi", "3pi/2", "2pi"]
    ticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]

    # fig 1: vertical load vs leg phase
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for i, (name, ax) in enumerate(zip(feet, axes.ravel())):
        prof = phase_profile(load[:, :, i], leg_phase[:, :, i], clean)
        c = prof["center"]
        ax.fill_between(c, prof["p10"], prof["p90"], alpha=0.25, label="p10-p90")
        ax.plot(c, prof["mean"], lw=2, label="mean")
        ax.axvspan(0.0, np.pi, color="tab:green", alpha=0.07, label="scheduled stance")
        ax.axhline(0.25, color="k", ls=":", lw=1, label="0.25 bw")
        ax.set_title(name)
        ax.set_xlim(0, 2 * np.pi)
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)
        ax.set_ylabel("Fz / bodyweight")
    axes[0, 0].legend(loc="upper right", fontsize=7)
    fig.suptitle(
        "Vertical foot load over the gait cycle (green = scheduled stance sin>0, lag 0)"
    )
    save(fig, "fig1_vertical_load_vs_phase.png")

    # fig 2: contact probability + duty factor
    fig, (ax, axb) = plt.subplots(
        1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [3, 2]}
    )
    duty = []
    for i, name in enumerate(feet):
        prof = phase_profile(
            loaded[:, :, i].astype(np.float64), leg_phase[:, :, i], clean
        )
        ax.plot(prof["center"], prof["mean"], lw=2, label=name)
        duty.append(summary["per_foot"][name]["duty_factor"])
    ax.axhline(0.5, color="k", ls=":", lw=1, label="scheduled duty (0.5)")
    ax.set_xlabel("leg phase (rad)")
    ax.set_ylabel(f"P(load > {CONTACT_FRACTION_BW:.0%} bodyweight)")
    ax.set_title("Contact probability by leg phase")
    ax.set_xlim(0, 2 * np.pi)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.legend(fontsize=7)
    axb.bar(range(n_feet), duty, color="tab:blue")
    axb.axhline(0.5, color="k", ls=":", lw=1)
    axb.set_xticks(range(n_feet))
    axb.set_xticklabels(feet)
    axb.set_ylim(0, 1)
    axb.set_ylabel("duty factor")
    axb.set_title("Measured duty factor")
    save(fig, "fig2_contact_probability.png")

    # fig 3: distributions
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(0.0, 1.6, 65)
    for i, name in enumerate(feet):
        ax.hist(
            load[:, :, i][clean],
            bins=bins,
            histtype="step",
            lw=1.5,
            density=True,
            label=name,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Fz / bodyweight (all samples)")
    ax.set_ylabel("density (log)")
    ax.set_title("Per-foot vertical load distribution")
    ax.legend(fontsize=7)
    axb.hist(
        support[clean], bins=np.linspace(0.0, 2.4, 73), color="tab:purple", alpha=0.85
    )
    axb.axvline(1.0, color="k", ls="--", lw=1)
    axb.set_xlabel("sum(Fz) / bodyweight")
    axb.set_ylabel("samples")
    airborne = summary["support"]["airborne_fraction_all_feet_below_threshold"]
    axb.set_title(f"Total vertical force; airborne {airborne:.1%}")
    save(fig, "fig3_force_distribution.png")

    # fig 4: support timing and sharing
    fig, (ax, axb) = plt.subplots(
        1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [3, 2]}
    )
    prof_tot = phase_profile(support, leg_phase[:, :, 0], clean)
    ax.plot(prof_tot["center"], prof_tot["mean"], lw=2, label="all four feet")
    ax.fill_between(prof_tot["center"], prof_tot["p10"], prof_tot["p90"], alpha=0.2)
    prof_a = phase_profile(diag_a, leg_phase[:, :, 0], clean)
    prof_b = phase_profile(diag_b, leg_phase[:, :, 0], clean)
    ax.plot(prof_a["center"], prof_a["mean"], lw=1.5, label="FL+RR diagonal")
    ax.plot(prof_b["center"], prof_b["mean"], lw=1.5, label="FR+RL diagonal")
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_xlabel("leg phase of FL (rad)")
    ax.set_ylabel("Fz / bodyweight")
    ax.set_title("Support sharing over the cycle")
    ax.set_xlim(0, 2 * np.pi)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.legend(fontsize=7)
    means = [summary["support"]["mean_fz_over_bodyweight_per_foot"][n] for n in feet]
    axb.bar(range(n_feet), means, color="tab:orange")
    axb.set_xticks(range(n_feet))
    axb.set_xticklabels(feet)
    axb.set_ylabel("mean Fz / bodyweight")
    axb.set_title("Mean share per foot")
    save(fig, "fig4_support_timing.png")

    # fig 5: command and frequency dependence
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    labels = list(summary["command_dependence"])
    axes[0].bar(
        range(len(labels)),
        [
            summary["command_dependence"][label]["total_load_bw_mean"]
            for label in labels
        ],
    )
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=20)
    axes[0].axhline(1.0, color="k", ls=":", lw=1)
    axes[0].set_ylabel("mean sum(Fz) / bodyweight")
    axes[0].set_title("Total support vs commanded vx (m/s)")
    axes[1].bar(
        range(len(labels)),
        [
            float(
                np.mean(
                    list(
                        summary["command_dependence"][label][
                            "per_foot_load_p95"
                        ].values()
                    )
                )
            )
            for label in labels
        ],
    )
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=20)
    axes[1].set_ylabel("p95 Fz / bodyweight")
    axes[1].set_title("Per-foot load tail vs commanded vx")
    flabels = list(summary["frequency_dependence"])
    axes[2].plot(
        range(len(flabels)),
        [
            summary["frequency_dependence"][label]["duty_factor_mean"]
            for label in flabels
        ],
        "o-",
        label="duty",
    )
    axes[2].plot(
        range(len(flabels)),
        [
            summary["frequency_dependence"][label]["airborne_fraction"]
            for label in flabels
        ],
        "s--",
        label="airborne",
    )
    axes[2].set_xticks(range(len(flabels)))
    axes[2].set_xticklabels(flabels, rotation=20)
    axes[2].set_ylabel("fraction of samples")
    axes[2].set_title("Duty / flight vs gait frequency (Hz)")
    axes[2].legend(fontsize=7)
    save(fig, "fig5_command_effects.png")

    # fig 6: observability
    fig, (ax, axb) = plt.subplots(
        1, 2, figsize=(11.5, 4), gridspec_kw={"width_ratios": [2, 3]}
    )
    tnames = list(targets)
    xpos = np.arange(len(tnames))
    width = 0.26
    for j, fname in enumerate(feature_sets):
        vals = [probe[fname]["random_feature_r2"][n] for n in tnames]
        ax.bar(xpos + (j - 1) * width, vals, width, label=fname)
    ax.set_xticks(xpos)
    ax.set_xticklabels(["foot load", "contact", "total load", "r'"], rotation=10)
    ax.set_ylabel("held-out R²")
    ax.set_ylim(min(0.0, ax.get_ylim()[0]), 1.0)
    ax.set_title("Nonlinear probe from state")
    ax.legend(fontsize=7)
    y_for = targets["foot_load_bw"][test_rows][:, 0]
    pred = best_pred[:, 0]
    axb.hexbin(y_for, pred, gridsize=40, bins="log", mincnt=1)
    lim = [0, max(y_for.max(), pred.max())]
    axb.plot(lim, lim, "r--", lw=1)
    axb.set_xlabel("actual FL load / bodyweight")
    axb.set_ylabel("predicted")
    axb.set_title("FL load from actor obs + action, held-out worlds")
    save(fig, "fig6_observability.png")

    # fig 7: candidate reward profile and histogram
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(11, 4))
    prof_r = phase_profile(r_prime, leg_phase[:, :, 0], clean)
    ax.plot(prof_r["center"], prof_r["mean"], lw=2)
    ax.fill_between(prof_r["center"], prof_r["p10"], prof_r["p90"], alpha=0.2)
    ax.set_xlabel("leg phase of FL (rad)")
    ax.set_ylabel("r'")
    ax.set_title("Candidate contact reward over the cycle")
    ax.set_xlim(0, 2 * np.pi)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    axb.hist(r_prime[clean], bins=np.linspace(0, 1, 51), color="tab:green", alpha=0.85)
    axb.set_xlabel("r'")
    axb.set_ylabel("samples")
    axb.set_title("Candidate reward distribution")
    save(fig, "fig7_candidate_reward.png")

    # ── CPU/Warp agreement (different random worlds, distribution level) ───
    try:
        cpu_data, cpu_meta = load_dataset("cpu")
        cpu_clean = clean_mask(cpu_data)
        cpu_load = cpu_data["foot_forces"][..., 2] / bodyweight(cpu_data)[None, :, None]
        cpu_phase_cmd = shifted_phase(cpu_data["phase"], LAG)
        cpu_leg = np.mod(cpu_phase_cmd, 2.0 * np.pi)
        profile_cpu = phase_profile(cpu_load[:, :, 0], cpu_leg, cpu_clean)
        profile_gpu = phase_profile(load[:, :, 0], leg_phase[:, :, 0], clean)
        support_cpu = cpu_load.sum(-1)
        summary["backend_agreement"] = {
            "cpu": {
                "num_envs": cpu_meta["num_envs"],
                "steps": cpu_meta["steps"],
                "checkpoint_sha256": cpu_meta["checkpoint_sha256"],
                "fl_duty": float(
                    (cpu_load[:, :, 0] > CONTACT_FRACTION_BW)[cpu_clean].mean()
                ),
                "fl_stance_load_bw": float(
                    cpu_load[:, :, 0][(np.sin(cpu_leg) > 0) & cpu_clean].mean()
                ),
                "total_load_bw_mean": float(support_cpu[cpu_clean].mean()),
                "airborne_fraction": float(
                    (cpu_load < CONTACT_FRACTION_BW).all(-1)[cpu_clean].mean()
                ),
            },
            "gpu": {
                "num_envs": meta["num_envs"],
                "steps": meta["steps"],
                "fl_duty": summary["per_foot"]["FL_foot"]["duty_factor"],
                "fl_stance_load_bw": summary["per_foot"]["FL_foot"][
                    "load_bw_scheduled_stance"
                ]["mean"],
                "total_load_bw_mean": summary["support"]["total_load_bw"]["mean"],
                "airborne_fraction": summary["support"][
                    "airborne_fraction_all_feet_below_threshold"
                ],
            },
            "fl_phase_curve_max_abs_diff_bw": float(
                np.nanmax(np.abs(profile_cpu["mean"] - profile_gpu["mean"]))
            ),
            "fl_phase_curve_mean_abs_diff_bw": float(
                np.nanmean(np.abs(profile_cpu["mean"] - profile_gpu["mean"]))
            ),
            "note": (
                "different randomized worlds/masses per backend; "
                "distribution-level check"
            ),
        }
    except FileNotFoundError:
        summary["backend_agreement"] = None

    (ART / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (ART / "summary.md").write_text(build_markdown(summary, feet) + "\n")
    print((ART / "summary.md").read_text())


def build_markdown(summary, feet):
    lines = ["# Foot contact study — generated tables", ""]
    lines += [
        "## Per-foot summary",
        "",
        "| foot | mean Fz (N) | p99 Fz (N) | max Fz (N) | duty | stance mean (bw) "
        "| stance p90 (bw) | swing p95 (bw) | P(contact|stance) | P(contact|swing) "
        "| peak load phase (rad) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in feet:
        s = summary["per_foot"][name]
        cs = summary["contact_schedule"][name]
        lines.append(
            f"| {name} | {s['fz_n']['mean']:.1f} | {s['fz_n']['p99']:.1f} | "
            f"{s['fz_n']['max']:.1f} | {s['duty_factor']:.3f} | "
            f"{s['load_bw_scheduled_stance']['mean']:.3f} | "
            f"{s['load_bw_scheduled_stance']['p90']:.3f} | "
            f"{s['load_bw_scheduled_swing']['p95']:.3f} | "
            f"{s['contact_given_scheduled_stance']:.3f} | "
            f"{s['contact_given_scheduled_swing']:.3f} | "
            f"{cs['phase_of_peak_mean_load']:.2f} |"
        )
    sup = summary["support"]
    lines += [
        "",
        "## Total support",
        "",
        f"- mean sum(Fz)/bodyweight: {sup['total_load_bw']['mean']:.3f} "
        f"(sd {sup['total_load_bw']['std']:.3f}, "
        f"p05 {sup['total_load_bw']['p05']:.3f}, "
        f"p95 {sup['total_load_bw']['p95']:.3f})",
        f"- airborne fraction (all feet below threshold): "
        f"{sup['airborne_fraction_all_feet_below_threshold']:.4f}",
        f"- front pair mean: {sup['front_pair_mean_bw']:.3f} bw, rear pair: "
        f"{sup['rear_pair_mean_bw']:.3f} bw",
        f"- diagonal means: FL+RR {sup['diagonal_FL_RR_mean_bw']:.3f} bw, "
        f"FR+RL {sup['diagonal_FR_RL_mean_bw']:.3f} bw",
        f"- instantaneous diagonal difference "
        f"|FL+RR - (FR+RL)|: {sup['diagonal_instantaneous_abs_difference_bw']:.3f} bw",
        f"- corr(sum Fz - 1, base vertical acceleration): "
        f"{sup['corr_total_load_minus_1_with_base_vz_change']:.3f}",
        "",
        "## Constraint rates (all feet, scheduled stance/swing from phase clock)",
        "",
        "| level (bw) | P(load >= level | stance) | P(load >= level | swing) |",
        "|---|---|---|",
    ]
    rates = summary["constraint_rates"]
    for key in rates["stance_load_above"]:
        lines.append(
            f"| {key.replace('_bw', '')} | "
            f"{rates['stance_load_above'][key]:.3f} | "
            f"{rates['swing_load_above'][key]:.3f} |"
        )
    lines += ["", "## Candidate reward grid", ""]
    lines += [
        "| target (bw) | swing tol (bw) | mean r' | p10 | p50 | p90 "
        "| frac<0.5 | frac>0.9 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, val in summary["candidate_reward"]["grid"].items():
        lines.append(
            f"| {key} | {val['reward_mean']:.3f} | {val['reward_p10']:.3f} | "
            f"{val['reward_p50']:.3f} | {val['reward_p90']:.3f} | "
            f"{val['fraction_below_0.5']:.3f} | {val['fraction_above_0.9']:.3f} |"
        )
    variants = summary["candidate_reward"].get("variants")
    if variants:
        lines += [
            "",
            "## Candidate reward variants",
            "",
            f"stance target {variants['params']['stance_target_bw']:.2f} bw, "
            f"swing tolerance {variants['params']['swing_tol_bw']:.2f} bw",
            "",
            "| variant | mean | std | p05 | p50 | p95 |",
            "|---|---|---|---|---|---|",
        ]
        for name, val in variants["stats"].items():
            lines.append(
                f"| {name} | {val['mean']:.3f} | {val['std']:.3f} | "
                f"{val['p05']:.3f} | {val['p50']:.3f} | {val['p95']:.3f} |"
            )
        lines += [
            "",
            "Observability of each variant from actor obs + action "
            "(held-out R² / MAE):",
            "",
        ]
        for name, val in summary["observability"]["candidate_reward_variants"].items():
            lines.append(
                f"- {name}: R² {val['random_feature_r2']:.3f}, "
                f"MAE {val['random_feature_mae']:.3f}"
            )
    lines += [
        "",
        "## Observability (held-out worlds; R² linear / random-feature)",
        "",
        "| feature set | foot load | foot contact | total load | r' |",
        "|---|---|---|---|---|",
    ]
    for fname, res in summary["observability"]["probe"].items():
        lines.append(
            f"| {fname} | "
            f"{res['linear_r2']['foot_load_bw']:.3f} / "
            f"{res['random_feature_r2']['foot_load_bw']:.3f} | "
            f"{res['linear_r2']['foot_contact']:.3f} / "
            f"{res['random_feature_r2']['foot_contact']:.3f} | "
            f"{res['linear_r2']['total_load_bw']:.3f} / "
            f"{res['random_feature_r2']['total_load_bw']:.3f} | "
            f"{res['linear_r2']['candidate_reward']:.3f} / "
            f"{res['random_feature_r2']['candidate_reward']:.3f} |"
        )
    align = summary["phase_alignment"]
    lines += [
        "",
        "## Phase alignment (lag = control steps, 20 ms)",
        "",
        f"- best separation at lag {align['best_separation_lag']}; "
        f"used {align['causal_lag_used']}",
    ]
    for lag, val in align["by_lag"].items():
        lines.append(
            f"- lag {lag}: P(contact|stance) "
            f"{val['contact_given_scheduled_stance']:.3f}, "
            f"P(contact|swing) {val['contact_given_scheduled_swing']:.3f}"
        )
    offset = align["sub_step_offset"]
    lines += [
        "",
        f"- sub-step best offset: {offset['best_phase_shift_rad']:.2f} rad "
        f"({offset['equivalent_time_ms']:.0f} ms of a "
        f"{offset['mean_cycle_s'] * 1e3:.0f} ms cycle); "
        f"separation {offset['best_separation']:.3f} vs "
        f"{offset['separation_at_zero']:.3f} at zero",
        "",
        "FL load R² by leg-phase bin (obs+action):",
        "",
    ]
    for key, val in summary["observability"]["fl_load_r2_by_phase_bin"].items():
        lines.append(f"- {key} rad: {val:.3f}")
    if summary.get("backend_agreement"):
        b = summary["backend_agreement"]
        lines += [
            "",
            "## CPU/Warp agreement (FL foot)",
            "",
            f"- FL duty: cpu {b['cpu']['fl_duty']:.3f} "
            f"vs gpu {b['gpu']['fl_duty']:.3f}",
            f"- FL stance load: cpu {b['cpu']['fl_stance_load_bw']:.3f} bw vs "
            f"gpu {b['gpu']['fl_stance_load_bw']:.3f} bw",
            f"- total load: cpu {b['cpu']['total_load_bw_mean']:.3f} vs "
            f"gpu {b['gpu']['total_load_bw_mean']:.3f} bw",
            f"- airborne: cpu {b['cpu']['airborne_fraction']:.3f} vs "
            f"gpu {b['gpu']['airborne_fraction']:.3f}",
            f"- FL phase-curve max |diff|: "
            f"{b['fl_phase_curve_max_abs_diff_bw']:.3f} bw "
            f"(mean {b['fl_phase_curve_mean_abs_diff_bw']:.3f} bw)",
        ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
