"""Threshold-based contact label: r' = -1 above a per-leg force threshold.

Thresholds come from each foot's own force distribution, so the label can flag
either a relative fraction of bad touchdowns or an absolute overload. This
script measures what each threshold family produces:

* the thresholds in bodyweight and Newtons;
* how often the label fires per control step, per leg, and per touchdown;
* the discounted return of policy `r` under the label (gamma 0.99, 250 steps);
* where in a stance the violations occur (lead time for an intervention);
* whether the actor observation predicts an imminent violation (held-out AUC).

Run from the product repo root:

    uv run --frozen python \
      docs/plans/supplements/foot_contact_study/analyze_threshold_label.py
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
GAMMA = 0.99
HORIZON = 250  # one 5 s episode at 50 Hz


def run_runs(mask_1d):
    padded = np.concatenate([[False], mask_1d, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[0::2], edges[1::2]))


def ridge_fit_predict(x_train, y_train, x_test, lam):
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
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 1.0 / np.sqrt(x.shape[1]), size=(x.shape[1], units))
    b = rng.normal(0.0, 1.0, size=units)
    return np.tanh(x @ w + b)


def auc(score, label):
    """Rank-based AUC (Mann-Whitney)."""
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    positives = label.astype(bool)
    n_pos = positives.sum()
    n_neg = len(label) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def discounted_return(reward, gamma=GAMMA, horizon=HORIZON):
    """G_t = sum_{k<horizon} gamma^k r_{t+k}, over the leading time axis."""
    n_steps = reward.shape[0]
    last = n_steps - horizon + 1
    weights = gamma ** np.arange(horizon)
    out = np.empty((last,) + reward.shape[1:])
    for start in range(last):
        out[start] = (reward[start : start + horizon] * weights[:, None]).sum(axis=0)
    return out


def main():
    data = np.load(DATA / "gpu.npz")
    meta = json.loads((DATA / "gpu.json").read_text())
    feet = meta["feet_names"]
    dt = float(data["dt"])
    bodyweight = data["total_mass"] * G
    load = data["foot_forces"][..., 2] / bodyweight[None, :, None]
    phase = data["phase"].astype(np.float64)
    clean = (data["nonfoot_contact_norm"] < 1.0) & (data["terminated"] < 0.5)
    n_steps, n_envs = load.shape[:2]
    contact = load > CONTACT_FRACTION_BW

    # ── per-touchdown peak distribution (same definition as §2.1) ──────────
    peaks = {name: [] for name in feet}
    peak_records = []
    for env in range(n_envs):
        for k, name in enumerate(feet):
            for start, end in run_runs(contact[:, env, k]):
                if end - start < 5:
                    continue
                peak = float(load[start:end, env, k].max())
                if peak < 0.30:
                    continue
                peaks[name].append(peak)
                peak_records.append(
                    {
                        "env": env,
                        "foot": name,
                        "start": start,
                        "end": end,
                        "peak": peak,
                        "peak_step": start + int(np.argmax(load[start:end, env, k])),
                    }
                )
    peaks = {name: np.asarray(values) for name, values in peaks.items()}

    # per-cycle maximum distribution, for the "cycle max" threshold family
    wraps = np.zeros_like(phase, dtype=np.int64)
    wraps[1:] = np.diff(phase, axis=0) < -np.pi
    cycle_id = np.cumsum(wraps, axis=0)
    cycle_max = {name: [] for name in feet}
    for env in range(n_envs):
        ids = np.unique(cycle_id[:, env])
        for cycle in ids[1:-1]:
            in_cycle = cycle_id[:, env] == cycle
            for k, name in enumerate(feet):
                cycle_max[name].append(float(load[in_cycle, env, k].max()))
    cycle_max = {name: np.asarray(v) for name, v in cycle_max.items()}

    def limits(percentile=90, scale=1.0, source="peak"):
        table = peaks if source == "peak" else cycle_max
        return np.array(
            [scale * np.percentile(table[name], percentile) for name in feet]
        )

    families = {
        "peak_p90": limits(90),
        "peak_p95": limits(95),
        "peak_p99": limits(99),
        "peak_0.9xp50": limits(50, 0.9),
        "peak_0.9xp99": limits(99, 0.9),
        "peak_0.9xmax": limits(100, 0.9),
        "cycle_max_p90": limits(90, source="cycle"),
    }

    def evaluate(limit):
        exceed = load > limit[None, None, :]
        any_exceed = exceed.any(axis=-1)
        flagged = sum(
            1
            for record in peak_records
            if record["peak"] > limit[feet.index(record["foot"])]
        )
        touch_fire = np.zeros_like(any_exceed)
        for record in peak_records:
            if record["peak"] > limit[feet.index(record["foot"])]:
                touch_fire[record["peak_step"], record["env"]] = True
        returns = discounted_return(-any_exceed.astype(np.float64))
        # offset of every violating step inside its own contact run
        offsets = []
        for env in range(n_envs):
            for k, name in enumerate(feet):
                for start, end in run_runs(contact[:, env, k]):
                    for step in range(start, end):
                        if exceed[step, env, k]:
                            offsets.append(
                                (
                                    (step - start) * dt,
                                    (end - step) * dt,
                                    (end - start) * dt,
                                )
                            )
        offsets = np.asarray(offsets) if offsets else np.zeros((0, 3))
        return {
            "limits_bw": {name: float(limit[k]) for k, name in enumerate(feet)},
            "limits_n_at_mean_bodyweight": {
                name: float(limit[k] * bodyweight.mean()) for k, name in enumerate(feet)
            },
            "steps_above_any_percent": 100.0 * float(any_exceed[clean].mean()),
            "violations_per_250_steps": 250.0 * float(any_exceed[clean].mean()),
            "steps_above_per_leg_percent": {
                name: 100.0 * float(exceed[:, :, k][clean].mean())
                for k, name in enumerate(feet)
            },
            "touchdowns_flagged_percent": 100.0 * flagged / max(len(peak_records), 1),
            "peak_step_label_percent": 100.0 * float(touch_fire[clean].mean()),
            "windows_with_any_violation_percent": 100.0
            * float((returns < 0.0).any(axis=1).mean()),
            "return_mean": float(returns.mean()),
            "return_p05": float(np.percentile(returns, 5)),
            "return_p50": float(np.percentile(returns, 50)),
            "return_min": float(returns.min()),
            "timing": {
                "violating_steps": int(len(offsets)),
                "seconds_since_run_start_p50": float(np.percentile(offsets[:, 0], 50)),
                "seconds_to_run_end_p50": float(np.percentile(offsets[:, 1], 50)),
                "run_duration_p50": float(np.percentile(offsets[:, 2], 50)),
                "fraction_in_first_quarter": float(
                    (offsets[:, 0] < 0.25 * offsets[:, 2]).mean()
                ),
                "fraction_in_last_quarter": float(
                    (offsets[:, 1] < 0.25 * offsets[:, 2]).mean()
                ),
            },
        }

    result = {
        "threshold_source": (
            "per-foot distribution under policy r; 'peak_pXX' uses the "
            "per-touchdown peak load (loaded stances), 'cycle_max_p90' the "
            "per-cycle maximum, 'peak_0.9x...' scales a tail value by 0.9"
        ),
        "touchdown_counts": {name: int(v.size) for name, v in peaks.items()},
        "families": {name: evaluate(limit) for name, limit in families.items()},
        "observability": {},
        "note": "labels are indicator rewards in {-1, 0}, zero after terminal",
    }

    # ── observability of the next-step label, for the p90 family ───────────
    limit = families["peak_p90"]
    any_exceed = (load > limit[None, None, :]).any(axis=-1)
    obs = data["actor_obs"].astype(np.float64)
    action = data["dof_pos_target"].astype(np.float64)
    features = np.concatenate([obs, action], axis=-1).reshape(-1, obs.shape[-1] + 12)
    shift = lambda a: np.concatenate([a[1:], a[-1:]], axis=0)  # noqa: E731
    target = shift(any_exceed).astype(np.float64).reshape(-1, 1)
    mask = (clean & shift(clean)).reshape(-1)
    env_ids = np.repeat(np.arange(n_envs), n_steps)
    train_env = np.arange(n_envs) < int(round(0.8 * n_envs))
    train = mask & np.isin(env_ids, np.flatnonzero(train_env))
    test = mask & ~np.isin(env_ids, np.flatnonzero(train_env))
    scores = {}
    for name, feats in (
        ("linear", features),
        ("random_feature", random_features(features, 256, seed=0)),
    ):
        pred = ridge_fit_predict(feats[train], target[train], feats[test], 1e-2)
        scores[name] = {
            "auc": auc(pred[:, 0], target[test, 0]),
            "pred_mean": float(pred.mean()),
        }
    result["observability"] = {
        "target": "any foot above its peak_p90 threshold at t+1",
        "base_rate_percent": 100.0 * float(target[test].mean()),
        "held_out_rows": int(test.sum()),
        "scores": scores,
    }

    # ── figure ─────────────────────────────────────────────────────────────
    plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.25})
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(11, 4))
    percentiles = [80, 85, 90, 95, 97, 99]
    step_rates = []
    touch_rates = []
    for percentile in percentiles:
        current = limits(percentile)
        exceed = (load > current[None, None, :]).any(axis=-1)
        step_rates.append(100.0 * float(exceed[clean].mean()))
        flagged = sum(
            1
            for record in peak_records
            if record["peak"] > current[feet.index(record["foot"])]
        )
        touch_rates.append(100.0 * flagged / max(len(peak_records), 1))
    ax.plot(percentiles, touch_rates, "o-", label="touchdowns flagged")
    ax.plot(percentiles, step_rates, "s--", label="control steps flagged")
    ax.axvline(90, color="k", ls=":", lw=1)
    ax.set_xlabel("per-leg peak percentile used as threshold")
    ax.set_ylabel("percent flagged")
    ax.set_title("Threshold choice: touchdown vs step rate")
    ax.legend(fontsize=7)
    labels = list(feet)
    low = [result["families"]["peak_p90"]["limits_bw"][n] for n in labels]
    high = [result["families"]["peak_p99"]["limits_bw"][n] for n in labels]
    xpos = np.arange(len(labels))
    axb.bar(xpos - 0.18, low, 0.36, label="p90 of peaks")
    axb.bar(xpos + 0.18, high, 0.36, label="p99 of peaks")
    axb.set_xticks(xpos)
    axb.set_xticklabels(labels)
    axb.set_ylabel("threshold Fz / bodyweight")
    axb.set_title("Per-leg thresholds")
    axb.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(
        FIG / "fig11_threshold_label.png",
        dpi=100,
        pil_kwargs={"optimize": True},
    )
    plt.close(fig)

    (ART / "threshold_label.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, entry in result["families"].items():
        print(
            f"{name:14s} limits "
            + " ".join(f"{entry['limits_bw'][n]:.2f}" for n in feet)
            + f" | steps {entry['steps_above_any_percent']:.2f}%"
            + f" | touchdowns {entry['touchdowns_flagged_percent']:.1f}%"
            + f" | return {entry['return_mean']:.2f}"
            + f" | first-quarter {entry['timing']['fraction_in_first_quarter']:.2f}"
        )
    print(json.dumps(result["observability"], indent=2))


if __name__ == "__main__":
    main()
