"""Summarize the rejection/distance threshold sweep without simulation.

The plain and historical best-eight references are read from the earlier
persistent-intervention run. All controllers share the same [state, execution]
layout and paired seed list, so pair gains are exact within-state contrasts.
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch

from learning.offline.contracts import sha256_file
from learning.offline.persistent_intervention import paired_survivor_tail_interval

TAIL = 0.99**500 / 0.01
EXECUTIONS = 1024
STATES = 128


def student_t_975(df):
    """Cornish-Fisher expansion for the 0.975 Student-t quantile.

    Torch in this environment does not implement ``StudentT.icdf`` and SciPy is
    not a project dependency. The expansion is accurate well below the reported
    precision for the large (1023) execution degrees of freedom used here.
    """
    z = 1.959963984540054
    z2 = z * z
    z3 = z * z2
    z5 = z3 * z2
    z7 = z5 * z2
    z9 = z7 * z2
    return (
        z
        + (z3 + z) / (4 * df)
        + (5 * z5 + 16 * z3 + 3 * z) / (96 * df**2)
        + (3 * z7 + 19 * z5 + 17 * z3 - 15 * z) / (384 * df**3)
        + (79 * z9 + 776 * z7 + 1482 * z5 - 1920 * z3 - 945 * z) / (92160 * df**4)
    )


T_CRIT = student_t_975(EXECUTIONS - 1)
THRESHOLDS = [-16.0, -4.866, -1.776, -0.956, -0.626]
CONTROLLERS = (
    ("reject_16", "rejection", -16.0),
    ("reject_4p866", "rejection", -4.866),
    ("reject_1p776", "rejection", -1.776),
    ("reject_0p956", "rejection", -0.956),
    ("reject_0p626", "rejection", -0.626),
    ("distance_4p866", "distance", -4.866),
    ("distance_1p776", "distance", -1.776),
)
REFERENCES = (("plain", "baseline"), ("best8", "original"))


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def load_blocks(directory, name):
    blocks = [
        torch.load(directory / f"{name}_{block:02d}.pt", weights_only=True)
        for block in range(16)
    ]
    first = blocks[0]
    keys = [key for key, value in first.items() if torch.is_tensor(value)]
    out = {}
    for key in keys:
        stacked = torch.stack([block[key] for block in blocks])
        tail = stacked.shape[3:]
        out[key] = (
            stacked.permute(1, 0, 2, *range(3, stacked.ndim))
            .reshape(STATES, -1, *tail)
            .double()
        )
    out["seeds"] = [list(block["seeds"]) for block in blocks]
    out["seconds"] = [float(block["seconds"]) for block in blocks]
    return out


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        raise ValueError("empty denominator")
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [centre - half, centre + half]


def interval(values, seed, draws=10_000):
    generator = torch.Generator().manual_seed(seed)
    index = torch.randint(len(values), (draws, len(values)), generator=generator)
    means = values[index].mean(-1)
    return [float(means.quantile(0.025)), float(means.quantile(0.975))]


def general_stats(returns, survived, steps, oob):
    mean = returns.mean(1)
    se = returns.std(1) / EXECUTIONS**0.5
    surv = survived.double().mean(1)
    pooled = returns.flatten()
    return {
        "mean_return": float(mean.mean()),
        "pooled_individual_sd": float(pooled.std()),
        "pooled_individual_min": float(pooled.min()),
        "pooled_individual_max": float(pooled.max()),
        "state_mean_sd": float(mean.std()),
        "state_mean_min": float(mean.min()),
        "state_mean_max": float(mean.max()),
        "fixed_state_execution_ci": [
            float(mean.mean() - 1.96 * (se**2).mean().sqrt() / STATES**0.5),
            float(mean.mean() + 1.96 * (se**2).mean().sqrt() / STATES**0.5),
        ],
        "state_cluster_ci": interval(mean, 202611111),
        "survivor_fraction": float(surv.mean()),
        "tail_interval": [
            float((mean - TAIL * surv).mean()),
            float(mean.mean()),
        ],
        "inside_band_step_fraction": float(1 - oob.sum() / steps.sum()),
        "fall_fraction": float(1 - survived.double().mean()),
        "mean_steps": float(steps.mean()),
        "_state_mean": mean,
        "_state_se": se,
        "_state_surv": surv,
    }


def threshold_stats(mean, se, surv, returns, c):
    low = mean - T_CRIT * se
    high = mean + T_CRIT * se
    tail_low = mean - TAIL * surv
    tail_high = mean
    above = low > c
    below = high < c
    tail_above = tail_low >= c
    tail_below = tail_high < c
    return {
        "c": c,
        "configs": STATES,
        "configs_mean_ge_c": int((mean >= c).sum()),
        "fraction_configs_mean_ge_c": float((mean >= c).double().mean()),
        "fraction_configs_mean_ge_c_wilson95": wilson(int((mean >= c).sum()), STATES),
        "configs_ci_confidently_above": int(above.sum()),
        "configs_ci_confidently_below": int(below.sum()),
        "configs_ci_uncertain": int((~(above | below)).sum()),
        "configs_tail_confidently_above": int(tail_above.sum()),
        "configs_tail_confidently_below": int(tail_below.sum()),
        "configs_tail_uncertain": int((~(tail_above | tail_below)).sum()),
        "fraction_individual_returns_ge_c": float((returns >= c).double().mean()),
        "individual_returns_ge_c_is_not_calibration": True,
        "ci_low_state_min": float(low.min()),
        "ci_high_state_max": float(high.max()),
        "tail_low_state_min": float(tail_low.min()),
        "tail_high_state_max": float(tail_high.max()),
    }


def pair_stats(a, b):
    """Paired gain a - b on the shared [state, execution] layout."""
    paired = a["returns"] - b["returns"]
    gain = paired.mean(1)
    mean = float(gain.mean())
    se_d = paired.std(1) / EXECUTIONS**0.5
    gain_sd = float(gain.std())
    tol = 1e-9
    win = int((gain > tol).sum())
    lose = int((gain < -tol).sum())
    tie = STATES - win - lose
    group_means = paired.reshape(STATES, 64, 16).mean(-1)
    positive = int((group_means > tol).sum())
    groups = int(group_means.numel())
    per_state_positive = (group_means > tol).sum(-1)
    return {
        "mean_gain": mean,
        "per_state_gain_sd": gain_sd,
        "per_state_gain_median": float(gain.median()),
        "per_state_gain_q05": float(gain.quantile(0.05)),
        "per_state_gain_min": float(gain.min()),
        "per_state_gain_max": float(gain.max()),
        "execution_diff_sd": float(paired.std()),
        "paired_execution_ci": [
            float(mean - 1.96 * (se_d**2).mean().sqrt() / STATES**0.5),
            float(mean + 1.96 * (se_d**2).mean().sqrt() / STATES**0.5),
        ],
        "paired_state_cluster_ci": interval(gain, 202611112),
        "wins": win,
        "ties": tie,
        "losses": lose,
        "prob_state_win": win / STATES,
        "prob_execution_win": float((paired > tol).double().mean()),
        "paired_tail_interval": list(
            paired_survivor_tail_interval(
                mean,
                float(b["survived"].double().mean()),
                float(a["survived"].double().mean()),
                TAIL,
            )
        ),
        "batch16_groups_per_state": 64,
        "batch16_groups_total": groups,
        "batch16_positive": positive,
        "batch16_positive_fraction": positive / groups,
        "batch16_tie_fraction": float((group_means.abs() <= tol).double().mean()),
        "batch16_negative_fraction": float((group_means < -tol).double().mean()),
        "states_with_positive_majority": int((per_state_positive > 32).sum()),
        "per_state_positive_groups_min": int(per_state_positive.min()),
        "per_state_positive_groups_median": float(per_state_positive.median()),
        "per_state_positive_groups_max": int(per_state_positive.max()),
    }


def controller_stats(data):
    returns = data["returns"]
    steps = data["steps"]
    survived = data["survived"].bool()
    base = general_stats(returns, survived, steps, data["oob"])
    out = {key: value for key, value in base.items() if not key.startswith("_")}
    if "selection_counts" in data:
        selection = data["selection_counts"]
        total = selection.sum()
        out["selection_fraction"] = (selection.sum((0, 1)) / total).tolist()
        out["first_candidate_fraction"] = float(selection[..., 0].sum() / total)
    if "q_gain_sum" in data:
        out["mean_q_gain_per_decision"] = float(data["q_gain_sum"].sum() / steps.sum())
    out["first_reward_negative_rate"] = float(
        (data["first_rewards"] == -1).double().mean()
    )
    out["first_step_termination_rate"] = float(data["first_terminated"].double().mean())
    out["xy_tracking_rmse"] = float(
        (data["xy_error_sq"].sum() / data["tracking_steps"].sum()).sqrt()
    )
    out["yaw_tracking_rmse"] = float(
        (data["yaw_error_sq"].sum() / data["tracking_steps"].sum()).sqrt()
    )
    out["speed_over_command"] = float(
        data["speed_sum"].sum() / data["command_speed_sum"].sum()
    )
    if "fallback_sum" in data:
        out["fallback_rate"] = float(data["fallback_sum"].sum() / steps.sum())
    if "selected_q_sum" in data:
        out["mean_selected_q"] = float(data["selected_q_sum"].sum() / steps.sum())
    if "selected_distance_sum" in data:
        out["mean_selected_distance"] = float(
            data["selected_distance_sum"].sum() / steps.sum()
        )
    if "mean_feasible_sum" in data:
        out["mean_feasible_rate"] = float(data["mean_feasible_sum"].sum() / steps.sum())
        out["mean_selected_fraction"] = float(
            data["mean_selected_sum"].sum() / steps.sum()
        )
        out["stochastic_feasible_rate"] = float(
            data["stochastic_feasible_sum"].sum() / steps.sum()
        )
    if "attempt_counts" in data:
        attempt = data["attempt_counts"]
        total = attempt.sum()
        out["effective_attempt_fraction"] = (attempt.sum((0, 1)) / total).tolist()
        out["effective_attempt_mean"] = float(
            (attempt.sum((0, 1)) * torch.arange(1, 9, dtype=torch.float64)).sum()
            / total
        )
    if "termination_step" in data:
        term = data["termination_step"]
        out["mean_termination_step"] = float(term.mean())
        bins = term.new_zeros(11)
        for index in range(10):
            low = index * 50
            high = (index + 1) * 50
            bins[index] = ((term >= low) & (term < high)).sum()
        bins[10] = (term >= 500).sum()
        out["termination_step_histogram_50"] = (bins / bins.sum()).tolist()
        terminated = ~survived
        out["terminated_oob_rate"] = (
            float((data["oob"][terminated] / steps[terminated]).mean())
            if terminated.any()
            else None
        )
    out["block_seconds"] = data["seconds"]
    out["collection_seconds"] = float(sum(data["seconds"]))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument(
        "--reuse",
        default="/tmp/opencode/height-q-persistent-intervention-20261110-v1",
    )
    args = parser.parse_args()
    directory = Path(args.directory)
    reuse = Path(args.reuse)

    new_provenance = json.loads((directory / "provenance.json").read_text())
    old_provenance = json.loads((reuse / "provenance.json").read_text())
    for key in ("source_sha256", "checkpoint_sha256"):
        if new_provenance[key] != old_provenance[key]:
            raise ValueError(f"reuse provenance mismatch: {key}")
    if (
        new_provenance["artifact_sha256"]
        != old_provenance["artifacts"][new_provenance["artifact"]]
    ):
        raise ValueError("reuse artifact hash mismatch")

    data = {name: load_blocks(directory, name) for name, _, _ in CONTROLLERS}
    for name, filename in REFERENCES:
        data[name] = load_blocks(reuse, filename)
    expected_seeds = [
        7100000 + 10 * (block * 8 + call) for block in range(16) for call in range(8)
    ]
    for name in data:
        flat = [seed for block in data[name]["seeds"] for seed in block]
        if flat != expected_seeds:
            raise ValueError(f"{name} seed identity mismatch")
        if data[name]["returns"].shape != (STATES, EXECUTIONS):
            raise ValueError(f"{name} return shape mismatch")
        if not torch.isfinite(data[name]["returns"]).all():
            raise ValueError(f"{name} nonfinite return")
        if data[name]["timeout_seen"].any():
            raise ValueError(f"{name} timeout observed")
        if not torch.isin(data[name]["first_rewards"], torch.tensor([-1.0, 0.0])).all():
            raise ValueError(f"{name} nonbinary first reward")
        decisions = data[name]["selection_counts"].sum(-1)
        if not torch.equal(decisions, data[name]["steps"]):
            raise ValueError(f"{name} selection accounting mismatch")

    report = {
        "generated_at": utc_now(),
        "tail_width": TAIL,
        "t_critical_1023": T_CRIT,
        "seed_identity": "(7100000+10*call,lane,state) for every controller",
        "seed_list_verified": True,
        "source_checkpoint_artifact_hashes_verified": True,
        "std_definitions": {
            "pooled_individual_sd": (
                "ddof=1 over 128*1024 individual H500 returns; outcome spread, "
                "not the standard error of the mean"
            ),
            "state_mean_sd": "ddof=1 over the 128 state-mean returns",
            "per_state_gain_sd": "ddof=1 over the 128 paired state-mean gains",
        },
        "controllers": {},
        "references": {},
    }
    for name, kind, c in CONTROLLERS:
        entry = controller_stats(data[name])
        entry.update(kind=kind, threshold=c)
        mean = data[name]["returns"].mean(1)
        se = data[name]["returns"].std(1) / EXECUTIONS**0.5
        surv = data[name]["survived"].double().mean(1)
        entry["threshold_attainment"] = threshold_stats(
            mean, se, surv, data[name]["returns"], c
        )
        report["controllers"][name] = entry

    for name, _ in REFERENCES:
        entry = controller_stats(data[name])
        entry.update(kind="reference", threshold=None)
        # Reference rows have no assigned threshold; report how often their
        # state means clear each candidate threshold for context only.
        mean = data[name]["returns"].mean(1)
        entry["state_means_ge_threshold"] = {
            str(c): int((mean >= c).sum()) for c in THRESHOLDS
        }
        report["references"][name] = entry

    for name, _, _ in CONTROLLERS:
        report["controllers"][name]["gain_vs_plain"] = pair_stats(
            data[name], data["plain"]
        )
        report["controllers"][name]["gain_vs_best8"] = pair_stats(
            data[name], data["best8"]
        )
    report["references"]["best8"]["gain_vs_plain"] = pair_stats(
        data["best8"], data["plain"]
    )

    # Per-configuration return spread.
    with (directory / "per_config_returns.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "state",
                "controller",
                "kind",
                "threshold",
                "mean_return",
                "pooled_sd",
                "min_return",
                "max_return",
                "survivor_fraction",
            ]
        )
        for name in [*data]:
            kind = "reference"
            c = ""
            for candidate, candidate_kind, threshold in CONTROLLERS:
                if candidate == name:
                    kind, c = candidate_kind, threshold
            for state in range(STATES):
                row = data[name]["returns"][state]
                writer.writerow(
                    [
                        state,
                        name,
                        kind,
                        c,
                        f"{row.mean():.6f}",
                        f"{row.std():.6f}",
                        f"{row.min():.6f}",
                        f"{row.max():.6f}",
                        f"{data[name]['survived'][state].double().mean():.6f}",
                    ]
                )

    # Per-configuration threshold attainment for the seven sweep controllers.
    with (directory / "per_config_threshold.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "state",
                "controller",
                "kind",
                "threshold",
                "mean_return",
                "ci_low",
                "ci_high",
                "ci_class",
                "tail_low",
                "tail_high",
                "tail_class",
                "mean_ge_c",
                "individual_ge_c",
            ]
        )
        for name, kind, c in CONTROLLERS:
            returns = data[name]["returns"]
            mean = returns.mean(1)
            se = returns.std(1) / EXECUTIONS**0.5
            surv = data[name]["survived"].double().mean(1)
            low = mean - T_CRIT * se
            high = mean + T_CRIT * se
            tail_low = mean - TAIL * surv
            tail_high = mean
            for state in range(STATES):
                ci_class = (
                    "above"
                    if low[state] > c
                    else "below"
                    if high[state] < c
                    else "uncertain"
                )
                tail_class = (
                    "above"
                    if tail_low[state] >= c
                    else "below"
                    if tail_high[state] < c
                    else "uncertain"
                )
                writer.writerow(
                    [
                        state,
                        name,
                        kind,
                        c,
                        f"{mean[state]:.6f}",
                        f"{low[state]:.6f}",
                        f"{high[state]:.6f}",
                        ci_class,
                        f"{tail_low[state]:.6f}",
                        f"{tail_high[state]:.6f}",
                        tail_class,
                        int(mean[state] >= c),
                        f"{(returns[state] >= c).double().mean():.6f}",
                    ]
                )

    # Per-configuration pair gains against plain and best8.
    with (directory / "pair_gains.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "state",
                "controller",
                "reference",
                "mean_gain",
                "ci_low",
                "ci_high",
                "tail_low",
                "tail_high",
                "batch16_positive_groups",
            ]
        )
        for name, _, _ in CONTROLLERS:
            for reference in ("plain", "best8"):
                paired = data[name]["returns"] - data[reference]["returns"]
                gain = paired.mean(1)
                se = paired.std(1) / EXECUTIONS**0.5
                low = gain - T_CRIT * se
                high = gain + T_CRIT * se
                base_surv = data[reference]["survived"].double().mean(1)
                ctrl_surv = data[name]["survived"].double().mean(1)
                tail_low = gain - TAIL * base_surv
                tail_high = gain + TAIL * ctrl_surv
                groups = (paired.reshape(STATES, 64, 16).mean(-1) > 1e-9).sum(-1)
                for state in range(STATES):
                    writer.writerow(
                        [
                            state,
                            name,
                            reference,
                            f"{gain[state]:.6f}",
                            f"{low[state]:.6f}",
                            f"{high[state]:.6f}",
                            f"{tail_low[state]:.6f}",
                            f"{tail_high[state]:.6f}",
                            int(groups[state]),
                        ]
                    )

    # Raw [state, execution] returns for every controller.
    torch.save(
        {
            "returns": {name: data[name]["returns"] for name in data},
            "survived": {name: data[name]["survived"] for name in data},
            "steps": {name: data[name]["steps"] for name in data},
            "oob": {name: data[name]["oob"] for name in data},
            "first_rewards": {name: data[name]["first_rewards"] for name in data},
            "seeds": expected_seeds,
            "tail_width": TAIL,
            "controllers": [
                {"name": name, "kind": kind, "threshold": c}
                for name, kind, c in CONTROLLERS
            ],
        },
        directory / "raw_returns.pt",
    )
    (directory / "results.json").write_text(json.dumps(report, indent=2))
    write_report(directory, report)
    with (directory / "commands.txt").open("a") as f:
        f.write(
            f"{utc_now()} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen "
            + "python "
            + " ".join(sys.argv)
            + "\n"
        )
    (directory / "analysis_code.json").write_text(
        json.dumps(
            {
                str(Path(__file__)): sha256_file(Path(__file__)),
                "learning/offline/persistent_intervention.py": sha256_file(
                    "learning/offline/persistent_intervention.py"
                ),
            },
            indent=2,
        )
    )
    print(json.dumps(report, indent=2))


def write_report(directory, report):
    lines = [
        "# Rejection and closest-mean threshold sweep — 2026-11-11",
        "",
        "Frozen `model_600`, original fair-100 PPO-data height Q, 128 fixed "
        "`k=100` native snapshots, H500, gamma 0.99, 1024 paired executions per "
        "state, reward 0 inside `0.32 ± 0.05 m` and -1 outside, absorbing true "
        "failure.",
        "",
        "## Aggregate returns",
        "",
        "| Controller | Kind | c | Mean H500 | Pooled SD | Min | Max | "
        "Inside-band step | Fall |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, kind, c in CONTROLLERS:
        e = report["controllers"][name]
        lines.append(
            f"| {name} | {kind} | {c:g} | {e['mean_return']:.4f} | "
            f"{e['pooled_individual_sd']:.4f} | {e['pooled_individual_min']:.4f} | "
            f"{e['pooled_individual_max']:.4f} | "
            f"{100 * e['inside_band_step_fraction']:.3f}% | "
            f"{100 * e['fall_fraction']:.3f}% |"
        )
    for name in ("plain", "best8"):
        e = report["references"][name]
        lines.append(
            f"| {name} | reference | — | {e['mean_return']:.4f} | "
            f"{e['pooled_individual_sd']:.4f} | {e['pooled_individual_min']:.4f} | "
            f"{e['pooled_individual_max']:.4f} | "
            f"{100 * e['inside_band_step_fraction']:.3f}% | "
            f"{100 * e['fall_fraction']:.3f}% |"
        )
    lines += [
        "",
        "`Pooled SD` is ddof=1 over all 131,072 individual returns; it is outcome "
        "spread, not the standard error of the mean. `Min`/`Max` are observed "
        "individual returns, not distribution bounds.",
        "",
        "## Threshold attainment by configuration",
        "",
        "Each configuration is one fixed state. Its 1024 executions give a mean "
        "and a Student-t interval (df=1023). `CI above/below` means the whole "
        "interval is on one side of the controller's fixed `c`; `tail "
        "above/below` uses the omitted survivor interval "
        f"`[-{TAIL:.6f}, 0]` instead. `Individual >= c` is not expectation "
        "calibration.",
        "",
        "| Controller | c | Mean >= c | 95% Wilson | CI above | CI below | "
        "CI uncertain | Tail above | Tail below | Tail uncertain | "
        "Individual >= c |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, kind, c in CONTROLLERS:
        t = report["controllers"][name]["threshold_attainment"]
        lines.append(
            f"| {name} | {c:g} | {t['configs_mean_ge_c']}/128 | "
            f"[{t['fraction_configs_mean_ge_c_wilson95'][0]:.3f}, "
            f"{t['fraction_configs_mean_ge_c_wilson95'][1]:.3f}] | "
            f"{t['configs_ci_confidently_above']} | "
            f"{t['configs_ci_confidently_below']} | "
            f"{t['configs_ci_uncertain']} | "
            f"{t['configs_tail_confidently_above']} | "
            f"{t['configs_tail_confidently_below']} | "
            f"{t['configs_tail_uncertain']} | "
            f"{100 * t['fraction_individual_returns_ge_c']:.2f}% |"
        )
    lines += [
        "",
        "Reference state means against each candidate threshold (no controller "
        "was assigned these):",
        "",
        "| Reference | " + " | ".join(f"c={c:g}" for c in THRESHOLDS) + " |",
        "|---|" + "---:|" * len(THRESHOLDS),
    ]
    for name in ("plain", "best8"):
        counts = report["references"][name]["state_means_ge_threshold"]
        lines.append(
            f"| {name} | "
            + " | ".join(f"{counts[str(c)]}/128" for c in THRESHOLDS)
            + " |"
        )
    lines += [
        "",
        "## Paired gains",
        "",
        "| Controller | vs | Mean gain | State CI | Paired tail | Exec win | "
        "Batch16 positive |",
        "|---|---|---:|---|---|---:|---:|",
    ]
    for name, _, _ in CONTROLLERS:
        for reference in ("plain", "best8"):
            g = report["controllers"][name][f"gain_vs_{reference}"]
            lines.append(
                f"| {name} | {reference} | {g['mean_gain']:.4f} | "
                f"[{g['paired_state_cluster_ci'][0]:.4f}, "
                f"{g['paired_state_cluster_ci'][1]:.4f}] | "
                f"[{g['paired_tail_interval'][0]:.4f}, "
                f"{g['paired_tail_interval'][1]:.4f}] | "
                f"{100 * g['prob_execution_win']:.2f}% | "
                f"{100 * g['batch16_positive_fraction']:.2f}% |"
            )
    g = report["references"]["best8"]["gain_vs_plain"]
    lines.append(
        f"| best8 | plain | {g['mean_gain']:.4f} | "
        f"[{g['paired_state_cluster_ci'][0]:.4f}, "
        f"{g['paired_state_cluster_ci'][1]:.4f}] | "
        f"[{g['paired_tail_interval'][0]:.4f}, "
        f"{g['paired_tail_interval'][1]:.4f}] | "
        f"{100 * g['prob_execution_win']:.2f}% | "
        f"{100 * g['batch16_positive_fraction']:.2f}% |"
    )
    lines += [
        "",
        "Paired tail intervals use the tested asymmetric survivor helper: the "
        "lower side subtracts the reference survivor weight and the upper side "
        "adds the controller survivor weight. Batch16 group fractions are over "
        "128 states × 64 contiguous groups of 16 executions; they are not "
        "`E[gain]`.",
        "",
        "## Selection diagnostics",
        "",
        "| Controller | Fallback | Mean feasible | Mean selected | "
        "First draw selected | Attempt mean | Mean selected Q |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, kind, c in CONTROLLERS:
        e = report["controllers"][name]
        if kind == "rejection":
            mean_feasible = "—"
            selected = "—"
            first = f"{100 * e['first_candidate_fraction']:.3f}%"
            attempt = f"{e['effective_attempt_mean']:.3f}"
        else:
            mean_feasible = f"{100 * e['mean_feasible_rate']:.3f}%"
            selected = f"{100 * e['mean_selected_fraction']:.3f}%"
            first = "—"
            attempt = "—"
        lines.append(
            f"| {name} | {100 * e['fallback_rate']:.3f}% | {mean_feasible} | "
            f"{selected} | {first} | {attempt} | {e['mean_selected_q']:.4f} |"
        )
    lines += [
        "",
        "Rejection selection uses eight stochastic candidates only; the mean is "
        "not included. `Effective attempt` is the first feasible draw index "
        "(1–8), with an all-infeasible fallback counted as eight. The "
        "deterministic controller uses the exact mean plus the same eight draws "
        "and minimizes the policy-standardized squared distance among feasible "
        "candidates, so its mean-feasible rate equals its mean-selected rate.",
        "",
        "## Caveats",
        "",
        "- These 128 `k=100` states are fixed snapshots from one frozen policy "
        "and one source lane. They are not a state population, hardware trial, "
        "or safety evidence.",
        "- A fixed `c` is compared with the expected H500 (plus absorbing) "
        "return from insertion. `Q(s_t, a_t) >= c` at each decision does not "
        "maintain an original budget, and a selected prediction is conditional "
        "on the actually executed policy, not `Q^pi` truth.",
        "- The critic was trained on nominal-policy data. Persistent deviations "
        "move the branch off its target distribution; fallback violations and "
        "hidden physical state can bias selection.",
        "- Threshold attainment need not be monotone in `c` across "
        "configurations. Multiple thresholds and controllers are examined; the "
        "reported intervals are per-comparison descriptive statements, not "
        "multiplicity-corrected guarantees.",
        "- `Min`/`Max` and `q05` are observed sample values, not bounds. "
        "Survivor tails are separate from sampling uncertainty.",
        "",
    ]
    (directory / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
