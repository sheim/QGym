"""Summarize persistent yaw-tracking intervention chunks without simulation.

Compares every controller with the plain baseline on the same fixed states and
seeds. Returns are finite-horizon discounted sums of a reward in [0, 1] with a
zero terminal value, so a surviving execution omits a nonnegative tail bounded
by ``gamma**H / (1 - gamma)``; there is no absorbing tail to add.
"""

import argparse
import json
from pathlib import Path

import torch

from learning.offline.tradeoff_analysis import paired_t_summary

RETURN_KEYS = (
    "returns",
    "steps",
    "survived",
    "first_rewards",
    "first_terminated",
    "tracking_steps",
    "yaw_error_sq",
    "q_gain_sum",
    "timeout_seen",
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory")
    p.add_argument(
        "--controllers",
        nargs="+",
        default=None,
        help="Controller names; defaults to provenance.json.",
    )
    return p


def load(directory: Path, controllers: list[str], blocks: int) -> dict:
    out = {}
    for name in controllers:
        rows = [
            torch.load(directory / f"{name}_{b:02d}.pt", weights_only=True)
            for b in range(blocks)
        ]
        # [block, state, execution] -> [state, block*execution]; the block-major
        # execution order is identical for every controller at a shared seed.
        out[name] = {
            k: torch.stack([b[k] for b in rows]).permute(1, 0, 2).flatten(1).double()
            for k in RETURN_KEYS
        }
        out[name]["selection_counts"] = (
            torch.stack([b["selection_counts"] for b in rows])
            .permute(1, 0, 2, 3)
            .flatten(1, 2)
        )
        for extra in ("fallback_sum", "selected_q_sum", "attempt_counts"):
            if extra in rows[0]:
                stacked = torch.stack([b[extra] for b in rows])
                out[name][extra] = (
                    stacked.permute(1, 0, 2, *range(3, stacked.ndim))
                    .flatten(1, 2)
                    .double()
                )
        out[name]["seconds"] = [b["seconds"] for b in rows]
    return out


def controllers_from_provenance(directory: Path) -> list[str]:
    provenance = json.loads((directory / "provenance.json").read_text())
    names = ["baseline"]
    if provenance["skip_baseline"]:
        raise ValueError("provenance records --skip_baseline; name controllers")
    names.extend(provenance["artifact_names"])
    for threshold in provenance["thresholds"]:
        names.append(f"rejection_c{threshold:g}")
    return names


def block_count(directory: Path, controllers: list[str]) -> int:
    blocks = 0
    while (directory / f"{controllers[0]}_{blocks:02d}.pt").is_file():
        blocks += 1
    if blocks == 0:
        raise FileNotFoundError("no result blocks found")
    for name in controllers:
        for b in range(blocks):
            if not (directory / f"{name}_{b:02d}.pt").is_file():
                raise FileNotFoundError(f"missing {name}_{b:02d}.pt")
        if (directory / f"{name}_{blocks:02d}.pt").is_file():
            raise ValueError(f"{name} has more blocks than {controllers[0]}")
    return blocks


def interval(values, seed, draws=10_000):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(values), (draws, len(values)), generator=g)
    means = values[idx].mean(-1)
    return [float(means.quantile(0.025)), float(means.quantile(0.975))]


def summarize(data: dict, controllers: list[str]) -> dict:
    base = data["baseline"]
    states, executions = base["returns"].shape
    if states != 128:
        raise ValueError(f"unexpected state count {states}")
    tail_bound = 0.99**500 / 0.01
    report = {
        "return_semantics": (
            "sum_t gamma**t r'_t, r' in [0,1], exactly 0 on true termination, "
            "no absorbing tail; horizon 500"
        ),
        "survivor_tail_upper_bound_per_execution": tail_bound,
        "validated_executions_per_state": executions,
        "controllers": controllers,
    }
    for name in controllers:
        r = data[name]
        if r["returns"].shape != (states, executions):
            raise ValueError(f"{name} shape {tuple(r['returns'].shape)}")
        if not torch.isfinite(r["returns"]).all():
            raise ValueError("nonfinite return")
        if r["timeout_seen"].any():
            raise ValueError("timeout observed")
        if not ((r["first_rewards"] >= 0) & (r["first_rewards"] <= 1)).all():
            raise ValueError("first reward outside [0, 1]")
        decisions = r["selection_counts"].sum(-1)
        if not torch.equal(decisions, r["steps"]):
            raise ValueError("selection count does not match live decisions")
        mean = r["returns"].mean(1)
        se = r["returns"].std(1) / executions**0.5
        surv = r["survived"].double()
        report[name] = {
            "mean_return": float(mean.mean()),
            "state_mean_sd": float(mean.std()),
            "state_mean_q05": float(mean.quantile(0.05)),
            "execution_se_mean": float((se**2).mean().sqrt()),
            "state_cluster_ci": interval(mean, 202611121),
            "survivor_fraction": float(surv.mean()),
            "survivor_tail_interval": [
                float(mean.mean()),
                float(mean.mean() + tail_bound * surv.mean(1).mean()),
            ],
            "mean_steps": float(r["steps"].mean()),
            "mean_reward_per_step": float(
                (r["returns"] / r["steps"].clamp(min=1)).mean()
            ),
            "first_reward_mean": float(r["first_rewards"].mean()),
            "first_step_termination_rate": float(r["first_terminated"].double().mean()),
            "yaw_tracking_rmse": float(
                (r["yaw_error_sq"].sum() / r["tracking_steps"].sum()).sqrt()
            ),
            "selection_fraction": (
                r["selection_counts"].sum((0, 1)) / r["selection_counts"].sum()
            ).tolist(),
            "candidate0_fraction": float(
                r["selection_counts"][..., 0].sum() / r["selection_counts"].sum()
            ),
            "mean_q_gain": float(r["q_gain_sum"].sum() / r["selection_counts"].sum()),
            "q_advantage_is_per_decision_not_cumulative": True,
            "block_seconds": r["seconds"],
            "collection_seconds": float(sum(r["seconds"])),
        }
        if "fallback_sum" in r:
            report[name]["fallback_fraction"] = float(
                r["fallback_sum"].sum() / r["selection_counts"].sum()
            )
            report[name]["mean_selected_q"] = float(
                r["selected_q_sum"].sum() / r["selection_counts"].sum()
            )
        if name == "baseline":
            continue
        paired = r["returns"] - base["returns"]
        gain = paired.mean(1)
        gains = float(gain.mean())
        se_d = paired.std(1) / executions**0.5
        tol = 1e-9
        win = int((gain > tol).sum())
        lose = int((gain < -tol).sum())
        tie = states - win - lose
        t_summary = paired_t_summary(gain)
        report[name].update(
            {
                "paired_mean_gain": gains,
                "per_state_gain_sd": float(gain.std()),
                "per_state_gain_median": float(gain.median()),
                "per_state_gain_q05": float(gain.quantile(0.05)),
                "per_state_gain_min": float(gain.min()),
                "per_state_gain_max": float(gain.max()),
                "execution_diff_sd": float(paired.std()),
                "paired_execution_ci": [
                    float(gains - 1.96 * (se_d**2).mean().sqrt() / states**0.5),
                    float(gains + 1.96 * (se_d**2).mean().sqrt() / states**0.5),
                ],
                "paired_state_cluster_ci": interval(gain, 202611122),
                "paired_t_test": t_summary,
                "wins": win,
                "ties": tie,
                "losses": lose,
                "prob_state_win": win / states,
                "prob_execution_win": float((paired > 0).double().mean()),
                "delta_survivor_rate": float(
                    (surv.mean(1) - base["survived"].double().mean(1)).mean()
                ),
                "delta_yaw_tracking_rmse": float(
                    (
                        (r["yaw_error_sq"].sum(1) / r["tracking_steps"].sum(1)).sqrt()
                        - (
                            base["yaw_error_sq"].sum(1) / base["tracking_steps"].sum(1)
                        ).sqrt()
                    ).mean()
                ),
            }
        )
    return report


def main():
    args = build_parser().parse_args()
    directory = Path(args.directory)
    controllers = args.controllers or controllers_from_provenance(directory)
    if controllers[0] != "baseline":
        controllers = ["baseline", *controllers]
    blocks = block_count(directory, controllers)
    data = load(directory, controllers, blocks)
    report = summarize(data, controllers)
    report["blocks"] = blocks
    (directory / "results.json").write_text(json.dumps(report, indent=2))
    header = ["state", "return_baseline"]
    header += [f"return_{name}" for name in controllers[1:]]
    header += [f"gain_{name}" for name in controllers[1:]]
    rows = [",".join(header)]
    for state in range(128):
        baseline = data["baseline"]["returns"][state].mean()
        values = [f"{state}", f"{baseline:.6f}"]
        values += [
            f"{data[name]['returns'][state].mean():.6f}" for name in controllers[1:]
        ]
        values += [
            f"{data[name]['returns'][state].mean() - baseline:.6f}"
            for name in controllers[1:]
        ]
        rows.append(",".join(values))
    (directory / "per_state.csv").write_text("\n".join(rows) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
