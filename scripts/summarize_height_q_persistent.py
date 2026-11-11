"""Audit and summarize persistent-intervention chunks without simulation."""

import json
import sys
from pathlib import Path

import torch

from learning.offline.persistent_intervention import paired_survivor_tail_interval

CONTROLLERS = ("baseline", "original", "evaldist")
TAIL = 0.99**500 / 0.01


def load(directory):
    out = {}
    for name in CONTROLLERS:
        blocks = [
            torch.load(directory / f"{name}_{b:02d}.pt", weights_only=True)
            for b in range(16)
        ]
        keys = (
            "returns",
            "steps",
            "oob",
            "survived",
            "first_rewards",
            "first_terminated",
            "tracking_steps",
            "xy_error_sq",
            "yaw_error_sq",
            "speed_sum",
            "command_speed_sum",
            "q_gain_sum",
            "timeout_seen",
        )
        # [block, state, execution] -> [state, block*execution]; the block-major
        # execution order is identical for every controller at a shared seed.
        out[name] = {
            k: torch.stack([b[k] for b in blocks]).permute(1, 0, 2).flatten(1).double()
            for k in keys
        }
        out[name]["selection_counts"] = (
            torch.stack([b["selection_counts"] for b in blocks])
            .permute(1, 0, 2, 3)
            .flatten(1, 2)
        )
        out[name]["seconds"] = [b["seconds"] for b in blocks]
    return out


def interval(values, seed, draws=10_000):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(values), (draws, len(values)), generator=g)
    means = values[idx].mean(-1)
    return [float(means.quantile(0.025)), float(means.quantile(0.975))]


def summarize(data):
    base = data["baseline"]
    report = {"tail_width": TAIL, "validated_executions_per_state": {}}
    for name in CONTROLLERS:
        r = data[name]
        if r["returns"].shape != (128, 1024):
            raise ValueError(f"{name} shape {tuple(r['returns'].shape)}")
        if not torch.isfinite(r["returns"]).all():
            raise ValueError("nonfinite return")
        if r["timeout_seen"].any():
            raise ValueError("timeout observed")
        if not torch.isin(r["first_rewards"], torch.tensor([-1.0, 0.0])).all():
            raise ValueError("nonbinary first reward")
        decisions = r["selection_counts"].sum(-1)
        if not torch.equal(decisions, r["steps"]):
            raise ValueError("selection count does not match live decisions")
        report["validated_executions_per_state"][name] = int(r["returns"].shape[1])
        mean = r["returns"].mean(1)
        se = r["returns"].std(1) / 1024**0.5
        surv = r["survived"].double().mean(1)
        report[name] = {
            "mean_return": float(mean.mean()),
            "state_mean_sd": float(mean.std()),
            "state_mean_q05": float(mean.quantile(0.05)),
            "execution_se_mean": float((se**2).mean().sqrt()),
            "execution_ci": [
                float(mean.mean() - 1.96 * (se**2).mean().sqrt() / 128**0.5),
                float(mean.mean() + 1.96 * (se**2).mean().sqrt() / 128**0.5),
            ],
            "state_cluster_ci": interval(mean, 202611101),
            "tail_interval": [
                float(mean.mean() - TAIL * surv.mean()),
                float(mean.mean()),
            ],
            "inside_band_step_fraction": float(1 - (r["oob"].sum() / r["steps"].sum())),
            "fall_fraction": float(1 - r["survived"].double().mean()),
            "mean_steps": float(r["steps"].mean()),
            "first_reward_negative_rate": float(
                (r["first_rewards"] == -1).double().mean()
            ),
            "first_step_termination_rate": float(r["first_terminated"].double().mean()),
            "xy_tracking_rmse": float(
                (r["xy_error_sq"].sum() / r["tracking_steps"].sum()).sqrt()
            ),
            "yaw_tracking_rmse": float(
                (r["yaw_error_sq"].sum() / r["tracking_steps"].sum()).sqrt()
            ),
            "speed_over_command": float(
                r["speed_sum"].sum() / r["command_speed_sum"].sum()
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
        if name != "baseline":
            paired = r["returns"] - base["returns"]
            gain = paired.mean(1)
            gains = gain.mean()
            se_d = paired.std(1) / 1024**0.5
            tol = 1e-9
            win = (gain > tol).sum().item()
            lose = (gain < -tol).sum().item()
            tie = 128 - win - lose
            report[name].update(
                {
                    "paired_mean_gain": float(gains),
                    "per_state_gain_sd": float(gain.std()),
                    "per_state_gain_median": float(gain.median()),
                    "per_state_gain_q05": float(gain.quantile(0.05)),
                    "per_state_gain_min": float(gain.min()),
                    "per_state_gain_max": float(gain.max()),
                    "execution_diff_sd": float(paired.std()),
                    "paired_execution_ci": [
                        float(gains - 1.96 * (se_d**2).mean().sqrt() / 128**0.5),
                        float(gains + 1.96 * (se_d**2).mean().sqrt() / 128**0.5),
                    ],
                    "paired_state_cluster_ci": interval(gain, 202611102),
                    "wins": win,
                    "ties": tie,
                    "losses": lose,
                    "prob_state_win": win / 128,
                    "prob_execution_win": float((paired > 0).double().mean()),
                    # Each execution's omitted survivor tail is [-TAIL, 0]; the
                    # asymmetric bound is centralized in the helper so both
                    # sides cannot double-count survivor fractions.
                    "paired_tail_interval": list(
                        paired_survivor_tail_interval(
                            gains,
                            float(base["survived"].double().mean(1).mean()),
                            float(surv.mean()),
                            TAIL,
                        )
                    ),
                    # Explicit sign conventions: delta_oob < 0 means fewer
                    # outside-band steps; delta_fall > 0 means more falls.
                    "delta_oob_rate": float(
                        ((r["oob"] / r["steps"]) - (base["oob"] / base["steps"])).mean()
                    ),
                    "delta_oob_rate_state_ci": interval(
                        ((r["oob"] / r["steps"]) - (base["oob"] / base["steps"])).mean(
                            1
                        ),
                        202611103,
                    ),
                    "delta_fall_rate": float(
                        (
                            (1 - r["survived"].double())
                            - (1 - base["survived"].double())
                        ).mean()
                    ),
                    "delta_fall_rate_state_ci": interval(
                        (
                            (1 - r["survived"].double())
                            - (1 - base["survived"].double())
                        ).mean(1),
                        202611104,
                    ),
                }
            )
    return report


def main():
    directory = Path(sys.argv[1])
    data = load(directory)
    report = summarize(data)
    (directory / "results.json").write_text(json.dumps(report, indent=2))
    gain = (data["original"]["returns"] - data["baseline"]["returns"]).mean(1)
    rows = ["state,return_baseline,return_original,return_evaldist,gain_original"]
    for s in range(128):
        rows.append(
            f"{s},{data['baseline']['returns'][s].mean():.6f},"
            f"{data['original']['returns'][s].mean():.6f},"
            f"{data['evaldist']['returns'][s].mean():.6f},{gain[s]:.6f}"
        )
    (directory / "per_state.csv").write_text("\n".join(rows) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
