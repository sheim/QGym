"""Summarize a nominal-trajectory Q pilot and propose fixed thresholds.

The five thresholds are quantiles of the decision-weighted exact-mean Q
distribution. They are chosen from Q values, never from realised gains or
returns. Feasibility rates describe the nominal distribution; persistent
intervention changes the state distribution.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

QUANTILES = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
THRESHOLD_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def _summarize(values, quantiles=QUANTILES):
    values = torch.as_tensor(values).reshape(-1).float()
    count = values.numel()
    if count == 0:
        return {"count": 0}
    ordered, _ = torch.sort(values)
    indices = (torch.tensor(quantiles) * (count - 1)).round().long()
    return {
        "count": int(count),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=True)) if count > 1 else 0.0,
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "quantiles": {
            f"{q:.3f}": float(ordered[index]) for q, index in zip(quantiles, indices)
        },
    }


def _threshold_rates(qmat, threshold):
    mean_q = qmat[:, 0]
    max_all = qmat.amax(-1)
    max_stochastic = qmat[:, 1:].amax(-1)
    mean_pass = (mean_q >= threshold).float().mean()
    any_stochastic = (max_stochastic >= threshold).float().mean()
    any_all = (max_all >= threshold).float().mean()
    fallback = 1.0 - float(any_all)
    stochastic_selected = (
        ((max_stochastic >= threshold) & (mean_q < threshold)).float().mean()
    )
    return {
        "mean_feasible_rate": float(mean_pass),
        "any_stochastic_feasible_rate": float(any_stochastic),
        "any_candidate_feasible_rate": float(any_all),
        "no_feasible_fallback_rate": fallback,
        "mean_selected_rate": float(mean_pass),
        "closest_stochastic_selected_rate": float(stochastic_selected),
        "per_stochastic_candidate_feasible_rate": float(
            (qmat[:, 1:] >= threshold).float().mean()
        ),
    }


def _per_step(q, alive):
    steps = q.shape[1]
    out = {
        "step": [],
        "active_count": [],
        "q_mean_action_mean": [],
        "q_mean_action_q05": [],
        "q_mean_action_q50": [],
        "q_mean_action_q95": [],
        "q_max9_mean": [],
        "q_min9_mean": [],
    }
    for step in range(steps):
        mask = alive[:, step, :]
        values = q[:, step, :, 0][mask]
        max9 = q[:, step, :, :].amax(-1)[mask]
        min9 = q[:, step, :, :].amin(-1)[mask]
        count = values.numel()
        out["step"].append(step)
        out["active_count"].append(int(mask.sum()))
        if count == 0:
            for key in (
                "q_mean_action_mean",
                "q_mean_action_q05",
                "q_mean_action_q50",
                "q_mean_action_q95",
                "q_max9_mean",
                "q_min9_mean",
            ):
                out[key].append(None)
            continue
        ordered, _ = torch.sort(values)
        quantiles = (torch.tensor([0.05, 0.5, 0.95]) * (count - 1)).round().long()
        out["q_mean_action_mean"].append(float(values.mean()))
        out["q_mean_action_q05"].append(float(ordered[quantiles[0]]))
        out["q_mean_action_q50"].append(float(ordered[quantiles[1]]))
        out["q_mean_action_q95"].append(float(ordered[quantiles[2]]))
        out["q_max9_mean"].append(float(max9.mean()))
        out["q_min9_mean"].append(float(min9.mean()))
    return out


def _write_csv(path, table):
    columns = list(table)
    with path.open("w") as f:
        f.write(",".join(columns) + "\n")
        for row in zip(*(table[column] for column in columns)):
            f.write(",".join(str(value) for value in row) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--directory",
        default="/tmp/opencode/height-q-threshold-pilot-20261111-v1",
    )
    args = p.parse_args()
    directory = Path(args.directory)
    files = sorted(directory.glob("call_*.pt"))
    if not files:
        raise FileNotFoundError("no call_*.pt files found")
    rows = [torch.load(path, weights_only=True) for path in files]
    seeds = [row["seed"] for row in rows]
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate call seeds")
    q = torch.stack([row["q"].float() for row in rows])
    alive = torch.stack([row["alive"] for row in rows])
    if q.shape[-1] != 9:
        raise ValueError("expected nine candidates")
    if not torch.isfinite(q).all():
        raise ValueError("non-finite Q values")
    calls, steps, worlds, _ = q.shape
    if worlds % 128:
        raise ValueError("world count must contain 128 states")
    lanes = worlds // 128

    flat_q = q.reshape(-1, 9)
    flat_alive = alive.reshape(-1)
    active_q = flat_q[flat_alive]
    if active_q.shape[0] == 0:
        raise ValueError("no active decisions")

    # Decision weighting: every active (call, lane, state, step) equally.
    q_mean_action = active_q[:, 0]
    q_first_stochastic = active_q[:, 1]
    q_max9 = active_q.amax(-1)
    q_min9 = active_q.amin(-1)
    q_max8 = active_q[:, 1:].amax(-1)
    q_min8 = active_q[:, 1:].amin(-1)
    q_spread9 = q_max9 - q_min9

    # State-step weighting: first average over calls/lanes for each (state, step).
    q_by_state_step = q.reshape(calls, steps, lanes, 128, 9)
    alive_by_state_step = alive.reshape(calls, steps, lanes, 128)
    state_step_sum = (q_by_state_step * alive_by_state_step[..., None]).sum((0, 2))
    state_step_count = alive_by_state_step.sum((0, 2))
    state_step_mean = state_step_sum / state_step_count[..., None].clamp_min(1)
    state_step_valid = state_step_count.reshape(-1) > 0
    state_step_q = state_step_mean.reshape(-1, 9)[state_step_valid]

    # State weighting: average over all steps/executions for each state.
    q_by_state = q.reshape(calls, steps, lanes, 128, 9)
    alive_by_state = alive.reshape(calls, steps, lanes, 128)
    state_sum = (q_by_state * alive_by_state[..., None]).sum((0, 1, 2))
    state_count = alive_by_state.sum((0, 1, 2)).reshape(128, 1)
    state_q = state_sum / state_count.clamp_min(1)

    per_step = _per_step(q, alive)
    _write_csv(directory / "per_step.csv", per_step)

    mean_values = q_mean_action
    threshold_values = torch.quantile(mean_values, torch.tensor(THRESHOLD_QUANTILES))
    thresholds = None
    for decimals in (3, 4, 5):
        candidate = [round(float(value), decimals) for value in threshold_values]
        if len(set(candidate)) == len(candidate):
            thresholds = candidate
            break
    if thresholds is None:
        thresholds = [float(value) for value in threshold_values]

    proposals = []
    for source_quantile, threshold in zip(THRESHOLD_QUANTILES, thresholds):
        decision = _threshold_rates(active_q, threshold)
        state_step = _threshold_rates(state_step_q, threshold)
        if (
            abs(
                decision["mean_selected_rate"]
                + decision["closest_stochastic_selected_rate"]
                + decision["no_feasible_fallback_rate"]
                - 1.0
            )
            > 1e-6
        ):
            raise RuntimeError("threshold selection decomposition does not sum to one")
        proposals.append(
            {
                "threshold_c": threshold,
                "source": f"Q(mean) quantile {source_quantile:.2f}",
                "decision_weighted": decision,
                "state_step_weighted": state_step,
            }
        )

    returns = torch.cat([row["returns"].double() for row in rows])
    oob = torch.cat([row["oob"].double() for row in rows])
    steps_sum = torch.cat([row["steps"].double() for row in rows])
    first_rewards = torch.cat([row["first_rewards"].double() for row in rows])
    survived = torch.cat([row["survived"] for row in rows])
    timeout_count = int(sum(int(row["timeout_seen"].sum()) for row in rows))
    if timeout_count:
        raise RuntimeError("administrative timeouts in pilot data")

    provenance_gamma = float(
        json.loads((directory / "provenance.json").read_text())["gamma"]
    )
    known_low = -1.0 / (1.0 - provenance_gamma)
    known_high = 0.0

    def range_diagnostics(values):
        return {
            "known_continuous_range": [known_low, known_high],
            "below_known_low_rate": float((values < known_low).double().mean()),
            "above_known_high_rate": float((values > known_high).double().mean()),
            "raw_min": float(values.min()),
            "raw_max": float(values.max()),
        }

    summary = {
        "directory": str(directory),
        "calls": calls,
        "executions_per_state": calls * lanes,
        "states": 128,
        "lanes_per_call": lanes,
        "horizon": steps,
        "active_decisions": int(active_q.shape[0]),
        "active_state_steps": int(state_step_valid.sum()),
        "weighting": {
            "decision": "each active (call, lane, state, step) equally",
            "state_step": (
                "average over calls/lanes for each (state, step), then each "
                "state-step equally"
            ),
            "state": (
                "average over steps/calls/lanes for each state, then each state equally"
            ),
            "pooled_candidate": "each active (decision, candidate) equally",
            "masking": "Q recorded before the step for alive worlds only",
        },
        "distributions": {
            "decision_q_mean_action": _summarize(q_mean_action),
            "decision_q_first_stochastic": _summarize(q_first_stochastic),
            "decision_q_max9": _summarize(q_max9),
            "decision_q_min9": _summarize(q_min9),
            "decision_q_max8": _summarize(q_max8),
            "decision_q_min8": _summarize(q_min8),
            "decision_q_spread9": _summarize(q_spread9),
            "pooled_candidate_q9": _summarize(active_q.flatten()),
            "pooled_stochastic_q8": _summarize(active_q[:, 1:].flatten()),
            "state_step_q_mean_action": _summarize(state_step_q[:, 0]),
            "state_step_q_max9": _summarize(state_step_q.amax(-1)),
            "state_step_q_min9": _summarize(state_step_q.amin(-1)),
            "state_q_mean_action": _summarize(state_q[:, 0]),
            "state_q_max9": _summarize(state_q.amax(-1)),
        },
        "outcomes": {
            "mean_return": float(returns.mean()),
            "return_sd": float(returns.std(unbiased=True)),
            "return_min": float(returns.min()),
            "return_max": float(returns.max()),
            "termination_fraction": float(1.0 - survived.double().mean()),
            "inside_band_step_fraction": float(1.0 - oob.sum() / steps_sum.sum()),
            "first_reward_mean": float(first_rewards.mean()),
            "first_reward_negative_rate": float((first_rewards < 0).double().mean()),
            "mean_active_steps": float(steps_sum.mean()),
            "timeouts": timeout_count,
        },
        "q_range_diagnostics": {
            "decision_q_mean_action": range_diagnostics(q_mean_action),
            "pooled_candidate_q9": range_diagnostics(active_q.flatten()),
        },
        "per_step_csv": "per_step.csv",
        "proposed_thresholds": proposals,
        "provenance": json.loads((directory / "provenance.json").read_text()),
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2))
    (directory / "thresholds.json").write_text(
        json.dumps(
            {
                "selection": "quantiles of decision-weighted Q(exact policy mean)",
                "threshold_quantiles": list(THRESHOLD_QUANTILES),
                "thresholds": thresholds,
                "proposals": proposals,
            },
            indent=2,
        )
    )
    compact = {
        "q_mean_action": q_mean_action,
        "q_first_stochastic": q_first_stochastic,
        "q_max9": q_max9,
        "q_min9": q_min9,
        "q_max8": q_max8,
        "q_min8": q_min8,
        "q_spread9": q_spread9,
        "state_step_q": state_step_q,
        "state_q": state_q,
    }
    torch.save(compact, directory / "distribution.pt")
    with (directory / "commands.txt").open("a") as f:
        f.write(
            "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen python "
            + " ".join(sys.argv)
            + "\n"
        )
    print(
        json.dumps(
            {
                "calls": summary["calls"],
                "executions_per_state": summary["executions_per_state"],
                "active_decisions": summary["active_decisions"],
                "outcomes": summary["outcomes"],
                "q_range_diagnostics": summary["q_range_diagnostics"],
                "thresholds": summary["proposed_thresholds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
