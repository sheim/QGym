"""Paired analysis for reused and newly evaluated height-Q controllers.

Loads the existing baseline/original/evaldist blocks and the new mixed/fresh
blocks, verifies the shared state/seed/execution identity, and reports
individual-execution and contiguous batch16 paired statistics. No simulator is
constructed and no rollout is executed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from learning.offline.persistent_intervention import paired_survivor_tail_interval

N_STATES = 128
N_BLOCKS = 16
N_CALLS = 8
N_LANES = 8
N_EXECUTIONS = N_BLOCKS * N_CALLS * N_LANES
BATCH_SIZE = 16
N_BATCHES = N_EXECUTIONS // BATCH_SIZE
SEED_BASE = 7_100_000
SEED_STRIDE = 10
TIE_TOL = 1e-9
GAMMA = 0.99
HORIZON = 500
TAIL = GAMMA**HORIZON / (1.0 - GAMMA)
BOOT_DRAWS = 10_000
# Paired-gain state-cluster CI reuses the paired16 comparison seed so the
# original controller reproduces its reported interval exactly.
BOOT_SEED = 20_261_111

SUM_KEYS = (
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_seed(block: int, call: int) -> int:
    return SEED_BASE + SEED_STRIDE * (block * N_CALLS + call)


def cluster_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    index = rng.integers(0, values.size, size=(BOOT_DRAWS, values.size))
    draws = values[index].mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def fixed_state_ci(values: np.ndarray) -> list[float]:
    mean = values.mean(axis=1)
    se = values.std(axis=1, ddof=1) / math.sqrt(values.shape[1])
    width = (
        1.959963984540054
        * math.sqrt(float((se**2).mean()))
        / math.sqrt(values.shape[0])
    )
    return [float(mean.mean() - width), float(mean.mean() + width)]


def load_blocks(directory: Path, name: str) -> dict:
    blocks = []
    for block in range(N_BLOCKS):
        path = directory / f"{name}_{block:02d}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        data = torch.load(path, weights_only=True, map_location="cpu")
        if list(data["axes"]) != [
            "state128",
            "execution64",
            "optional_candidate8",
        ]:
            raise ValueError(f"{path}: unexpected axes {data['axes']}")
        seeds = [expected_seed(block, call) for call in range(N_CALLS)]
        if list(data["seeds"]) != seeds:
            raise ValueError(f"{path}: seed identity mismatch")
        if tuple(data["returns"].shape) != (N_STATES, N_CALLS * N_LANES):
            raise ValueError(f"{path}: unexpected block shape")
        if data["returns"].dtype != torch.float64:
            raise ValueError(f"{path}: returns must be float64")
        if bool(data["timeout_seen"].any()):
            raise ValueError(f"{path}: timeout observed in H500 rollout")
        if not torch.isfinite(data["returns"]).all():
            raise ValueError(f"{path}: non-finite return")
        decisions = data["selection_counts"].sum(-1)
        if not torch.equal(decisions, data["steps"]):
            raise ValueError(f"{path}: selection counts != live decisions")
        if name == "baseline" and int(data["selection_counts"][..., 1:].sum()):
            raise ValueError(f"{path}: baseline selected a non-candidate0 index")
        blocks.append(data)
    result = {
        key: torch.stack([block[key] for block in blocks])
        .permute(1, 0, 2)
        .flatten(1)
        .double()
        for key in SUM_KEYS
    }
    result["selection_counts"] = (
        torch.stack([block["selection_counts"] for block in blocks])
        .permute(1, 0, 2, 3)
        .flatten(1, 2)
    )
    result["seconds"] = [float(block["seconds"]) for block in blocks]
    result["seeds"] = [
        [expected_seed(block, call) for call in range(N_CALLS)]
        for block in range(N_BLOCKS)
    ]
    return result


def controller_summary(r: dict) -> dict:
    returns = r["returns"].numpy()
    steps = r["steps"].numpy()
    oob = r["oob"].numpy()
    survived = r["survived"].numpy()
    mean_state = returns.mean(axis=1)
    survivor_fraction = float(survived.mean())
    return {
        "mean_return": float(mean_state.mean()),
        "state_mean_sd": float(mean_state.std(ddof=1)),
        "state_mean_median": float(np.median(mean_state)),
        "state_mean_q05": float(np.quantile(mean_state, 0.05)),
        "state_mean_min": float(mean_state.min()),
        "state_mean_max": float(mean_state.max()),
        "execution_ci95": fixed_state_ci(returns),
        "state_cluster_ci95": cluster_ci(mean_state, BOOT_SEED + 10),
        "tail_interval": [
            float(mean_state.mean() - TAIL * survivor_fraction),
            float(mean_state.mean()),
        ],
        "survivor_fraction": survivor_fraction,
        "inside_band_step_fraction": float(1.0 - oob.sum() / steps.sum()),
        "fall_fraction": float(1.0 - survivor_fraction),
        "mean_steps": float(steps.mean()),
        "first_reward_negative_rate": float((r["first_rewards"].numpy() == -1).mean()),
        "first_step_termination_rate": float(r["first_terminated"].numpy().mean()),
        "xy_tracking_rmse": float(
            math.sqrt(r["xy_error_sq"].sum() / r["tracking_steps"].sum())
        ),
        "yaw_tracking_rmse": float(
            math.sqrt(r["yaw_error_sq"].sum() / r["tracking_steps"].sum())
        ),
        "speed_over_command": float(
            r["speed_sum"].sum() / r["command_speed_sum"].sum()
        ),
        "selection_counts_total": r["selection_counts"].sum().item(),
        "candidate0_fraction": float(
            r["selection_counts"][..., 0].sum() / r["selection_counts"].sum()
        ),
        "selection_fraction": (
            r["selection_counts"].sum((0, 1)) / r["selection_counts"].sum()
        ).tolist(),
        "mean_q_gain": float(r["q_gain_sum"].sum() / r["selection_counts"].sum()),
        "collection_seconds": float(sum(r["seconds"])),
    }


def paired_summary(r: dict, base: dict) -> dict:
    paired = (r["returns"] - base["returns"]).numpy()
    gain_state = paired.mean(axis=1)
    base_returns = base["returns"].numpy()
    batch_means = paired.reshape(N_STATES, N_BATCHES, BATCH_SIZE).mean(axis=2)
    batch_win = batch_means > TIE_TOL
    batch_tie = np.abs(batch_means) <= TIE_TOL
    batch_loss = batch_means < -TIE_TOL
    win = paired > TIE_TOL
    tie = np.abs(paired) <= TIE_TOL
    loss = paired < -TIE_TOL
    base_surv_state = base["survived"].numpy().mean(axis=1)
    surv_state = r["survived"].numpy().mean(axis=1)
    base_surv = float(base_surv_state.mean())
    surv = float(surv_state.mean())
    per_state_tail_lower = gain_state - TAIL * base_surv_state
    base_rate = (base["oob"] / base["steps"]).numpy()
    rate = (r["oob"] / r["steps"]).numpy()
    fall_delta = (1.0 - r["survived"].numpy()) - (1.0 - base["survived"].numpy())
    return {
        "paired_executions": int(paired.size),
        "paired_mean_gain": float(gain_state.mean()),
        "per_state_gain_sd": float(gain_state.std(ddof=1)),
        "per_state_gain_median": float(np.median(gain_state)),
        "per_state_gain_q05": float(np.quantile(gain_state, 0.05)),
        "per_state_gain_min": float(gain_state.min()),
        "per_state_gain_max": float(gain_state.max()),
        "per_state_gain_positive": int((gain_state > TIE_TOL).sum()),
        "per_state_gain_zero": int((np.abs(gain_state) <= TIE_TOL).sum()),
        "per_state_gain_negative": int((gain_state < -TIE_TOL).sum()),
        "execution_win_fraction": float(win.mean()),
        "execution_tie_fraction": float(tie.mean()),
        "execution_loss_fraction": float(loss.mean()),
        "execution_win_count": int(win.sum()),
        "execution_tie_count": int(tie.sum()),
        "execution_loss_count": int(loss.sum()),
        "paired_execution_ci95": fixed_state_ci(paired),
        "paired_state_cluster_ci95": cluster_ci(gain_state, BOOT_SEED),
        "batch16_positive_fraction": float(batch_win.mean()),
        "batch16_tie_fraction": float(batch_tie.mean()),
        "batch16_negative_fraction": float(batch_loss.mean()),
        "batch16_positive_count": int(batch_win.sum()),
        "batch16_tie_count": int(batch_tie.sum()),
        "batch16_negative_count": int(batch_loss.sum()),
        "batch16_mean_gain": float(batch_means.mean()),
        "batch16_pooled_sd": float(batch_means.std(ddof=1)),
        "batch16_mean_state_sd": float(batch_means.std(axis=1, ddof=1).mean()),
        "batch16_all_positive_states": int((batch_win.sum(axis=1) == N_BATCHES).sum()),
        "batch16_majority_positive_states": int(
            (batch_win.sum(axis=1) > N_BATCHES // 2).sum()
        ),
        "batch16_none_positive_states": int((batch_win.sum(axis=1) == 0).sum()),
        "batch16_positive_count_by_state": batch_win.sum(axis=1).tolist(),
        "execution_win_fraction_by_state": win.mean(axis=1).tolist(),
        "paired_tail_interval": list(
            paired_survivor_tail_interval(
                float(gain_state.mean()), base_surv, surv, TAIL
            )
        ),
        "per_state_tail_lower_positive": int((per_state_tail_lower > 0).sum()),
        "delta_inside_band_rate": float((rate - base_rate).mean()),
        "delta_inside_band_rate_state_ci95": cluster_ci(
            (rate - base_rate).mean(axis=1), BOOT_SEED + 30
        ),
        "delta_fall_rate": float(fall_delta.mean()),
        "delta_fall_rate_state_ci95": cluster_ci(
            fall_delta.mean(axis=1), BOOT_SEED + 40
        ),
        "base_return_checksum": float(base_returns.sum()),
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def render_report(
    summaries: dict[str, dict],
    paired: dict[str, dict],
    manifest: dict,
) -> str:
    names = ["original", "evaldist", "mixed", "fresh"]
    all_names = ["baseline", *names]
    lines = [
        "# Two additional height-Q critics: equal-epoch data comparison",
        "",
        "**Question.** On the same 128 fixed k=100 states and the same 1024 "
        "execution seeds, how do four frozen height-Q critics compare when each "
        "selects the maximum of eight fresh stochastic `model_600` actions at "
        "every decision? The new critics are fitted at equal 102.4 epochs on the "
        "existing 2M mixed replay and on 1M fresh frozen-policy transitions.",
        "",
        "**Scope and estimand.** Fixed-state policy intervention, H500, gamma "
        "0.99, binary height reward 0.32 +/- 0.05 m, absorbing true failure. "
        "This is not policy optimization, not `Q^pi` validation, and not an "
        "equal-data-size comparison (mixed has 2M transitions, fresh has 1M).",
        "",
        "## Data identity",
        "",
        f"- Baseline/original/evaldist blocks reused from "
        f"`{manifest['baseline_dir']}`; no rerun.",
        f"- New mixed/fresh blocks from `{manifest['new_dir']}`.",
        "- Every block stores `[state128, execution64]`; execution order is "
        "`e = call*8 + lane` and `seed = 7100000 + 10*(block*8 + call)`.",
        "- Primary batch16 is contiguous `e//16` (two calls x all eight lanes), "
        "64 disjoint batches per state.",
        "- Source insertion sha256 matches across reused and new runs; all "
        "controllers share the same 128 states and seed lists.",
        "",
        "## Per-controller results",
        "",
        "| Controller | Mean H500 return | Execution CI95 | State-cluster CI95 | "
        "Tail interval | Inside band | Falls | XY RMSE | Speed/cmd |",
        "|---|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for name in all_names:
        s = summaries[name]
        lines.append(
            f"| {name} | {fmt(s['mean_return'])} | "
            f"[{fmt(s['execution_ci95'][0])}, {fmt(s['execution_ci95'][1])}] | "
            f"[{fmt(s['state_cluster_ci95'][0])}, {fmt(s['state_cluster_ci95'][1])}] | "
            f"[{fmt(s['tail_interval'][0])}, {fmt(s['tail_interval'][1])}] | "
            f"{pct(s['inside_band_step_fraction'])} | {pct(s['fall_fraction'])} | "
            f"{fmt(s['xy_tracking_rmse'])} | {fmt(s['speed_over_command'])} |"
        )
    lines += [
        "",
        "## Paired gain versus plain baseline",
        "",
        "| Controller | Mean gain | Execution CI95 | State-cluster CI95 | "
        "Individual win/tie/loss | Batch16 positive | Paired tail interval |",
        "|---|---:|---|---|---|---:|---|",
    ]
    for name in names:
        p = paired[name]
        lines.append(
            f"| {name} | {fmt(p['paired_mean_gain'])} | "
            f"[{fmt(p['paired_execution_ci95'][0])}, "
            f"{fmt(p['paired_execution_ci95'][1])}] | "
            f"[{fmt(p['paired_state_cluster_ci95'][0])}, "
            f"{fmt(p['paired_state_cluster_ci95'][1])}] | "
            f"{pct(p['execution_win_fraction'])} / "
            f"{pct(p['execution_tie_fraction'])} / "
            f"{pct(p['execution_loss_fraction'])} | "
            f"{pct(p['batch16_positive_fraction'])} | "
            f"[{fmt(p['paired_tail_interval'][0])}, "
            f"{fmt(p['paired_tail_interval'][1])}] |"
        )
    lines += [
        "",
        "Cross-check: the `original` row reproduces the paired16 report's mean "
        "1.719137, individual 56.91/21.22/21.87%, batch16 85.49% and tail "
        "[1.0763, 2.3615]; the paired state CI uses the same bootstrap seed.",
        "",
        "## Behaviour changes and per-state distribution",
        "",
        "| Controller | Delta outside-band (pp) | State CI pp | Delta fall (pp) | "
        "State CI pp | Per-state mean gain positive | Batch16 all-positive states |",
        "|---|---:|---|---:|---|---:|---:|",
    ]
    for name in names:
        p = paired[name]
        lines.append(
            f"| {name} | {100.0 * p['delta_inside_band_rate']:+.2f} | "
            f"[{100.0 * p['delta_inside_band_rate_state_ci95'][0]:+.2f}, "
            f"{100.0 * p['delta_inside_band_rate_state_ci95'][1]:+.2f}] | "
            f"{100.0 * p['delta_fall_rate']:+.2f} | "
            f"[{100.0 * p['delta_fall_rate_state_ci95'][0]:+.2f}, "
            f"{100.0 * p['delta_fall_rate_state_ci95'][1]:+.2f}] | "
            f"{p['per_state_gain_positive']}/128 | "
            f"{p['batch16_all_positive_states']}/{N_STATES} |"
        )
    lines += [
        "",
        "## Reading the comparison",
        "",
    ]
    for name in names:
        p = paired[name]
        lines.append(
            f"- **{name}**: mean paired gain {fmt(p['paired_mean_gain'])}; "
            f"individual win/tie/loss {pct(p['execution_win_fraction'])}/"
            f"{pct(p['execution_tie_fraction'])}/"
            f"{pct(p['execution_loss_fraction'])}; batch16 positive "
            f"{pct(p['batch16_positive_fraction'])}; per-state means positive "
            f"{p['per_state_gain_positive']}/128; per-state tail lower bound "
            f"positive {p['per_state_tail_lower_positive']}/128."
        )
    lines += [
        "",
        "In the behaviour table, a negative delta outside-band rate means fewer "
        "out-of-band steps (an improvement); a positive delta fall rate means "
        "more falls. Both are controller minus baseline.",
        "",
        "The individual win fraction and the batch16 positive fraction are "
        "different estimands. Averaging 16 executions reduces sign noise but "
        "does not change the mean, estimate expected utility, or imply that a "
        "typical execution improves. Execution rows are not independent IID "
        "draws: they share fixed states and seed/lane structure. The "
        "state-cluster bootstrap is the honest across-state statement; the "
        "execution CI is conditional precision.",
        "",
        "The paired tail interval is asymmetric: the lower side uses the "
        "baseline survivor fraction and the upper side the controller survivor "
        "fraction. It is an omitted-tail uncertainty interval for the finite "
        "H500 return, not a sampling CI and not a noise ceiling.",
        "",
        "## Data and training budgets",
        "",
        f"- Fresh collection: {manifest['collection']['policy_steps']} policy "
        f"steps x {manifest['collection']['num_envs']} worlds = "
        f"{manifest['collection']['transitions_appended']} appended, "
        f"{manifest['collection']['retained_transitions']} retained, "
        f"{manifest['collection']['unique_episode_env_pairs']} unique "
        f"env/episode pairs, {manifest['collection']['terminated_rows']} true "
        f"terminations, {manifest['collection']['timed_out_rows']} timeouts, "
        f"{fmt(manifest['collection']['wall_seconds'], 2)} s.",
        f"- Mixed dataset: {manifest['mixed']['valid_count']} transitions; "
        f"800,000 updates x batch 256 = 102.4 epochs.",
        f"- Fresh dataset: {manifest['fresh']['valid_count']} transitions; "
        f"400,000 updates x batch 256 = 102.4 epochs.",
        f"- Training wall time: mixed "
        f"{fmt(manifest['training']['mixed_seconds'], 2)} s, fresh "
        f"{fmt(manifest['training']['fresh_seconds'], 2)} s.",
        "",
        "## Limitations",
        "",
        "- Four critics are compared on one frozen checkpoint, one reward, and "
        "128 fixed states from one source lane; this is not a state-population "
        "or checkpoint-generalization claim.",
        "- The mixed dataset contains early-policy and final-policy transitions; "
        "the fresh dataset is nominal `model_600` only. Their state/return "
        "distributions differ by construction, so a critic difference is not a "
        "clean data-size effect.",
        "- Persistent Q-selected actions leave the nominal on-policy "
        "distribution; the frozen critics' targets are off-policy once the "
        "branch deviates. No claim that repeated intervention estimates `Q^pi` "
        "or that any critic is 'correct' is made.",
        "- The return is truncated at H500; the omitted survivor tail is bounded "
        "only by the reported interval.",
        "- Runtime model identity is the session configuration "
        "`opencode-go/deepseek-v4.1-flash`; it is not independently verified "
        "inside the process.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--new_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    baseline_dir = Path(args.baseline_dir)
    new_dir = Path(args.new_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    data = {
        name: load_blocks(baseline_dir, name)
        for name in ("baseline", "original", "evaldist")
    }
    data["mixed"] = load_blocks(new_dir, "mixed")
    data["fresh"] = load_blocks(new_dir, "fresh")
    base = data["baseline"]
    experiment = new_dir.parent
    old_provenance = json.loads((baseline_dir / "provenance.json").read_text())
    new_provenance = json.loads((new_dir / "provenance.json").read_text())
    for key in ("source_sha256", "checkpoint_sha256"):
        if old_provenance[key] != new_provenance[key]:
            raise ValueError(f"reused and new runs disagree on {key}")
    for name, r in data.items():
        if r["returns"].shape != (N_STATES, N_EXECUTIONS):
            raise ValueError(f"{name}: unexpected return shape")
        if r["seeds"] != base["seeds"]:
            raise ValueError(f"{name}: seed identity differs from baseline")
        if r["steps"].shape != base["steps"].shape:
            raise ValueError(f"{name}: execution grid differs")
        if not torch.isin(r["first_rewards"], torch.tensor([-1.0, 0.0])).all():
            raise ValueError(f"{name}: nonbinary first reward")
    for name in ("original", "evaldist", "mixed", "fresh"):
        if int(data[name]["selection_counts"][..., 1:].sum()) == 0:
            raise ValueError(f"{name}: Q controller never left candidate0")

    summaries = {name: controller_summary(r) for name, r in data.items()}
    paired = {
        name: paired_summary(data[name], base)
        for name in ("original", "evaldist", "mixed", "fresh")
    }
    manifest = {
        "baseline_dir": str(baseline_dir),
        "new_dir": str(new_dir),
        "baseline_code_hashes": {
            str(path): sha256_file(path)
            for path in (
                baseline_dir / "code.json",
                baseline_dir / "validation.json",
                baseline_dir / "provenance.json",
            )
        },
        "new_code_hashes": {
            str(path): sha256_file(path)
            for path in (
                new_dir / "code.json",
                new_dir / "validation.json",
                new_dir / "provenance.json",
            )
        },
        "blocks": {
            name: {
                str(path): sha256_file(path)
                for path in sorted(new_dir.glob(f"{name}_*.pt"))
            }
            for name in ("mixed", "fresh")
        },
        "tail_width": TAIL,
        "tie_tolerance": TIE_TOL,
        "collection": json.loads(
            (experiment / "fresh_1m" / "collect_provenance.json").read_text()
        ),
        "mixed": json.loads((experiment / "data_audit.json").read_text())["mixed"],
        "fresh": json.loads((experiment / "data_audit.json").read_text())["fresh"],
        "training": {
            "mixed_seconds": 745.76,
            "fresh_seconds": 371.89,
        },
    }
    results = {
        "controllers": summaries,
        "paired": paired,
        "identity": {
            "n_states": N_STATES,
            "n_executions": N_EXECUTIONS,
            "n_batches": N_BATCHES,
            "seed_formula": "7100000 + 10*(block*8 + call)",
            "execution_order": "e = call*8 + lane (block-major)",
            "primary_batch": "contiguous e//16",
            "source_insertion_sha256": json.loads(
                (new_dir / "provenance.json").read_text()
            )["source_sha256"],
        },
    }
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    rows = []
    for state in range(N_STATES):
        row = {"state": state}
        for name in ("baseline", "original", "evaldist", "mixed", "fresh"):
            row[f"return_{name}"] = float(data[name]["returns"][state].mean())
            if name == "baseline":
                continue
            gains = (data[name]["returns"][state] - base["returns"][state]).numpy()
            batches = gains.reshape(N_BATCHES, BATCH_SIZE).mean(axis=1)
            row[f"gain_{name}"] = float(gains.mean())
            row[f"exec_win_fraction_{name}"] = float((gains > TIE_TOL).mean())
            row[f"batch16_positive_{name}"] = int((batches > TIE_TOL).sum())
            row[f"batch16_mean_gain_{name}"] = float(batches.mean())
        rows.append(row)
    fields = ["state", "return_baseline"]
    for name in ("original", "evaldist", "mixed", "fresh"):
        fields += [
            f"return_{name}",
            f"gain_{name}",
            f"exec_win_fraction_{name}",
            f"batch16_positive_{name}",
            f"batch16_mean_gain_{name}",
        ]
    write_csv(output / "per_state.csv", rows, fields)
    report = render_report(summaries, paired, manifest)
    (output / "REPORT.md").write_text(report)
    print(json.dumps({"controllers": summaries, "paired": paired}, indent=2))


if __name__ == "__main__":
    main()
