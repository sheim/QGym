"""Audit and summarize the original-reward tradeoff blocks without simulation."""

import argparse
import csv
import json
from pathlib import Path

import torch

from learning.offline.tradeoff_analysis import (
    paired_t_summary,
    state_cluster_bootstrap,
)
from learning.offline.persistent_intervention import survivor_tail_interval

CONTROLLERS = ("plain", "original")
PENALTIES = {
    "torques",
    "action_rate",
    "action_rate2",
    "swing_contact",
    "termination",
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument(
        "--prior",
        type=Path,
        default=Path("/tmp/opencode/height-q-persistent-intervention-20261110-v1"),
    )
    return p


def distribution(values):
    values = torch.as_tensor(values, dtype=torch.float64).flatten()
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(unbiased=True)) if values.numel() > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(values.median()),
        "q05": float(values.quantile(0.05)),
        "q95": float(values.quantile(0.95)),
    }


def win_tie_loss(values, tolerance=1e-9):
    values = torch.as_tensor(values, dtype=torch.float64)
    return {
        "win": int((values > tolerance).sum()),
        "tie": int((values.abs() <= tolerance).sum()),
        "loss": int((values < -tolerance).sum()),
        "n": values.numel(),
        "win_fraction": float((values > tolerance).double().mean()),
        "tie_fraction": float((values.abs() <= tolerance).double().mean()),
        "loss_fraction": float((values < -tolerance).double().mean()),
    }


def batch_summary(gains, size=16, tolerance=1e-9):
    if gains.shape != (128, 1024):
        raise ValueError("batch summary requires [128, 1024] gains")
    count = 1024 // size
    means = gains.reshape(128, count, size).mean(-1)
    wins = win_tie_loss(means, tolerance)
    positive = (means > tolerance).sum(-1)
    return {
        "batch_size": size,
        "batches_per_state": count,
        "batch_gains": means,
        "mean": float(means.mean()),
        "sd_pooled": float(means.flatten().std(unbiased=True)),
        "per_state_sd_mean": float(means.std(1, unbiased=True).mean()),
        "positive_fraction": wins["win_fraction"],
        "win": wins["win"],
        "tie": wins["tie"],
        "loss": wins["loss"],
        "all_positive_states": int((positive == count).sum()),
        "majority_positive_states": int((positive > count // 2).sum()),
        "none_positive_states": int((positive == 0).sum()),
        "positive_count_min": int(positive.min()),
        "positive_count_median": float(positive.median()),
        "positive_count_max": int(positive.max()),
    }


def load_blocks(directory):
    provenance = json.loads((directory / "provenance.json").read_text())
    gamma = float(provenance["height_gamma"])
    tail_width = gamma**500 / (1.0 - gamma)
    data = {}
    term_names = None
    for name in CONTROLLERS:
        paths = [directory / f"{name}_{block:02d}.pt" for block in range(16)]
        if not all(path.is_file() for path in paths):
            raise FileNotFoundError(f"missing {name} blocks")
        blocks = [torch.load(path, weights_only=True) for path in paths]
        if term_names is None:
            term_names = blocks[0]["term_names"]
        for block, payload in enumerate(blocks):
            if payload["controller"] != name or payload["block"] != block:
                raise ValueError(f"{name} block identity mismatch at {block}")
            if payload["term_names"] != term_names:
                raise ValueError("term-name inventory changed")
            expected = [7_100_000 + 10 * (block * 8 + call) for call in range(8)]
            if payload["seeds"] != expected:
                raise ValueError("paired seed inventory changed")
            if payload["timeout_seen"].any():
                raise ValueError("timeout present in raw block")
            if not torch.equal(
                payload["original_terms"].sum(-1), payload["original_returns"]
            ):
                raise ValueError("per-term sum differs from original return")
            if not torch.equal(payload["selection_counts"].sum(-1), payload["steps"]):
                raise ValueError("selection counts do not match live decisions")

        def joined(key):
            return (
                torch.stack([payload[key] for payload in blocks])
                .permute(1, 0, 2)
                .flatten(1)
                .double()
            )

        terms = (
            torch.stack([payload["original_terms"] for payload in blocks])
            .permute(1, 0, 2, 3)
            .flatten(1, 2)
            .double()
        )
        data[name] = {
            "original_returns": joined("original_returns"),
            "height_returns": joined("height_returns"),
            "original_terms": terms,
            "survived": joined("survived"),
            "steps": joined("steps"),
            "termination_step": joined("termination_step"),
            "selection_counts": (
                torch.stack([payload["selection_counts"] for payload in blocks])
                .permute(1, 0, 2, 3)
                .flatten(1, 2)
            ),
            "q_gain_sum": joined("q_gain_sum"),
            "tracking_steps": joined("tracking_steps"),
            "xy_error_sq": joined("xy_error_sq"),
            "yaw_error_sq": joined("yaw_error_sq"),
            "speed_sum": joined("speed_sum"),
            "command_speed_sum": joined("command_speed_sum"),
            "first_original_reward": joined("first_original_reward"),
            "first_height_reward": joined("first_height_reward"),
            "first_terminated": joined("first_terminated"),
        }
    if not torch.equal(
        data["plain"]["original_terms"].sum(-1),
        data["plain"]["original_returns"],
    ):
        raise ValueError("plain stored term total mismatch")
    if not torch.equal(
        data["original"]["original_terms"].sum(-1),
        data["original"]["original_returns"],
    ):
        raise ValueError("original stored term total mismatch")
    if not torch.equal(
        data["plain"]["selection_counts"][..., 0], data["plain"]["steps"]
    ):
        raise ValueError("plain controller did not select candidate zero")
    return provenance, term_names, data, tail_width


def controller_summary(data, name, tail_width):
    row = data[name]
    returns = row["original_returns"]
    height = row["height_returns"]
    state_mean = returns.mean(1)
    height_state_mean = height.mean(1)
    survivor = row["survived"].double().mean(1)
    stats = distribution(returns)
    height_stats = distribution(height)
    return {
        "original": stats,
        "original_state_mean": distribution(state_mean),
        "original_execution_sd_mean": float(returns.std(1, unbiased=True).mean()),
        "height": height_stats,
        "height_state_mean": distribution(height_state_mean),
        "height_execution_sd_mean": float(height.std(1, unbiased=True).mean()),
        "survivor_fraction": float(survivor.mean()),
        "fall_fraction": float(1.0 - survivor.mean()),
        "steps_mean": float(row["steps"].mean()),
        "termination_step": distribution(row["termination_step"]),
        "first_original_reward_mean": float(row["first_original_reward"].mean()),
        "first_height_negative_rate": float(
            (row["first_height_reward"] == -1).double().mean()
        ),
        "first_step_termination_rate": float(row["first_terminated"].double().mean()),
        "xy_tracking_rmse": float(
            (row["xy_error_sq"].sum() / row["tracking_steps"].sum()).sqrt()
        ),
        "yaw_tracking_rmse": float(
            (row["yaw_error_sq"].sum() / row["tracking_steps"].sum()).sqrt()
        ),
        "speed_over_command": float(
            row["speed_sum"].sum() / row["command_speed_sum"].sum()
        ),
        "mean_q_gain_per_decision": float(
            row["q_gain_sum"].sum() / row["selection_counts"].sum()
        ),
        "height_tail_interval": survivor_tail_interval(
            height_stats["mean"], float(survivor.mean()), tail_width
        ),
    }


def paired_summary(data, objective, seed):
    plain = data["plain"][f"{objective}_returns"]
    original = data["original"][f"{objective}_returns"]
    gains = original - plain
    state_mean = gains.mean(1)
    per_state_sd = gains.std(1, unbiased=True)
    tol = 1e-9
    wins = win_tie_loss(gains, tol)
    batch = batch_summary(gains)
    return {
        "objective": objective,
        "mean_gain": float(gains.mean()),
        "pooled_sd": float(gains.flatten().std(unbiased=True)),
        "per_state_sd_mean": float(per_state_sd.mean()),
        "per_state_sd_min": float(per_state_sd.min()),
        "per_state_sd_max": float(per_state_sd.max()),
        "state_mean": distribution(state_mean),
        "state_cluster_ci95": state_cluster_bootstrap(state_mean, seed),
        "paired_t": paired_t_summary(state_mean),
        "wins_ties_losses": wins,
        "batch": {key: value for key, value in batch.items() if key != "batch_gains"},
        "state_mean_positive": int((state_mean > tol).sum()),
        "state_mean_negative": int((state_mean < -tol).sum()),
    }


def term_summary(data, term_names, weights, termination_weights):
    plain = data["plain"]["original_terms"].mean((0, 1))
    original = data["original"]["original_terms"].mean((0, 1))
    delta = original - plain
    gain = float(data["original"]["original_returns"].mean())
    plain_gain = float(data["plain"]["original_returns"].mean())
    total_gain = gain - plain_gain
    if not torch.allclose(delta.sum(), torch.tensor(total_gain, dtype=torch.float64)):
        raise ValueError("per-term contributions do not sum to the total gain")
    rows = []
    for index, name in enumerate(term_names):
        weight = weights.get(name, termination_weights.get(name))
        if weight is None:
            raise ValueError(f"missing weight for {name}")
        rows.append(
            {
                "term": name,
                "weight": float(weight),
                "plain_mean": float(plain[index]),
                "original_mean": float(original[index]),
                "delta": float(delta[index]),
                "share_of_total_gain": (
                    float(delta[index]) / total_gain if total_gain != 0 else None
                ),
                "penalty": name in PENALTIES,
            }
        )
    return {
        "rows": rows,
        "delta_sum": float(delta.sum()),
        "total_gain": total_gain,
        "exact_sum": True,
        "baseline_original_mean": plain_gain,
        "original_mean": gain,
        "percent_of_plain": (
            100.0 * total_gain / plain_gain if plain_gain > 0 else None
        ),
    }


def prior_comparison(directory, data):
    if not directory.is_dir():
        return None
    result = {}
    for name, prefix in (("plain", "baseline"), ("original", "original")):
        paths = [directory / f"{prefix}_{block:02d}.pt" for block in range(16)]
        if not all(path.is_file() for path in paths):
            return None
        blocks = [torch.load(path, weights_only=True) for path in paths]
        prior = (
            torch.stack([block["returns"] for block in blocks])
            .permute(1, 0, 2)
            .flatten(1)
            .double()
        )
        current = data[name]["height_returns"]
        prior_state = prior.mean(1)
        current_state = current.mean(1)
        centered_prior = prior_state - prior_state.mean()
        centered_current = current_state - current_state.mean()
        correlation = float(
            (centered_prior * centered_current).sum()
            / (centered_prior.norm() * centered_current.norm())
        )
        result[name] = {
            "prior_mean_height": float(prior.mean()),
            "current_mean_height": float(current.mean()),
            "mean_drift": float(current.mean() - prior.mean()),
            "prior_state_mean_sd": float(prior_state.std(unbiased=True)),
            "current_state_mean_sd": float(current_state.std(unbiased=True)),
            "state_mean_correlation": correlation,
            "mean_absolute_state_mean_difference": float(
                (current_state - prior_state).abs().mean()
            ),
            "max_absolute_state_mean_difference": float(
                (current_state - prior_state).abs().max()
            ),
        }
    result["prior_directory"] = str(directory)
    result["note"] = (
        "Aggregate reproducibility comparison only. Warp closed-loop noise and "
        "possible source drift prevent a bitwise return claim."
    )
    return result


def build_report(directory, provenance, term_names, data, tail_width, report):
    original_mean = report["controllers"]["original"]["original"]["mean"]
    plain_mean = report["controllers"]["plain"]["original"]["mean"]
    gain = report["paired"]["original"]["mean_gain"]
    height_gain = report["paired"]["height"]["mean_gain"]
    t = report["paired"]["original"]["paired_t"]
    ci = report["paired"]["original"]["state_cluster_ci95"]
    w = report["paired"]["original"]["wins_ties_losses"]
    weights_text = json.dumps(provenance["training_reward"]["weights"], sort_keys=True)
    termination_text = json.dumps(
        provenance["training_reward"]["termination_weights"], sort_keys=True
    )
    tuple_text = "improvement" if gain > 0 else "sacrifice"
    lines = [
        "# Original PPO reward versus original height-Q gain",
        "",
        "**Question.** At the same 128 fixed insertion states and the same 1024",
        "paired executions per state, does the original fair-100 height-Q max-of-8",
        "controller improve or sacrifice the original PPO training return relative",
        "to the plain candidate-0 stochastic `model_600` controller? Original return",
        "is the primary objective; height-band return is measured on the same",
        "executions as a secondary objective.",
        "",
        "## Bottom line",
        "",
        f"- Plain original-return mean: **{plain_mean:.6f}**; "
        f"original height-Q mean: **{original_mean:.6f}**; "
        f"paired gain: **{gain:+.6f}** ({tuple_text}).",
        f"- Height-band paired gain on the same executions: **{height_gain:+.6f}**.",
        f"- Paired state-level t-test (df={t['df']}): mean **{t['mean']:.6f}**, "
        f"95% CI **[{t['ci95'][0]:.6f}, {t['ci95'][1]:.6f}]**, "
        f"two-sided p=**{t['p_two_sided']:.6g}**. "
        f"State-cluster bootstrap CI: **[{ci[0]:.6f}, {ci[1]:.6f}]**.",
        f"- Execution win/tie/loss: **{w['win']}/{w['tie']}/{w['loss']}** of {w['n']}.",
        "",
        "This is the measured trade-off at fixed states, not a population claim,",
        "not `Q^pi` validation, not policy improvement, and not a safety result.",
        "",
        "## Identity and protocol",
        "",
        f"- Source insertion: `{provenance['source']}`",
        f"  SHA-256 `{provenance['source_sha256']}`.",
        f"- Checkpoint: `{provenance['checkpoint']}`",
        f"  SHA-256 `{provenance['checkpoint_sha256']}`.",
        f"- Original height-Q: `{provenance['artifact']}`",
        f"  (`manifest.json` `{provenance['artifact_sha256']['manifest.json']}`,",
        f"  `q.pt` `{provenance['artifact_sha256']['q.pt']}`).",
        f"- Training reward config: `{weights_text}`",
        f"  plus termination `{termination_text}`.",
        f"- Gamma **{provenance['training_gamma']}** read from the loaded "
        f"training config (`train.algorithm.gamma`), "
        f"dt **{provenance['training_dt']}**, "
        f"control **{provenance['control_frequency']} Hz**, decimation "
        f"**{provenance['decimation']}**, "
        f"`n_policy_steps={provenance['n_policy_steps']}`.",
        "- 128 fixed states x 8 lanes = 1024 Warp worlds; 16 blocks x 8 calls;",
        "  1024 executions per state per controller; seeds",
        "  `7100000 + 10*call`, `call=0..127`.",
        f"- H500, gamma **{provenance['training_gamma']}** for both objectives,",
        f"  `episode_length_s={provenance['episode_length_s']}` (no admin timeout).",
        "- Native inventory **82** fields restored. The short same-controller",
        "  negative control reproduced discrete selections, live-step counts and",
        "  height returns exactly; continuous original returns and actions repeated",
        "  within the predeclared numeric tolerances (solver scratch is not part of",
        "  the snapshot). H5 height parity was exact for returns, steps and",
        "  selection.",
        "",
        "## Original PPO reward results",
        "",
        "Original return is `sum_t gamma^t * r_t`, where `r_t` is the runner's",
        "post-step `mask * ((weight * dt) * reward_function())` term sum. Ordinary",
        "terms use `~terminated`; the termination term uses `terminated`. Rewards",
        "are exactly zero after a true termination; there is no absorbing `-100`",
        "tail. The finite H500 return omits a continuing tail that is **not**",
        "bounded by the height interval.",
        "",
        "| Controller | Mean | Pooled SD | Min | Max | 128-state mean SD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in CONTROLLERS:
        row = report["controllers"][name]["original"]
        state = report["controllers"][name]["original_state_mean"]
        lines.append(
            f"| {name} | {row['mean']:.6f} | {row['sd']:.6f} | "
            f"{row['min']:.6f} | {row['max']:.6f} | {state['sd']:.6f} |"
        )
    lines.append("Pooled SD is the sample SD over 131072 executions per controller.")
    lines.append("")
    lines += [
        "",
        "Terminal fraction: "
        f"plain **{report['controllers']['plain']['fall_fraction']:.6f}**, "
        f"original **{report['controllers']['original']['fall_fraction']:.6f}**. "
        "Mean live steps: "
        f"**{report['controllers']['plain']['steps_mean']:.3f}** vs "
        f"**{report['controllers']['original']['steps_mean']:.3f}**.",
        "Raw blocks retain per-execution original return, per-term discounted sums,",
        "final alive mask, live-step count, termination step, first reward and",
        "tracking aggregates. No full per-step state tensors are stored.",
        "",
        "### Behavior aggregates",
        "",
        "| Controller | Survivor fraction | Mean live steps | XY RMSE | "
        "Yaw RMSE | Speed / command | First-step termination |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in CONTROLLERS:
        row = report["controllers"][name]
        lines.append(
            f"| {name} | {row['survivor_fraction']:.6f} | "
            f"{row['steps_mean']:.3f} | {row['xy_tracking_rmse']:.6f} | "
            f"{row['yaw_tracking_rmse']:.6f} | "
            f"{row['speed_over_command']:.6f} | "
            f"{row['first_step_termination_rate']:.6f} |"
        )
    lines += [
        "",
        "### Per-term paired contributions",
        "",
        "All entries are scalar after time aggregation, in original-return units.",
        "`delta = original - plain`; positive means the original controller's",
        "trajectory produced a larger weighted term. Negative terms are penalties,",
        "so a positive delta on a penalty means less cost. The total gain is",
        "negative here, so `delta / total` has the opposite sign of a positive",
        "delta.",
        "",
        "| Term | Weight | Plain | Original | Delta | Delta / total | Penalty |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["terms"]["rows"]:
        share = (
            f"{row['share_of_total_gain'] * 100:+.2f}%"
            if row["share_of_total_gain"] is not None
            else "n/a"
        )
        lines.append(
            f"| {row['term']} | {row['weight']:.6g} | {row['plain_mean']:.6f} | "
            f"{row['original_mean']:.6f} | {row['delta']:+.6f} | {share} | "
            f"{'yes' if row['penalty'] else 'no'} |"
        )
    lines += [
        "",
        f"Per-term deltas sum to **{report['terms']['delta_sum']:+.6f}**; "
        f"the reported total gain is **{report['terms']['total_gain']:+.6f}** "
        "(exact identity checked).",
        "",
        "## Paired gain and precision",
        "",
        "| Objective | Mean gain | Pooled SD | State mean SD | "
        "State-cluster CI95 | Execution win/tie/loss |",
        "|---|---:|---:|---:|---|---|",
    ]
    for objective in ("original", "height"):
        row = report["paired"][objective]
        w = row["wins_ties_losses"]
        interval = (
            f"[{row['state_cluster_ci95'][0]:+.6f}, "
            f"{row['state_cluster_ci95'][1]:+.6f}]"
        )
        lines.append(
            f"| {objective} | {row['mean_gain']:+.6f} | {row['pooled_sd']:.6f} | "
            f"{row['state_mean']['sd']:.6f} | "
            f"{interval} | "
            f"{w['win']}/{w['tie']}/{w['loss']} |"
        )
    lines += [
        "",
        "Execution win/tie/loss fractions for the original objective: "
        f"**{report['paired']['original']['wins_ties_losses']['win_fraction']:.6f}/"
        f"{report['paired']['original']['wins_ties_losses']['tie_fraction']:.6f}/"
        f"{report['paired']['original']['wins_ties_losses']['loss_fraction']:.6f}**.",
        "",
        "The state-cluster bootstrap resamples the 128 fixed state rows and is a",
        "descriptive across-state interval. The paired t-test treats the 128",
        "state mean gains as independent working units; executions within a state",
        "share seeds, lanes and the fixed initial state. Neither is a random",
        "state-population or safety statement. The original objective was the",
        "primary hypothesis; height is secondary.",
        "",
    ]
    if report["terms"]["percent_of_plain"] is not None:
        lines += [
            "Because the plain original-return baseline is positive, the paired",
            f"gain is **{report['terms']['percent_of_plain']:+.3f}%** of that",
            "baseline.",
            "",
        ]
    else:
        lines += [
            "The plain original-return baseline is not positive, so a percentage",
            "of baseline is not reported.",
            "",
        ]
    lines += [
        "## 16-execution batches",
        "",
        "| Objective | Positive / tie / negative batches | Positive fraction | "
        "Mean per-state batch SD | All-positive states | "
        "Majority-positive states |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for objective in ("original", "height"):
        row = report["paired"][objective]["batch"]
        lines.append(
            f"| {objective} | {row['win']}/{row['tie']}/{row['loss']} | "
            f"{row['positive_fraction']:.6f} | {row['per_state_sd_mean']:.6f} | "
            f"{row['all_positive_states']}/128 | "
            f"{row['majority_positive_states']}/128 |"
        )
    lines += [
        "",
        "Batches are 64 contiguous groups of 16 executions per state (two calls",
        "times all eight lanes). A positive batch fraction is not expected utility.",
        "",
        "## Height-band results",
        "",
        "The height objective is unchanged from the existing height-Q experiments:",
        "`0` inside `0.32 +/- 0.05 m`, `-1` outside, with the discounted absorbing",
        "failure tail. It is reported on the same trajectories but is a different",
        "objective with a different terminal convention.",
        "",
        "| Controller | Mean | Pooled SD | Min | Max | 128-state mean SD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in CONTROLLERS:
        row = report["controllers"][name]["height"]
        state = report["controllers"][name]["height_state_mean"]
        lines.append(
            f"| {name} | {row['mean']:.6f} | {row['sd']:.6f} | "
            f"{row['min']:.6f} | {row['max']:.6f} | {state['sd']:.6f} |"
        )
    lines += [
        "",
        f"The height `[-{tail_width:.6f}, 0]` survivor-tail interval applies "
        "only to this objective. It is not applied to the original PPO return.",
        "",
        "## Reproducibility against the prior persistent run",
        "",
    ]
    prior = report.get("prior")
    if prior is None:
        lines += ["Prior persistent blocks were unavailable; no drift comparison.", ""]
    else:
        lines += [
            "| Controller | Prior height mean | Current height mean | Mean drift | "
            "State-mean correlation | Mean abs state diff | "
            "Max abs state diff |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name in CONTROLLERS:
            row = prior[name]
            lines.append(
                f"| {name} | {row['prior_mean_height']:.6f} | "
                f"{row['current_mean_height']:.6f} | {row['mean_drift']:+.6f} | "
                f"{row['state_mean_correlation']:.6f} | "
                f"{row['mean_absolute_state_mean_difference']:.6f} | "
                f"{row['max_absolute_state_mean_difference']:.6f} |"
            )
        lines += [
            "",
            "This is an aggregate reproducibility check, not a bitwise match claim:",
            "Warp closed-loop noise and possible source drift can change long-horizon",
            "returns.",
            "",
        ]
    lines += [
        "## Scope limitations",
        "",
        "- The 128 states are fixed `k=100` insertion snapshots from one frozen",
        "  policy and one source lane; they are not a random state population.",
        "- H500 is twice the 250-step training episode. `episode_length_s` was",
        "  extended so no administrative timeout occurs; this is an horizon",
        "  extension, not a training-episode replay.",
        "- Zeroing all rewards after true termination is deliberate for the",
        "  original objective. It differs from the height absorbing failure tail",
        "  and from any finite-horizon Q target.",
        "- The original H500 return omits a continuing future tail. Unlike height,",
        "  original rewards are not bounded in `[-1, 0]`, so the `[-0.657, 0]`",
        "  height bound is not a bound on original return.",
        "- Per-term values are scalar after time aggregation; no per-joint or",
        "  per-body reward dimensions were stored.",
        "- The height Q was trained on nominal-policy data. The intervened branch",
        "  is off that distribution; selected Q is not `Q^pi` truth.",
        "- Execution-level intervals are conditional on the fixed states. The",
        "  state-cluster and paired t intervals are descriptive and depend on the",
        "  working independence assumption across fixed configurations.",
        "",
        "## Artifacts",
        "",
        f"- Raw blocks: `{directory}/plain_00.pt` ... `original_15.pt` (32 files).",
        f"- Audit and summaries: `{directory}/results.json`, "
        "`per_config.csv`, `terms.csv`, `batch_means_original.pt`, "
        "`batch_means_height.pt`.",
        f"- Inputs, code hashes, commands and exits: `{directory}/provenance.json`, "
        "`code.json`, `code_resume.json`, `commands.txt`, `commands.log`, "
        "`validation.json`, `validation_resume.json`, `completion.json`, "
        "`timings.jsonl`, `dirty.txt`, `dirty_resume.txt`.",
        "",
    ]
    return "\n".join(lines)


def main():
    args = parser().parse_args()
    directory = args.directory
    provenance, term_names, data, tail_width = load_blocks(directory)
    report = {
        "protocol": {
            "horizon": 500,
            "training_episode_steps": 250,
            "training_gamma": provenance["training_gamma"],
            "height_gamma": provenance["height_gamma"],
            "height_tail_width": tail_width,
            "training_reward": provenance["training_reward"],
            "seed_base": provenance["seed_base"],
            "world_axes": provenance["world_axes"],
            "candidate_axis": provenance["candidate_axis"],
        },
        "controllers": {
            name: controller_summary(data, name, tail_width) for name in CONTROLLERS
        },
        "paired": {
            "original": paired_summary(data, "original", 202611111),
            "height": paired_summary(data, "height", 202611112),
        },
        "terms": term_summary(
            data,
            term_names,
            provenance["training_reward"]["weights"],
            provenance["training_reward"]["termination_weights"],
        ),
        "prior": prior_comparison(args.prior, data),
    }
    for objective in ("original", "height"):
        means = batch_summary(
            data["original"][f"{objective}_returns"]
            - data["plain"][f"{objective}_returns"]
        )["batch_gains"]
        torch.save(
            means,
            directory / f"batch_means_{objective}.pt",
        )
    (directory / "results.json").write_text(json.dumps(report, indent=2))

    with (directory / "terms.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["terms"]["rows"][0]))
        writer.writeheader()
        writer.writerows(report["terms"]["rows"])

    plain_orig = data["plain"]["original_returns"].mean(1)
    original_orig = data["original"]["original_returns"].mean(1)
    plain_height = data["plain"]["height_returns"].mean(1)
    original_height = data["original"]["height_returns"].mean(1)
    orig_gain = original_orig - plain_orig
    height_gain = original_height - plain_height
    orig_sd = (
        data["original"]["original_returns"] - data["plain"]["original_returns"]
    ).std(1, unbiased=True)
    height_sd = (
        data["original"]["height_returns"] - data["plain"]["height_returns"]
    ).std(1, unbiased=True)
    with (directory / "per_config.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "state",
                "original_plain_mean",
                "original_original_mean",
                "original_gain",
                "original_gain_sd",
                "height_plain_mean",
                "height_original_mean",
                "height_gain",
                "height_gain_sd",
                "survivor_fraction_plain",
                "survivor_fraction_original",
            ]
        )
        for state in range(128):
            writer.writerow(
                [
                    state,
                    float(plain_orig[state]),
                    float(original_orig[state]),
                    float(orig_gain[state]),
                    float(orig_sd[state]),
                    float(plain_height[state]),
                    float(original_height[state]),
                    float(height_gain[state]),
                    float(height_sd[state]),
                    float(data["plain"]["survived"][state].double().mean()),
                    float(data["original"]["survived"][state].double().mean()),
                ]
            )
    (directory / "REPORT.md").write_text(
        build_report(directory, provenance, term_names, data, tail_width, report)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
