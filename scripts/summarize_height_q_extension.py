"""Summarize matched 16/64 estimates with whole-64-batch cluster bootstrap."""

import argparse
import csv
import itertools
import json
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

import torch

from learning.offline.fixed_state_variance import summarize_returns
from learning.offline.contracts import sha256_file


def quantities(means):
    variance = means.var(0)
    spread = means.var(-1)
    contrast = (means[..., 1:] - means[..., :1]).var(0)
    values = {
        "fixed_action_variance": variance,
        "fixed_action_sd": variance.sqrt(),
        "within_batch_action_variance": spread,
        "within_batch_action_sd": spread.sqrt(),
        "within_batch_action_range": means.amax(-1) - means.amin(-1),
        "paired_estimator_variance": contrast,
    }
    result = {
        k: {"mean": float(v.mean()), "median": float(v.median())}
        for k, v in values.items()
    }
    pairs = list(itertools.combinations(range(8), 2))
    signs = torch.stack(
        [torch.sign(means[..., a] - means[..., b]) for a, b in pairs], -1
    )
    n = means.shape[0]
    pos, neg, zero = [(signs == v).sum(0).double() for v in (1, -1, 0)]
    same = pos * (pos - 1) / 2 + neg * (neg - 1) / 2
    tied = zero * (n - zero) + zero * (zero - 1) / 2
    result["rank_agreement"] = float(((same + tied * 0.5) / (n * (n - 1) / 2)).mean())
    winners = torch.nn.functional.one_hot(means.argmax(-1), 8).sum(0).double()
    result["winner_agreement"] = float(
        (winners * (winners - 1)).sum(-1).mean() / (n * (n - 1))
    )
    result["pair_tie_fraction"] = float((signs == 0).double().mean())
    return result


def flat_report(report):
    return {
        f"{k}.{sub}": value
        for k, values in report.items()
        if isinstance(values, dict)
        for sub, value in values.items()
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    args = p.parse_args()
    directory = args.directory
    raw = torch.load(directory / "raw_returns.pt", weights_only=True)
    insertion = torch.load(directory / "insertion.pt", weights_only=True)
    provenance = json.loads((directory / "provenance.json").read_text())
    x = raw["returns"].double()
    assert x.shape == (16, 64, 16, 8)
    assert torch.isfinite(x).all() and (x >= -100.000001).all() and (x <= 0).all()
    assert ((raw["first_rewards"] == 0) | (raw["first_rewards"] == -1)).all()
    source = torch.load(
        Path(provenance["source"]) / "raw_returns.pt", weights_only=True
    )
    assert torch.equal(
        x[:2], source["returns"].reshape(2, 64, 128, 8)[:, :, raw["state_indices"]]
    )
    identities = [
        tuple(identity) for block in raw["new_seed_identities"] for identity in block
    ]
    assert len(identities) == len(set(identities)) == 896
    original = torch.load(
        Path(provenance["source"]) / "insertion.pt", weights_only=True
    )
    ids = raw["state_indices"]
    assert torch.equal(insertion["candidates"], original["candidates"][ids])
    assert torch.equal(insertion["predictions"], original["predictions"][:, ids])
    assert torch.equal(
        insertion["insertion_observations"],
        original["insertion_observations"].reshape(128, 8, -1)[ids],
    )
    dof_pos = next(
        e["value"]
        for e in original["snapshot"]["entries"]
        if e["owner"] == "Go2Trot" and e["field"] == "dof_pos"
    ).reshape(128, 8, 12)[ids, 0]
    joints = {
        j.get("name"): j.find("limit")
        for j in ET.parse("resources/robots/go2/urdf/go2.urdf")
        .getroot()
        .findall("joint")
    }
    names = original["env_config"]["asset"]["robot_layout"]["dof_names"]
    lower = torch.tensor([float(joints[n].get("lower")) for n in names])
    upper = torch.tensor([float(joints[n].get("upper")) for n in names])
    excess = torch.maximum(lower - dof_pos, dof_pos - upper).clamp_min(0)
    audit = {
        "selected_out_of_limit_positions": int((excess > 0).sum()),
        "selected_states_with_excess": int((excess > 0).any(-1).sum()),
        "max_excess_rad": float(excess.max()),
        "all_insertion_observations_candidates_predictions_exact": True,
    }
    (directory / "insertion_audit.json").write_text(json.dumps(audit, indent=2))
    stats64 = summarize_returns(x)
    stats16 = summarize_returns(x.reshape(64, 16, 16, 8))
    assert torch.allclose(
        stats64["batch_means"].mean(0), stats16["batch_means"].mean(0)
    )
    cov = stats64["batch_candidate_covariance"]
    assert torch.allclose(
        stats64["paired_batch_estimator_variance"],
        cov.diagonal(dim1=-2, dim2=-1) + cov[:, :1, 0] - 2 * cov[:, :, 0],
    )
    report = {
        "at64": quantities(stats64["batch_means"]),
        "at16_same_dataset": quantities(stats16["batch_means"]),
        "original_subset_at16": quantities(
            source["returns"][:, :, raw["state_indices"]].double().mean(1)
        ),
        "new_only_at64_sensitivity": quantities(stats64["batch_means"][2:]),
    }
    for label, stats in (("at64", stats64), ("at16_same_dataset", stats16)):
        assert (
            abs(
                report[label]["rank_agreement"]
                - float(stats["batch_ranking_agreement_by_state_pair"].mean())
            )
            < 1e-12
        )
    # Resample entire 64-return batches, preserving every state/action and the
    # four nested 16-return subgroups. No pseudo-replication over actions/states.
    gen = torch.Generator().manual_seed(2026111065)
    submeans = x.reshape(16, 4, 16, 16, 8).mean(2)
    draws = {"at64": [], "at16_same_dataset": [], "variance_ratio64_to16": []}
    for _ in range(4000):
        ids = torch.randint(16, (16,), generator=gen)
        a = quantities(stats64["batch_means"][ids])
        b = quantities(submeans[ids].reshape(64, 16, 8))
        draws["at64"].append(flat_report(a))
        draws["at16_same_dataset"].append(flat_report(b))
        draws["variance_ratio64_to16"].append(
            a["fixed_action_variance"]["mean"] / b["fixed_action_variance"]["mean"]
        )
    ci = {}
    for label in ("at64", "at16_same_dataset"):
        ci[label] = {
            k: torch.tensor([v[k] for v in draws[label]])
            .quantile(torch.tensor([0.025, 0.975]))
            .tolist()
            for k in draws[label][0]
        }
    ci["variance_ratio64_to16"] = (
        torch.tensor(draws["variance_ratio64_to16"])
        .quantile(torch.tensor([0.025, 0.975]))
        .tolist()
    )
    report["bootstrap_95_percentile"] = ci
    report["bootstrap_method"] = (
        "4000 whole-64-batch resamples, seed 2026111065; fixed states/candidates; "
        "paired regrouping; approximate percentile intervals with only 16 "
        "clusters, not exact variance coverage or state-population uncertainty"
    )
    report["variance_ratio64_to16"] = (
        report["at64"]["fixed_action_variance"]["mean"]
        / report["at16_same_dataset"]["fixed_action_variance"]["mean"]
    )
    paired = x[..., 1:] - x[..., :1]
    report["pooled"] = {
        k: {"mean": float(v.mean()), "median": float(v.median())}
        for k, v in {
            "paired_mean": paired.mean((0, 1)),
            "absolute_paired_mean": paired.mean((0, 1)).abs(),
            "individual_paired_variance": paired.flatten(0, 1).var(0),
        }.items()
    }
    tail = raw["survived"].double() * (0.99**500 / 0.01)
    report.update(
        mean_return=float(x.mean()),
        first_reward_negative_rate=float((raw["first_rewards"] == -1).double().mean()),
        terminated_fraction=float((~raw["survived"]).double().mean()),
        tail_only_mean_interval=[float((x - tail).mean()), float(x.mean())],
        survivor_tail_width=0.99**500 / 0.01,
        paired_tail_interval_width_mean=float((tail[..., 1:] + tail[..., :1]).mean()),
        new_seconds=sum(raw["new_block_seconds"]),
        new_block_seconds=raw["new_block_seconds"],
        state_indices=raw["state_indices"].tolist(),
    )
    target = x.mean((0, 1))
    report["critics"] = []
    for prediction in insertion["predictions"]:
        signs = torch.stack(
            [
                torch.sign(prediction[:, a] - prediction[:, b])
                * torch.sign(target[:, a] - target[:, b])
                for a, b in itertools.combinations(range(8), 2)
            ]
        )
        report["critics"].append(
            {
                "mae_to_pooled_finite_estimate": float(
                    (prediction - target).abs().mean()
                ),
                "bias": float((prediction - target).mean()),
                "pair_accuracy_to_noisy_estimate": float(((signs + 1) / 2).mean()),
                "prediction_range": float(
                    (prediction.amax(-1) - prediction.amin(-1)).mean()
                ),
            }
        )
    torch.save(
        {
            "at64": stats64,
            "at16": stats16,
            "tail_lower": (x - tail).mean((0, 1)),
            "tail_upper": target,
            "paired_tail_lower": ((x - tail)[..., 1:] - x[..., :1]).mean((0, 1)),
            "paired_tail_upper": (x[..., 1:] - (x - tail)[..., :1]).mean((0, 1)),
        },
        directory / "statistics.pt",
    )
    with (directory / "per_state_action.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "source_state",
                "candidate",
                "pooled_return",
                "var64",
                "var16",
                "paired_var64",
            ]
        )
        for s, source_id in enumerate(raw["state_indices"].tolist()):
            for c in range(8):
                writer.writerow(
                    [
                        source_id,
                        c,
                        float(target[s, c]),
                        float(stats64["batch_estimator_variance"][s, c]),
                        float(stats16["batch_estimator_variance"][s, c]),
                        float(stats64["paired_batch_estimator_variance"][s, c]),
                    ]
                )
    (directory / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)
    )
    code_paths = [
        "learning/offline/fixed_state_variance.py",
        "scripts/evaluate_height_q_variance.py",
        "scripts/extend_height_q_variance.py",
        "scripts/summarize_height_q_extension.py",
        "tests/unit_tests/test_fixed_state_variance.py",
    ]
    code_provenance = {
        "sha256": {p: sha256_file(p) for p in code_paths},
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "torch_version": str(torch.__version__),
    }
    (directory / "code_provenance.json").write_text(
        json.dumps(code_provenance, indent=2)
    )
    lines = [
        "# Sixteen independent 64-return batches — 2026-11-10",
        "",
        "## Protocol",
        "",
        f"Random selection seed {provenance['selection_seed']}; original state indices "
        f"`{report['state_indices']}`. Selection precedes reading returns.",
        "Exact original physical snapshots, noisy insertion observations, eight "
        "candidate actions and both critics' predictions are retained; "
        "no prefix rerun.",
        "Two batches reuse all 128 original continuations; fourteen new batches "
        "add 896 continuations per state/action: 114688 new, 131072 retained rollouts.",
        "1024 worlds = lane8 × state16 × candidate8. Each new 64-return batch "
        "contains eight calls × eight lanes. RNG identity is (call seed,lane): "
        "distinct vectorized random subsequences across lanes, not eight separately "
        "seeded generators. Candidate noise is shared within each state/lane. "
        "New call seeds 6400000..6401110 stride10 do not collide with old streams.",
        "Frozen policy, H500, gamma .99, height reward 0 inside .32±.05m and -1 "
        "outside; observed terminal reward plus exact absorbing failure tail.",
        "",
        "## Matched-data results",
        "",
        "All rows summarize per-state/action quantities, not pooled-state variance. "
        "At64 uses 16 batches (15 df); at16 regroups the SAME returns into 64 "
        "batches (63 df). Medians use torch's lower-middle convention. Paired "
        "contrasts exclude nominal-minus-itself; covariance is retained.",
        "",
        "| Quantity | At16 mean | At16 median | At64 mean | At64 median |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in (
        "fixed_action_variance",
        "fixed_action_sd",
        "within_batch_action_variance",
        "within_batch_action_sd",
        "within_batch_action_range",
        "paired_estimator_variance",
    ):
        a, b = report["at16_same_dataset"][key], report["at64"][key]
        lines.append(
            f"| {key} | {a['mean']:.6f} | {a['median']:.6f} | "
            f"{b['mean']:.6f} | {b['median']:.6f} |"
        )
    for key in ("rank_agreement", "winner_agreement"):
        lines.append(
            f"\n{key}: at16 **{report['at16_same_dataset'][key]:.6%}**, "
            f"at64 **{report['at64'][key]:.6%}**."
        )
    lines += [
        "Pair ties count half; winner ties select the first candidate. Repeatability "
        "is not accuracy against true Q. Action spread contains both signal and noise.",
        "",
        "## Uncertainty",
        "",
        report["bootstrap_method"]
        + ". Actions/states stay together in every resample. "
        "Ordinary percentile bootstrap variance intervals can be biased with only "
        "16 batches and skewed returns; they are conditional, approximate intervals.",
    ]
    for key in (
        "fixed_action_variance.mean",
        "fixed_action_variance.median",
        "within_batch_action_variance.mean",
        "paired_estimator_variance.mean",
    ):
        lo, hi = ci["at64"][key]
        lines.append(f"- At64 {key}: 95% bootstrap interval **[{lo:.6f}, {hi:.6f}]**.")
    lines += [
        f"- Mean variance ratio64/16: **{report['variance_ratio64_to16']:.6f}**, "
        f"95% paired cluster-bootstrap interval {ci['variance_ratio64_to16']}; "
        "1/4 is the IID scaling reference, not a fitted noise ceiling.",
        "",
        "The old 128-state summary and this random 16-state subset have different "
        "coverage; do not directly interpret their differences as "
        "rollout-count effects. "
        "`original_subset_at16` and `new_only_at64_sensitivity` are retained in JSON. "
        "More batches reduce variance-estimation uncertainty, not hidden-state "
        "critic bias, finite-horizon truncation or state-population uncertainty.",
        "",
        "## Pooled diagnostics and critics",
        "",
        f"First reward -1 rate **{report['first_reward_negative_rate']:.6%}**, "
        f"termination **{report['terminated_fraction']:.6%}**. Rates use individual "
        "binary rewards, never thresholded averages.",
        f"Mean finite/absorbing return **{report['mean_return']:.6f}**. Mean tail-only "
        f"interval **{report['tail_only_mean_interval']}**; surviving omitted tail "
        f"[-{report['survivor_tail_width']:.7f},0]. Paired tail interval mean width "
        f"{report['paired_tail_interval_width_mean']:.6f}. "
        "These exclude sampling uncertainty.",
        "",
        "| Critic | MAE to pooled finite estimate | Bias | "
        "Pair accuracy to noisy estimate | Predicted action range |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, c in zip(("Original fair100", "Eval-distribution100"), report["critics"]):
        lines.append(
            f"| {name} | {c['mae_to_pooled_finite_estimate']:.6f} | "
            f"{c['bias']:.6f} | {c['pair_accuracy_to_noisy_estimate']:.6f} | "
            f"{c['prediction_range']:.6f} |"
        )
    lines += [
        "",
        "## Fidelity, timing and artifacts",
        "",
        "All 82 native integration/model/task/controller/DR entries use the same "
        "index mapping; every continuation asserts exact restoration. Native "
        "one-step source/replica and lane restore gates, plus command/push candidate "
        "pairing and lane distinction pass (`validation.json`, "
        "`validation_resume.json`). This proves numerical one-step fidelity, "
        "not bitwise long-horizon equivalence. "
        "World reordering can amplify numerical solver differences in closed loop; "
        "reused versus new batches also differ in native world placement.",
        f"Selected-state joint-limit audit: {audit}. Native positions are preserved "
        "without reset clamping. Adapter remains restricted to this 1024-world "
        "nonsleeping Go2Trot/Warp configuration.",
        f"New collection **{report['new_seconds']:.6f}s** "
        f"({report['new_seconds'] / 60:.3f} min), "
        f"first block **{report['new_block_seconds'][0]:.6f}s**, retained. "
        "Original full collection was 459.0989s. Timings exclude environment setup, "
        "native validation and offline analysis; no extra critic rollout is used.",
        "",
        "- `raw_returns.pt`: explicit [batch64,repeat64,state16,candidate8] axes, "
        "returns/first rewards/survival, old seeds and new (seed,lane) identities.",
        "- `new_block_00.pt`..`new_block_13.pt`: retained new blocks and timings.",
        "- `insertion.pt`, `insertion_audit.json`: exact selected snapshot, "
        "observations, candidates and both original predictions.",
        "- `statistics.pt`, `per_state_action.csv`, `results.json`: full statistics, "
        "tail bounds, matched grouping, bootstrap intervals and sensitivity summaries.",
        "- `provenance.json`, `code_provenance.json`, `commands.txt`: selection, "
        "stream identities, source hashes, code hashes/revision/dirty status and CLI.",
        "",
        "Reproduce analysis from BigRedGym: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        f"uv run --frozen python scripts/summarize_height_q_extension.py {directory}`.",
        "",
        "**Top-Q sampled-action policy selection remains explicitly deferred.** "
        "No noise-ceiling claim, commits, todos, plan completion or archival.",
    ]
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
