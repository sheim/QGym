"""Simulation-free per-state analysis of the contact-force intervention.

Answers: which initial conditions violate the contact constraint under the plain
frozen policy, and does a controller remove those violations entirely? Uses the
per-state, per-lane block files written by
``scripts/evaluate_contact_q_persistent.py`` (128 states x 8 lanes x blocks), so
nothing is re-simulated.

Usage:

    uv run --frozen python scripts/summarize_contact_q_persistent.py \
      --source logs/go2trot/<run>/contact_persistent \
      --artifact logs/go2trot/<run>/offline_q_60k/contact_q/data/100pct \
      --output logs/go2trot/<run>/contact_persistent_analysis
"""

import argparse
import json
from pathlib import Path

import torch

from learning.offline.q_artifact import ContactQ
from learning.offline.rewards import observation_field_slice

EMPTY = torch.zeros(0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--primary",
        default=None,
        help="Controller suffix compared against plain (default: maxq).",
    )
    parser.add_argument(
        "--rescues",
        nargs="+",
        type=int,
        default=[1, 2, 3, 5, 10],
        help="Plain-execution violation counts used for the rescue table.",
    )
    return parser


def load_blocks(source: Path) -> tuple[dict, dict]:
    files = sorted(source.glob("block_*.pt"))
    if not files:
        raise FileNotFoundError(f"no block_*.pt in {source}")
    blocks = [torch.load(path, weights_only=True) for path in files]
    controllers = list(blocks[0]["controllers"])
    stacked = {}
    for key in controllers:
        stacked[key] = {
            field: torch.cat(
                [block["controllers"][key][field].double() for block in blocks], dim=1
            )
            for field in ("returns", "violation_steps", "steps", "survived")
        }
    return stacked, {"blocks": len(files), "seeds": [b["seed"] for b in blocks]}


def describe_state(observations, manifest, index):
    """Command and phase of one insertion state from its recorded observation."""
    names = list(manifest["obs_names"])
    dims = [int(dim) for dim in manifest["obs_dims"]]
    contract = {"obs_names": names, "obs_dims": dims}
    command = observations[index][observation_field_slice(contract, "commands")]
    phase = observations[index][observation_field_slice(contract, "phase_obs")]
    return {
        "vx": float(command[0]),
        "vy": float(command[1]),
        "yaw": float(command[2]),
        "phase_sin": float(phase[0]),
        "phase_cos": float(phase[1]),
    }


def main():
    args = build_parser().parse_args()
    source = Path(args.source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    primary = args.primary or "maxq"

    stacked, block_info = load_blocks(source)
    artifact = ContactQ(args.artifact)
    provenance = json.loads((source / "provenance.json").read_text())
    state_names = provenance["artifact_names"][0]
    plain_key = f"{state_names}_plain"
    primary_key = f"{state_names}_{primary}"
    if plain_key not in stacked or primary_key not in stacked:
        raise ValueError(f"missing {plain_key!r} or {primary_key!r}")

    violations = {key: value["violation_steps"] for key, value in stacked.items()}
    steps = stacked[plain_key]["steps"]
    states, executions = violations[plain_key].shape
    plain_viol = violations[plain_key]
    plain_per_state = plain_viol.mean(1)

    insertion = torch.load(
        source.parent / "contact_ranking" / "insertion.pt", weights_only=True
    )
    observations = insertion["insertion_observations"].reshape(
        states, -1, insertion["insertion_observations"].shape[-1]
    )[:, 0]
    policy_manifest = artifact.manifest["policy_spec"]

    summary = {
        "source": str(source),
        "blocks": block_info,
        "states": states,
        "executions_per_state": executions,
        "horizon_steps_mean": float(steps.mean()),
        "primary": primary,
        "controllers": {},
    }
    for key, value in sorted(stacked.items()):
        per_state = violations[key].mean(1)
        zero_executions = float((violations[key] == 0).double().mean())
        zero_states = float((per_state == 0).double().mean())
        entry = {
            "violations_per_execution_mean": float(per_state.mean()),
            "violations_per_state_p50": float(per_state.median()),
            "violations_per_state_p90": float(per_state.quantile(0.9)),
            "violations_per_state_max": float(per_state.max()),
            "executions_with_zero_violations_fraction": zero_executions,
            "states_with_zero_violations_fraction": zero_states,
            "states": int(per_state.numel()),
        }
        if key != plain_key:
            delta = per_state - plain_per_state
            entry["vs_plain"] = {
                "states_fewer": int((delta < 0).sum()),
                "states_equal": int((delta == 0).sum()),
                "states_more": int((delta > 0).sum()),
                "mean_delta": float(delta.mean()),
                "max_reduction": float(-delta.min()),
            }
        summary["controllers"][key] = entry

    # Rescue table: states with many plain violations that the primary zeroes.
    rescued = {}
    for threshold in args.rescues:
        mask = plain_per_state >= threshold
        zeroed = mask & (violations[primary_key].mean(1) == 0)
        rescued[str(threshold)] = {
            "states_with_plain_at_least": int(mask.sum()),
            "of_which_primary_zero": int(zeroed.sum()),
            "plain_violations_mean_over_those": float(plain_per_state[mask].mean())
            if bool(mask.any())
            else 0.0,
        }
    summary["rescues"] = rescued

    # Worst initial conditions under the plain policy.
    order = torch.argsort(plain_per_state, descending=True)
    worst = []
    for index in order[:10].tolist():
        entry = {
            "state": index,
            "plain_violations_per_execution": float(plain_per_state[index]),
            "plain_zero_execution_fraction": float(
                (violations[plain_key][index] == 0).double().mean()
            ),
            **describe_state(observations, policy_manifest, index),
        }
        for key in sorted(stacked):
            if key == plain_key:
                continue
            entry[key] = float(violations[key][index].mean())
            entry[f"{key}_zero_fraction"] = float(
                (violations[key][index] == 0).double().mean()
            )
        worst.append(entry)
    summary["worst_plain_states"] = worst

    # Where does the primary help most? Correlation of plain violations with
    # the reduction, and the states that get worse.
    primary_delta = violations[primary_key].mean(1) - plain_per_state
    correlation = 0.0
    if float(plain_per_state.std()) > 0 and float(primary_delta.std()) > 0:
        x = plain_per_state - plain_per_state.mean()
        y = primary_delta - primary_delta.mean()
        correlation = float(
            (x * y).sum() / (x.square().sum() * y.square().sum()).sqrt()
        )
    # Paired single executions: same state and the same continuation seed, plain
    # violates k times and the primary violates zero times. This is the "is there
    # an initial condition where the intervention cleans it up" question at the
    # granularity of one rollout rather than a 32-execution mean.
    plain_exec = violations[plain_key]
    primary_exec = violations[primary_key]
    paired = {}
    for threshold in args.rescues:
        mask = (plain_exec >= threshold) & (primary_exec == 0)
        paired[str(threshold)] = {
            "executions": int(mask.sum()),
            "fraction": float(mask.double().mean()),
            "max_plain_violations": float(plain_exec[mask].max())
            if bool(mask.any())
            else 0.0,
        }
    summary["paired_executions"] = paired

    summary["primary_vs_plain"] = {
        "correlation_plain_violations_with_delta": correlation,
        "states_worse": int((primary_delta > 0).sum()),
        "worst_regression_state": int(primary_delta.argmax()),
        "worst_regression_violations": float(primary_delta.max()),
    }

    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = ["# Contact-force intervention by initial condition", ""]
    lines += [
        f"{states} states x {executions} executions, horizon "
        f"{summary['horizon_steps_mean']:.0f} steps, blocks {block_info['blocks']}.",
        "",
        "| controller | violations/execution | p90 | max "
        "| executions with 0 | states with 0 |",
        "|---|---|---|---|---|---|",
    ]
    for key, entry in summary["controllers"].items():
        lines.append(
            f"| {key} | {entry['violations_per_execution_mean']:.2f} | "
            f"{entry['violations_per_state_p90']:.2f} | "
            f"{entry['violations_per_state_max']:.2f} | "
            f"{100 * entry['executions_with_zero_violations_fraction']:.1f}% | "
            f"{100 * entry['states_with_zero_violations_fraction']:.1f}% |"
        )
    lines += [
        "",
        "## Rescue: states with many plain violations, primary zeroes them",
        "",
    ]
    lines += [
        "| plain violations/execution >= | states | of which primary zero "
        "| mean plain violations |",
        "|---|---|---|---|",
    ]
    for threshold, entry in summary["rescues"].items():
        lines.append(
            f"| {threshold} | {entry['states_with_plain_at_least']} | "
            f"{entry['of_which_primary_zero']} | "
            f"{entry['plain_violations_mean_over_those']:.2f} |"
        )
    lines += ["", "## Worst states under the plain policy", ""]
    lines += [
        "| state | vx | vy | yaw | plain viol | plain zero "
        "| primary viol | primary zero |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for entry in worst:
        lines.append(
            f"| {entry['state']} | {entry['vx']:+.2f} | {entry['vy']:+.2f} | "
            f"{entry['yaw']:+.2f} | {entry['plain_violations_per_execution']:.2f} | "
            f"{100 * entry['plain_zero_execution_fraction']:.0f}% | "
            f"{entry[primary_key]:.2f} | "
            f"{100 * entry[f'{primary_key}_zero_fraction']:.0f}% |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print((output / "summary.md").read_text())
    print(f"summary={output / 'summary.json'}")


if __name__ == "__main__":
    main()
