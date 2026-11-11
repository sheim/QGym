"""Train a height-reward Q by fixed-policy evaluation on replay snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path

from learning.offline.contracts import HeightRewardSpec, TrainingSpec
from learning.offline.dataset import (
    TransitionDataset,
    concat_datasets,
    load_snapshot,
    nested_subsets,
    sample_n,
)
from learning.offline.rewards import relabel_height
from learning.offline.training import fraction_label, load_policy, train_height_q


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", nargs="+", default=None)
    parser.add_argument("--early_snapshot", default=None)
    parser.add_argument("--late_snapshot", default=None)
    parser.add_argument("--balance_per_snapshot", type=int, default=None)
    parser.add_argument("--dataset_label", default="data")
    parser.add_argument("--label", default="height_q")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--height_reference", type=float, required=True)
    parser.add_argument("--height_bound", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument(
        "--subset_group",
        choices=("transition", "episode"),
        required=True,
    )
    parser.add_argument("--subset_seed", type=int, default=0)
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.1, 0.5, 1.0],
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--updates", type=int, default=100_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[256, 256])
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--target_tau", type=float, default=0.005)
    parser.add_argument("--target_action_samples", type=int, default=1)
    parser.add_argument(
        "--no_normalize_observations",
        action="store_true",
        default=False,
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source_revision", default=None)
    return parser


def _validate_manifest_compatibility(datasets: list[TransitionDataset]) -> None:
    first = datasets[0].manifest
    keys = (
        "obs_names",
        "obs_dims",
        "action_names",
        "action_dim",
        "actor_cfg",
        "noise",
        "scales",
        "control_dt",
        "n_policy_steps",
        "episode_length_s",
        "config_fingerprint",
    )
    for dataset in datasets[1:]:
        for key in keys:
            if dataset.manifest[key] != first[key]:
                raise ValueError(
                    f"snapshot manifests disagree on {key}; cannot combine"
                )
    revisions = {
        dataset.manifest.get("source_revision")
        for dataset in datasets
        if dataset.manifest.get("source_revision") not in (None, "unknown")
    }
    if len(revisions) > 1:
        raise ValueError("snapshot manifests come from different source revisions")


def _single_dataset(args) -> dict[str, TransitionDataset]:
    datasets = [load_snapshot(path) for path in args.snapshot]
    _validate_manifest_compatibility(datasets)
    combined = concat_datasets(datasets)
    return {args.dataset_label: combined}


def _early_late_datasets(args) -> dict[str, TransitionDataset]:
    early = load_snapshot(args.early_snapshot)
    late = load_snapshot(args.late_snapshot)
    _validate_manifest_compatibility([early, late])
    size = min(len(early), len(late))
    if args.balance_per_snapshot is not None:
        if args.balance_per_snapshot <= 0:
            raise ValueError("balance_per_snapshot must be positive")
        if args.balance_per_snapshot > size:
            raise ValueError("balance_per_snapshot exceeds the smaller snapshot")
        size = args.balance_per_snapshot
    early_subset = sample_n(
        early,
        size,
        seed=args.subset_seed,
        group_by="transition",
    )
    late_subset = sample_n(
        late,
        size,
        seed=args.subset_seed,
        group_by="transition",
    )
    mixed = concat_datasets([early_subset, late_subset])
    print(f"early={len(early_subset)} late={len(late_subset)} mixed={len(mixed)}")
    return {"early": early_subset, "late": late_subset, "mixed": mixed}


def main() -> None:
    args = build_parser().parse_args()
    if (args.early_snapshot is None) != (args.late_snapshot is None):
        raise SystemExit("--early_snapshot and --late_snapshot must be used together")
    if args.early_snapshot is not None and args.snapshot is not None:
        raise SystemExit("use either --snapshot or early/late snapshots")
    if args.early_snapshot is None and args.snapshot is None:
        raise SystemExit("provide --snapshot or --early_snapshot/--late_snapshot")

    reward_spec = HeightRewardSpec(
        reference_height_m=args.height_reference,
        bound_m=args.height_bound,
    )
    training_spec = TrainingSpec(
        gamma=args.gamma,
        batch_size=args.batch_size,
        updates=args.updates,
        learning_rate=args.learning_rate,
        hidden_dims=tuple(args.hidden_dims),
        activation=args.activation,
        target_tau=args.target_tau,
        seed=args.seed,
        device=args.device,
        target_action_samples=args.target_action_samples,
        normalize_observations=not args.no_normalize_observations,
    )
    fractions = tuple(args.fractions)
    datasets = (
        _early_late_datasets(args)
        if args.early_snapshot is not None
        else _single_dataset(args)
    )
    first_manifest = next(iter(datasets.values())).manifest
    source_revision = args.source_revision or first_manifest.get(
        "source_revision", "unknown"
    )

    for dataset_label, dataset in datasets.items():
        policy, policy_spec = load_policy(
            dataset.manifest,
            args.checkpoint,
            args.device,
        )
        relabeled = relabel_height(dataset, reward_spec)
        subsets = nested_subsets(
            relabeled,
            fractions,
            seed=args.subset_seed,
            group_by=args.subset_group,
        )
        for fraction, subset in zip(fractions, subsets, strict=True):
            output_dir = (
                Path(args.output)
                / args.label
                / dataset_label
                / fraction_label(fraction)
            )
            train_height_q(
                subset,
                policy,
                policy_spec,
                spec=training_spec,
                output_dir=output_dir,
                subset={
                    "fraction": fraction,
                    "group_by": args.subset_group,
                    "seed": args.subset_seed,
                    "dataset_label": dataset_label,
                },
                source_revision=source_revision,
            )


if __name__ == "__main__":
    main()
