"""Train a foot contact-force Q by fixed-policy evaluation on replay snapshots.

The reward is the per-step ``-1/0`` indicator defined by
``ContactForceRewardSpec``: ``-1`` when any foot's post-action vertical load
exceeds that foot's threshold, otherwise ``0``. Thresholds are the per-foot
percentile of the recorded per-touchdown peak distribution unless explicitly
supplied, and they are stored in the artifact.

Usage:

    uv run --frozen python scripts/train_contact_q.py \
      --snapshot logs/go2trot/<run>/replay/replay_iter_0000616 \
      --checkpoint logs/go2trot/<run>/model_616.pt \
      --foot_names FL_foot FR_foot RL_foot RR_foot \
      --gamma 0.99 --subset_group episode --output logs/go2trot/<run>/offline_q
"""

from __future__ import annotations

import argparse
from pathlib import Path

from learning.offline.contracts import ContactForceRewardSpec, TrainingSpec
from learning.offline.dataset import (
    TransitionDataset,
    concat_datasets,
    episode_split,
    load_snapshot,
    nested_subsets,
)
from learning.offline.rewards import (
    require_contact_fields,
    derive_contact_thresholds,
    relabel_contact_force,
)
from learning.offline.training import (
    fraction_label,
    load_policy,
    train_contact_q,
)

MANIFEST_KEYS = (
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
DEFAULT_FEET = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", nargs="+", required=True)
    parser.add_argument("--dataset_label", default="data")
    parser.add_argument("--label", default="contact_q")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--foot_names",
        nargs="+",
        default=list(DEFAULT_FEET),
        help="Canonical body names of the feet, in threshold order.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=None,
        help="Explicit per-foot thresholds in bodyweight. Omit to derive them "
        "from the snapshot at --percentile.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=90.0,
        help="Percentile of the per-touchdown peak distribution used as the "
        "threshold when --thresholds is omitted.",
    )
    parser.add_argument(
        "--touchdown_threshold_bw",
        type=float,
        default=0.05,
        help="Load above which a control step counts as foot contact.",
    )
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument(
        "--subset_group",
        choices=("transition", "episode"),
        required=True,
    )
    parser.add_argument("--subset_seed", type=int, default=0)
    parser.add_argument(
        "--validation_fraction",
        type=float,
        default=0.0,
        help="Episode-grouped held-out fraction; 0 disables the split and "
        "validation reporting.",
    )
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument(
        "--validation_every",
        type=int,
        default=25_000,
        help="Optimizer updates between held-out Bellman-residual checks.",
    )
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


def validate_manifest_compatibility(datasets: list[TransitionDataset]) -> None:
    first = datasets[0].manifest
    for dataset in datasets[1:]:
        for key in MANIFEST_KEYS:
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


def resolve_thresholds(
    dataset: TransitionDataset,
    feet: tuple[str, ...],
    args,
) -> dict[str, float]:
    """Per-foot thresholds in bodyweight, from the CLI or the snapshot."""
    if args.thresholds is not None:
        if len(args.thresholds) != len(feet):
            raise ValueError(
                f"--thresholds needs {len(feet)} values, got {len(args.thresholds)}"
            )
        return {
            name: float(value)
            for name, value in zip(feet, args.thresholds, strict=True)
        }
    return derive_contact_thresholds(
        dataset,
        feet,
        args.percentile,
        contact_threshold_bw=args.touchdown_threshold_bw,
    )


def main() -> None:
    args = build_parser().parse_args()
    feet = tuple(args.foot_names)
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
    datasets = [load_snapshot(path) for path in args.snapshot]
    validate_manifest_compatibility(datasets)
    dataset = concat_datasets(datasets)
    require_contact_fields(dataset)
    source_revision = args.source_revision or dataset.manifest.get(
        "source_revision", "unknown"
    )

    policy, policy_spec = load_policy(
        dataset.manifest,
        args.checkpoint,
        args.device,
    )
    if args.validation_fraction > 0:
        pool, validation = episode_split(
            dataset,
            args.validation_fraction,
            seed=args.split_seed,
        )
        print(
            f"split train={len(pool)} validation={len(validation)} "
            f"fraction={args.validation_fraction} seed={args.split_seed}"
        )
    else:
        pool, validation = dataset, None

    # Thresholds are derived from the training pool only: the held-out split
    # must not influence the reward definition.
    thresholds = resolve_thresholds(pool, feet, args)
    reward_spec = ContactForceRewardSpec(
        foot_names=feet,
        thresholds_bw=tuple(thresholds[name] for name in feet),
        percentile=args.percentile,
    )
    print(
        "thresholds_bw="
        + " ".join(f"{name}:{thresholds[name]:.4f}" for name in feet)
        + f" percentile={args.percentile}"
    )
    relabeled = relabel_contact_force(pool, reward_spec)
    relabeled_validation = (
        None if validation is None else relabel_contact_force(validation, reward_spec)
    )
    rate = float((relabeled.data["reward"] < 0).float().mean())
    print(f"label_rate={rate:.5f} transitions={len(relabeled)}")
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
            / args.dataset_label
            / fraction_label(fraction)
        )
        train_contact_q(
            subset,
            policy,
            policy_spec,
            spec=training_spec,
            output_dir=output_dir,
            subset={
                "fraction": fraction,
                "group_by": args.subset_group,
                "seed": args.subset_seed,
                "dataset_label": args.dataset_label,
                "validation_fraction": args.validation_fraction,
                "split_seed": args.split_seed,
            },
            source_revision=source_revision,
            validation_dataset=relabeled_validation,
            validation_every=args.validation_every,
            report_path=output_dir.parent / f"{fraction_label(fraction)}_report.json",
        )


if __name__ == "__main__":
    main()
