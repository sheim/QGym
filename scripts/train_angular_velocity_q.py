"""Train a yaw angular-velocity Q by fixed-policy evaluation on replay snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path

from learning.offline.contracts import AngularVelocityRewardSpec, TrainingSpec
from learning.offline.dataset import (
    TransitionDataset,
    concat_datasets,
    episode_split,
    load_snapshot,
    nested_subsets,
)
from learning.offline.rewards import relabel_angular_velocity
from learning.offline.training import (
    fraction_label,
    load_policy,
    train_angular_velocity_q,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", nargs="+", required=True)
    parser.add_argument("--dataset_label", default="data")
    parser.add_argument("--label", default="angular_vel_q")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--sigma",
        type=float,
        required=True,
        help="tracking_sigma from cfg.reward_settings (0.25 for go2trot).",
    )
    parser.add_argument(
        "--error_scale",
        type=float,
        default=2.5,
        help=(
            "Yaw-error divisor inside Go2Trot._reward_tracking_ang_vel; it is "
            "hard-coded at 2.5 rad/s in the task, not a config field."
        ),
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


def validate_recorded_sigma(manifest: dict, spec: AngularVelocityRewardSpec) -> None:
    """The label must use the sigma the recorded run actually trained with."""
    recorded = manifest["env_cfg"]["reward_settings"]["tracking_sigma"]
    if float(recorded) != spec.tracking_sigma:
        raise ValueError(
            f"snapshot trained with tracking_sigma={recorded}, "
            f"requested {spec.tracking_sigma}"
        )


def main() -> None:
    args = build_parser().parse_args()
    reward_spec = AngularVelocityRewardSpec(
        tracking_sigma=args.sigma,
        error_scale=args.error_scale,
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
    datasets = [load_snapshot(path) for path in args.snapshot]
    validate_manifest_compatibility(datasets)
    dataset = concat_datasets(datasets)
    validate_recorded_sigma(dataset.manifest, reward_spec)
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
    relabeled = relabel_angular_velocity(pool, reward_spec)
    relabeled_validation = (
        None
        if validation is None
        else relabel_angular_velocity(validation, reward_spec)
    )
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
        train_angular_velocity_q(
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
