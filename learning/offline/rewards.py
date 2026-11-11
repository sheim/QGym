"""Observation-based reward relabeling."""

from __future__ import annotations

from dataclasses import asdict

import torch

from .contracts import HeightRewardSpec
from .dataset import TransitionDataset


def relabel_height(
    dataset: TransitionDataset, spec: HeightRewardSpec
) -> TransitionDataset:
    """Label transitions from their recorded post-action height.

    The source snapshot is immutable; the returned dataset gains a ``reward``
    field. Height exactly on the bound is inside.
    """
    height = dataset.data["successor_height_m"].squeeze(1)
    inside = (height - spec.reference_height_m).abs() <= spec.bound_m
    reward = torch.where(
        inside,
        torch.full_like(height, spec.inside_reward),
        torch.full_like(height, spec.outside_reward),
    )
    data = dict(dataset.data)
    data["reward"] = reward
    manifest = dict(dataset.manifest)
    manifest["reward_spec"] = asdict(spec)
    manifest["reward_field"] = "reward"
    return TransitionDataset(data=data, manifest=manifest, source=dataset.source)
