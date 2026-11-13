"""Observation-based reward relabeling."""

from __future__ import annotations

from dataclasses import asdict

import torch

from .contracts import AngularVelocityRewardSpec, HeightRewardSpec
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


def angular_velocity_reward(
    command: torch.Tensor,
    ang_vel: torch.Tensor,
    spec: AngularVelocityRewardSpec,
) -> torch.Tensor:
    """Unmasked yaw tracking term, exactly as the task computes it before weights."""
    if command.shape != ang_vel.shape:
        raise ValueError("command and ang_vel must have matching shapes")
    error = (command - ang_vel) / spec.error_scale
    return torch.exp(-error.square() / spec.tracking_sigma)


def observation_field_slice(manifest: dict, name: str) -> slice:
    """Locate one recorded actor-observation block by manifest name, not index."""
    names = list(manifest["obs_names"])
    dims = [int(dim) for dim in manifest["obs_dims"]]
    if len(names) != len(dims):
        raise ValueError("obs_names and obs_dims have different lengths")
    if name not in names:
        raise ValueError(f"snapshot does not record observation {name!r}")
    offset = 0
    for candidate, dim in zip(names, dims, strict=True):
        if candidate == name:
            return slice(offset, offset + dim)
        offset += dim
    raise ValueError(f"snapshot does not record observation {name!r}")


def relabel_angular_velocity(
    dataset: TransitionDataset, spec: AngularVelocityRewardSpec
) -> TransitionDataset:
    """Label transitions from the successor yaw command and yaw rate.

    Both fields come from ``next_actor_obs`` so the command is the one that
    governed the recorded interval: the task resamples commands inside
    ``_post_decimation_step`` and the runner computes rewards after ``step()``.
    The source snapshot is immutable; the returned dataset gains a ``reward``
    field that is exactly zero on true termination.
    """
    manifest = dataset.manifest
    command_slice = observation_field_slice(manifest, "commands")
    ang_vel_slice = observation_field_slice(manifest, "base_ang_vel")
    if command_slice.stop - command_slice.start < 3:
        raise ValueError("commands observation must have at least three axes")
    if ang_vel_slice.stop - ang_vel_slice.start < 3:
        raise ValueError("base_ang_vel observation must have at least three axes")
    successor = dataset.data["next_actor_obs"]
    if successor.shape[1] != sum(int(dim) for dim in manifest["obs_dims"]):
        raise ValueError("next_actor_obs width does not match obs_dims")
    reward = angular_velocity_reward(
        successor[:, command_slice][:, 2],
        successor[:, ang_vel_slice][:, 2],
        spec,
    )
    reward = reward * (~dataset.data["terminated"])
    data = dict(dataset.data)
    data["reward"] = reward
    manifest = dict(dataset.manifest)
    manifest["reward_spec"] = asdict(spec)
    manifest["reward_field"] = "reward"
    return TransitionDataset(data=data, manifest=manifest, source=dataset.source)
