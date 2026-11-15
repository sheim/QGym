"""Observation-based reward relabeling."""

from __future__ import annotations

from dataclasses import asdict

import torch

import numpy as np

from .contracts import (
    AngularVelocityRewardSpec,
    ContactForceRewardSpec,
    HeightRewardSpec,
)
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


def require_contact_fields(dataset: TransitionDataset) -> list[str]:
    """Validate that the snapshot carries the contact-force contract."""
    manifest = dataset.manifest
    required = ("contact_forces", "next_contact_forces", "bodyweight")
    extra_fields = tuple(str(name) for name in manifest.get("extra_fields", ()))
    missing = [name for name in required if name not in extra_fields]
    if missing:
        raise ValueError(
            "snapshot was not recorded with the contact-force contract; missing "
            f"fields {missing}. Record with --replay_fields=contacts."
        )
    for name in ("contact_forces", "next_contact_forces", "bodyweight"):
        if name not in dataset.data:
            raise ValueError(f"snapshot data is missing field {name!r}")
    metadata = manifest.get("field_metadata") or {}
    body_names = metadata.get("body_names")
    if not body_names:
        raise ValueError("snapshot manifest is missing field_metadata.body_names")
    return list(body_names)


def foot_vertical_load(
    dataset: TransitionDataset, foot_names: tuple[str, ...]
) -> torch.Tensor:
    """Post-action per-foot vertical load in bodyweight units, ``[N, feet]``.

    Uses ``next_contact_forces`` (the force after the action, before the reset)
    and the recorded per-environment ``bodyweight``: startup mass randomisation
    varies bodyweight by up to +-20 %, so a nominal constant would mislabel.
    """
    body_names = require_contact_fields(dataset)
    missing = [name for name in foot_names if name not in body_names]
    if missing:
        raise ValueError(f"feet {missing} are not in the recorded body list")
    forces = dataset.data["next_contact_forces"]
    bodies = len(body_names)
    if forces.ndim != 2 or forces.shape[1] != 3 * bodies:
        raise ValueError(
            f"next_contact_forces width {tuple(forces.shape[1:])} does not match "
            f"{bodies} bodies x 3 axes"
        )
    bodyweight = dataset.data["bodyweight"]
    if bodyweight.ndim != 2 or bodyweight.shape[1] != 1:
        raise ValueError("bodyweight must be shaped [N, 1]")
    if not torch.all(bodyweight > 0):
        raise ValueError("bodyweight must be positive")
    feet = [body_names.index(name) for name in foot_names]
    fz = forces.reshape(-1, bodies, 3)[:, feet, 2]
    return fz / bodyweight


def contact_force_reward(
    load_bw: torch.Tensor, spec: ContactForceRewardSpec
) -> torch.Tensor:
    """``-1`` where any foot exceeds its threshold, otherwise ``0``."""
    if load_bw.ndim != 2 or load_bw.shape[1] != len(spec.foot_names):
        raise ValueError("load_bw must be shaped [N, len(foot_names)]")
    thresholds = torch.tensor(
        spec.thresholds_bw, dtype=load_bw.dtype, device=load_bw.device
    )
    exceed = load_bw > thresholds.unsqueeze(0)
    return -exceed.any(dim=1).to(load_bw.dtype)


def relabel_contact_force(
    dataset: TransitionDataset, spec: ContactForceRewardSpec
) -> TransitionDataset:
    """Label transitions from their post-action foot load.

    The reward is exactly zero on true termination: the constraint is bounded in
    ``[-1, 0]`` and a finished episode has no continuation. The source snapshot
    is immutable; the returned dataset gains a ``reward`` field.
    """
    load = foot_vertical_load(dataset, spec.foot_names)
    reward = contact_force_reward(load, spec)
    reward = reward * (~dataset.data["terminated"])
    data = dict(dataset.data)
    data["reward"] = reward
    manifest = dict(dataset.manifest)
    manifest["reward_spec"] = asdict(spec)
    manifest["reward_field"] = "reward"
    return TransitionDataset(data=data, manifest=manifest, source=dataset.source)


def derive_contact_thresholds(
    dataset: TransitionDataset,
    foot_names: tuple[str, ...],
    percentile: float,
    *,
    contact_threshold_bw: float = 0.05,
    min_steps: int = 5,
    min_peak_bw: float = 0.30,
) -> dict[str, float]:
    """Per-foot ``percentile`` of the per-touchdown peak load, in bodyweight.

    A touchdown is a contiguous run of at least ``min_steps`` control steps with
    load above ``contact_threshold_bw`` whose peak is at least ``min_peak_bw``;
    the run must stay inside one episode. Terminated rows are excluded. Rows are
    grouped by ``(env_id, episode_id)`` and ordered by ``step_id``, so the
    derivation does not depend on the recorded row order.
    """
    if not 0.0 < percentile < 100.0:
        raise ValueError("percentile must be strictly between 0 and 100")
    if min_steps <= 0:
        raise ValueError("min_steps must be positive")
    load = foot_vertical_load(dataset, foot_names).numpy()
    terminated = dataset.data["terminated"].numpy().astype(bool)
    env_id = dataset.data["env_id"].numpy()
    episode_id = dataset.data["episode_id"].numpy()
    step_id = dataset.data["step_id"].numpy()
    order = np.lexsort((step_id, episode_id, env_id))
    env_sorted = env_id[order]
    episode_sorted = episode_id[order]
    peaks = {name: [] for name in foot_names}
    group_start = 0
    rows = len(order)
    for index in range(1, rows + 1):
        if (
            index == rows
            or env_sorted[index] != env_sorted[group_start]
            or episode_sorted[index] != episode_sorted[group_start]
        ):
            group = order[group_start:index]
            group_start = index
            if group.size < min_steps:
                continue
            alive = ~terminated[group]
            for k, name in enumerate(foot_names):
                mask = (load[group, k] > contact_threshold_bw) & alive
                if not mask.any():
                    continue
                padded = np.concatenate([[False], mask, [False]])
                edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
                for start, end in zip(edges[0::2], edges[1::2], strict=True):
                    if end - start < min_steps:
                        continue
                    peak = float(load[group[start:end], k].max())
                    if peak >= min_peak_bw:
                        peaks[name].append(peak)
    missing = [name for name, values in peaks.items() if not values]
    if missing:
        raise ValueError(f"no touchdowns found for feet {missing}")
    return {
        name: float(np.percentile(np.asarray(values), percentile))
        for name, values in peaks.items()
    }
