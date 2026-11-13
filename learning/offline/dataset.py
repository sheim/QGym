"""Load replay snapshots and build deterministic training subsets."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .contracts import (
    REPLAY_FORMAT,
    SCHEMA_VERSION,
    TRANSITION_FIELDS,
    jsonable,
    replay_config_fingerprint,
)


@dataclass
class TransitionDataset:
    data: dict[str, torch.Tensor]
    manifest: dict
    source: str | None = None

    def __len__(self) -> int:
        return int(self.data["actor_obs"].shape[0])

    def to(self, device: str | torch.device) -> "TransitionDataset":
        return TransitionDataset(
            data={name: value.to(device) for name, value in self.data.items()},
            manifest=dict(self.manifest),
            source=self.source,
        )

    def subset(self, indices: torch.Tensor) -> "TransitionDataset":
        if indices.dtype != torch.int64:
            raise ValueError("subset indices must be int64")
        if indices.ndim != 1:
            raise ValueError("subset indices must be one-dimensional")
        if len(indices) == 0:
            raise ValueError("subset indices must not be empty")
        if int(indices.min()) < 0 or int(indices.max()) >= len(self):
            raise ValueError("subset indices are out of range")
        data = {name: value[indices].clone() for name, value in self.data.items()}
        manifest = dict(self.manifest)
        manifest["valid_count"] = int(len(indices))
        manifest["parent_snapshot"] = self.source
        return TransitionDataset(data=data, manifest=manifest, source=self.source)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_snapshot(source: str | Path) -> TransitionDataset:
    path = Path(source)
    manifest_path = path / "manifest.json"
    data_path = path / "data.pt"
    if not manifest_path.is_file() or not data_path.is_file():
        raise FileNotFoundError(f"snapshot is incomplete or missing: {path}")
    manifest = json.loads(manifest_path.read_text())
    _require(manifest.get("format") == REPLAY_FORMAT, "unsupported replay format")
    _require(
        manifest.get("schema_version") == SCHEMA_VERSION,
        "unsupported replay schema version",
    )
    data = torch.load(data_path, map_location="cpu", weights_only=True)
    _require(isinstance(data, dict), "snapshot data must be a tensor dictionary")
    _require(
        set(data.keys()) == set(TRANSITION_FIELDS),
        "snapshot fields do not match the replay contract",
    )

    valid_count = int(manifest["valid_count"])
    _require(valid_count > 0, "snapshot contains no transitions")
    for name in TRANSITION_FIELDS:
        _require(data[name].shape[0] == valid_count, f"{name} has wrong length")
    if data["successor_height_m"].ndim == 1:
        data["successor_height_m"] = data["successor_height_m"][:, None]
    _require(
        data["successor_height_m"].shape[1:] == (1,),
        "successor_height_m must have shape [N, 1]",
    )

    obs_dim = int(manifest["obs_dim"])
    action_dim = int(manifest["action_dim"])
    _require(
        sum(int(dim) for dim in manifest["obs_dims"]) == obs_dim,
        "obs_dims do not sum to obs_dim",
    )
    _require(
        sum(int(dim) for dim in manifest["action_dims"]) == action_dim,
        "action_dims do not sum to action_dim",
    )
    _require(
        len(manifest["obs_names"]) == len(manifest["obs_dims"]),
        "obs_names and obs_dims have different lengths",
    )
    _require(
        len(manifest["action_names"]) == len(manifest["action_dims"]),
        "action_names and action_dims have different lengths",
    )
    if "config_fingerprint" not in manifest:
        if "env_cfg" not in manifest:
            raise ValueError("snapshot manifest is missing environment provenance")
        manifest["config_fingerprint"] = replay_config_fingerprint(manifest["env_cfg"])
    _require(data["actor_obs"].shape[1:] == (obs_dim,), "actor_obs has wrong width")
    _require(
        data["next_actor_obs"].shape[1:] == (obs_dim,),
        "next_actor_obs has wrong width",
    )
    _require(data["action"].shape[1:] == (action_dim,), "action has wrong width")

    for name in ("actor_obs", "next_actor_obs", "action", "successor_height_m"):
        _require(
            data[name].dtype == torch.float32,
            f"{name} must be float32",
        )
        _require(
            bool(torch.isfinite(data[name]).all()),
            f"{name} contains non-finite values",
        )
    for name in ("terminated", "timed_out"):
        _require(data[name].dtype == torch.bool, f"{name} must be bool")
        _require(data[name].shape == (valid_count,), f"{name} has wrong shape")
    for name in ("episode_id", "step_id", "env_id", "behavior_iteration"):
        _require(data[name].dtype == torch.int64, f"{name} must be int64")
        _require(data[name].shape == (valid_count,), f"{name} has wrong shape")
        _require(int(data[name].min()) >= 0, f"{name} must be nonnegative")

    return TransitionDataset(data=data, manifest=manifest, source=str(path))


def _concat_field(datasets: list[TransitionDataset], name: str) -> torch.Tensor:
    return torch.cat([dataset.data[name] for dataset in datasets], dim=0)


def concat_datasets(
    datasets: list[TransitionDataset], *, deduplicate: bool = True
) -> TransitionDataset:
    if not datasets:
        raise ValueError("at least one dataset is required")
    field_set = set(datasets[0].data.keys())
    for dataset in datasets[1:]:
        if set(dataset.data.keys()) != field_set:
            raise ValueError("datasets have incompatible fields")
        for name in TRANSITION_FIELDS:
            if dataset.data[name].shape[1:] != datasets[0].data[name].shape[1:]:
                raise ValueError(f"datasets have incompatible {name} shapes")

    data = {name: _concat_field(datasets, name) for name in TRANSITION_FIELDS}
    manifest = dict(datasets[0].manifest)
    manifest["valid_count"] = int(data["actor_obs"].shape[0])
    manifest["source_snapshots"] = [dataset.source for dataset in datasets]
    manifest["run_id"] = "combined"

    keep = torch.arange(len(data["actor_obs"]))
    run_ids = [str(dataset.manifest.get("run_id", "unknown")) for dataset in datasets]
    if deduplicate and len(set(run_ids)) != len(run_ids):
        seen: set[tuple[str, int, int, int]] = set()
        keep_list: list[int] = []
        run_id_per_dataset = []
        for dataset, run_id in zip(datasets, run_ids, strict=True):
            run_id_per_dataset.extend([run_id] * len(dataset))
        for index in range(len(data["actor_obs"])):
            key = (
                run_id_per_dataset[index],
                int(data["env_id"][index]),
                int(data["episode_id"][index]),
                int(data["step_id"][index]),
            )
            if key in seen:
                continue
            seen.add(key)
            keep_list.append(index)
        keep = torch.tensor(keep_list, dtype=torch.int64)
        removed = len(data["actor_obs"]) - len(keep)
        if removed:
            data = {name: value[keep].clone() for name, value in data.items()}
            manifest["valid_count"] = int(data["actor_obs"].shape[0])
            manifest["deduplicated"] = removed

    return TransitionDataset(data=data, manifest=manifest, source="combined")


def _validate_fractions(fractions: tuple[float, ...]) -> None:
    if len(fractions) == 0:
        raise ValueError("at least one fraction is required")
    if any(not 0.0 < fraction <= 1.0 for fraction in fractions):
        raise ValueError("fractions must satisfy 0 < fraction <= 1")
    if any(
        fractions[index] >= fractions[index + 1] for index in range(len(fractions) - 1)
    ):
        raise ValueError("fractions must be strictly increasing")
    if fractions[-1] != 1.0:
        raise ValueError("the largest fraction must be 1.0")


def _episode_groups(dataset: TransitionDataset) -> tuple[torch.Tensor, int]:
    env_ids = dataset.data["env_id"].numpy()
    episode_ids = dataset.data["episode_id"].numpy()
    pairs = np.stack((env_ids, episode_ids), axis=1)
    _, inverse = np.unique(pairs, axis=0, return_inverse=True)
    return torch.from_numpy(inverse.astype(np.int64)), int(inverse.max()) + 1


def nested_subsets(
    dataset: TransitionDataset,
    fractions: tuple[float, ...],
    *,
    seed: int,
    group_by: str,
) -> list[TransitionDataset]:
    """Return deterministic nested subsets of one transition pool."""
    _validate_fractions(fractions)
    if group_by not in ("transition", "episode"):
        raise ValueError("group_by must be 'transition' or 'episode'")
    generator = torch.Generator().manual_seed(seed)
    size = len(dataset)

    if group_by == "transition":
        permutation = torch.randperm(size, generator=generator)
        subsets = []
        for fraction in fractions:
            count = max(1, math.ceil(fraction * size))
            subsets.append(dataset.subset(permutation[:count]))
        return subsets

    inverse, episode_count = _episode_groups(dataset)
    permutation = torch.randperm(episode_count, generator=generator)
    subsets = []
    for fraction in fractions:
        count = max(1, math.ceil(fraction * episode_count))
        selected = permutation[:count]
        indices = torch.isin(inverse, selected).nonzero().flatten()
        subsets.append(dataset.subset(indices))
    return subsets


def episode_split(
    dataset: TransitionDataset, validation_fraction: float, *, seed: int
) -> tuple[TransitionDataset, TransitionDataset]:
    """Split complete episodes into disjoint train/validation datasets.

    Grouping is the ``(env_id, episode_id)`` pair, so no episode contributes rows
    to both sides. The returned manifests record the split identity; observation
    normalization must be fitted on the train side only.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    inverse, group_count = _episode_groups(dataset)
    if group_count < 2:
        raise ValueError("episode split needs at least two episode groups")
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(group_count, generator=generator)
    validation_groups = max(1, round(validation_fraction * group_count))
    if validation_groups >= group_count:
        raise ValueError("validation split would leave no training episodes")
    selected = permutation[:validation_groups]
    validation_mask = torch.isin(inverse, selected)
    train = dataset.subset((~validation_mask).nonzero().flatten())
    validation = dataset.subset(validation_mask.nonzero().flatten())

    def groups(item: TransitionDataset) -> set[tuple[int, int]]:
        return {
            (int(env_id), int(episode_id))
            for env_id, episode_id in zip(
                item.data["env_id"].tolist(),
                item.data["episode_id"].tolist(),
                strict=True,
            )
        }

    overlap = groups(train) & groups(validation)
    if overlap:
        raise ValueError(f"episode split leaked {len(overlap)} groups across sides")
    for item, label in ((train, "train"), (validation, "validation")):
        item.manifest["validation_fraction"] = validation_fraction
        item.manifest["split_seed"] = seed
        item.manifest["split_side"] = label
    return train, validation


def sample_n(
    dataset: TransitionDataset, count: int, *, seed: int, group_by: str
) -> TransitionDataset:
    """Take a deterministic equal-size subset for balanced comparisons."""
    if count <= 0 or count > len(dataset):
        raise ValueError("count must be in [1, len(dataset)]")
    if group_by not in ("transition", "episode"):
        raise ValueError("group_by must be 'transition' or 'episode'")
    generator = torch.Generator().manual_seed(seed)
    if group_by == "transition":
        permutation = torch.randperm(len(dataset), generator=generator)
        return dataset.subset(permutation[:count])

    inverse, episode_count = _episode_groups(dataset)
    permutation = torch.randperm(episode_count, generator=generator)
    selected: list[int] = []
    selected_episodes = 0
    while len(selected) < count and selected_episodes < episode_count:
        episode = permutation[selected_episodes]
        selected_episodes += 1
        selected.extend(
            torch.isin(inverse, episode.reshape(1)).nonzero().flatten().tolist()
        )
    return dataset.subset(torch.tensor(selected[:count], dtype=torch.int64))


def write_manifest(path: str | Path, manifest: dict) -> None:
    Path(path).write_text(
        json.dumps(jsonable(manifest), indent=2, sort_keys=True) + "\n"
    )
