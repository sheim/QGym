"""Persist and load fitted height Q artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from .contracts import (
    HEIGHT_Q_FORMAT,
    RETURN_CONVENTION,
    SCHEMA_VERSION,
    PolicySpec,
    TrainingSpec,
    jsonable,
    utc_timestamp,
)
from learning.algorithms.fqe import QNetwork


class HeightQ:
    """Runtime interface for an unprivileged actor-observation Q function."""

    def __init__(self, directory: str | Path) -> None:
        path = Path(directory)
        manifest_path = path / "manifest.json"
        weights_path = path / "q.pt"
        if not manifest_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(f"Q artifact is incomplete: {path}")
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("format") != HEIGHT_Q_FORMAT:
            raise ValueError("unsupported Q artifact format")
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported Q artifact schema version")
        if self.manifest.get("return_convention") != RETURN_CONVENTION:
            raise ValueError("unsupported return convention")
        self.obs_dim = int(self.manifest["obs_dim"])
        self.action_dim = int(self.manifest["action_dim"])
        self.q = QNetwork(
            self.obs_dim,
            self.action_dim,
            tuple(self.manifest["hidden_dims"]),
            str(self.manifest["activation"]),
        )
        self.q.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True)
        )
        self.q.eval()
        self.obs_mean = torch.tensor(self.manifest["obs_mean"], dtype=torch.float32)
        self.obs_std = torch.tensor(self.manifest["obs_std"], dtype=torch.float32)
        if self.obs_mean.shape != (self.obs_dim,):
            raise ValueError("obs_mean has wrong shape")
        if self.obs_std.shape != (self.obs_dim,):
            raise ValueError("obs_std has wrong shape")
        if not torch.all(self.obs_std > 0):
            raise ValueError("obs_std must be positive")

    @torch.no_grad()
    def predict(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        obs = torch.as_tensor(observations, dtype=torch.float32)
        act = torch.as_tensor(actions, dtype=torch.float32)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if act.ndim == 1:
            act = act.unsqueeze(0)
        if obs.shape[1] != self.obs_dim:
            raise ValueError("observations have wrong width")
        if act.shape[1] != self.action_dim:
            raise ValueError("actions have wrong width")
        if obs.shape[0] == 1 and act.shape[0] > 1:
            obs = obs.expand(act.shape[0], -1)
        if obs.shape[0] != act.shape[0]:
            raise ValueError("observations and actions need matching rows")
        device = next(self.q.parameters()).device
        obs = obs.to(device)
        act = act.to(device)
        mean = self.obs_mean.to(device)
        std = self.obs_std.to(device)
        prediction = self.q((obs - mean) / std, act)
        if not torch.isfinite(prediction).all():
            raise ValueError("Q prediction is non-finite")
        return prediction


def save_height_q(
    path: str | Path,
    q_network: QNetwork,
    *,
    obs_mean: torch.Tensor,
    obs_std: torch.Tensor,
    gamma: float,
    failure_value: float,
    reward_spec: dict,
    policy_spec: PolicySpec,
    training_spec: TrainingSpec,
    dataset_manifest: dict,
    subset: dict,
    source_revision: str,
) -> Path:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Q artifact already exists: {destination}")
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary Q artifact already exists: {temporary}")
    if obs_mean.shape != (q_network.obs_dim,):
        raise ValueError("obs_mean has wrong shape")
    if obs_std.shape != (q_network.obs_dim,):
        raise ValueError("obs_std has wrong shape")
    if not torch.isfinite(obs_mean).all() or not torch.isfinite(obs_std).all():
        raise ValueError("observation statistics must be finite")
    if not torch.all(obs_std > 0):
        raise ValueError("observation standard deviations must be positive")

    sources = dataset_manifest.get("source_snapshots")
    if not sources:
        source = dataset_manifest.get("source")
        sources = [source] if source else []

    manifest = {
        "format": HEIGHT_Q_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_timestamp(),
        "obs_dim": q_network.obs_dim,
        "action_dim": q_network.action_dim,
        "hidden_dims": list(q_network.hidden_dims),
        "activation": q_network.activation,
        "obs_mean": obs_mean.detach().cpu().tolist(),
        "obs_std": obs_std.detach().cpu().tolist(),
        "gamma": gamma,
        "failure_value": failure_value,
        "reward_spec": reward_spec,
        "return_convention": RETURN_CONVENTION,
        "policy_spec": asdict(policy_spec),
        "training_spec": asdict(training_spec),
        "dataset": {
            "source": sources,
            "run_id": dataset_manifest.get("run_id"),
            "valid_count": dataset_manifest.get("valid_count"),
            "config_fingerprint": dataset_manifest.get("config_fingerprint"),
            "subset": subset,
        },
        "source_revision": source_revision,
        "validated_domain": None,
        "validation_report_reference": None,
    }
    temporary.mkdir(parents=True)
    torch.save(q_network.state_dict(), temporary / "q.pt")
    (temporary / "manifest.json").write_text(
        json.dumps(jsonable(manifest), indent=2, sort_keys=True) + "\n"
    )
    temporary.rename(destination)
    return destination
