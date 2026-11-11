"""Orchestrate height Q fitting and artifact export."""

from __future__ import annotations

from pathlib import Path

import torch

from learning.algorithms.fqe import FQE, QNetwork
from learning.offline.contracts import (
    PolicySpec,
    TrainingSpec,
    sha256_file,
)
from learning.offline.dataset import TransitionDataset
from learning.offline.nominal_policy import FrozenTrotPolicy
from learning.offline.q_artifact import save_height_q


def fraction_label(fraction: float) -> str:
    return f"{round(fraction * 100):03d}pct"


def policy_spec_from_manifest(manifest: dict, checkpoint: str | Path) -> PolicySpec:
    return PolicySpec(
        checkpoint=str(checkpoint),
        checkpoint_sha256=sha256_file(checkpoint),
        actor_cfg=manifest["actor_cfg"],
        obs_names=tuple(manifest["obs_names"]),
        obs_dims=tuple(int(dim) for dim in manifest["obs_dims"]),
        action_names=tuple(manifest["action_names"]),
        action_dim=int(manifest["action_dim"]),
        noise=manifest["noise"],
        scales=manifest["scales"],
        control_dt=float(manifest["control_dt"]),
        n_policy_steps=int(manifest["n_policy_steps"]),
        episode_length_s=float(manifest["episode_length_s"]),
        smooth_exploration=bool(manifest["actor_cfg"]["smooth_exploration"]),
    )


def load_policy(
    manifest: dict, checkpoint: str | Path, device: str
) -> tuple[FrozenTrotPolicy, PolicySpec]:
    spec = policy_spec_from_manifest(manifest, checkpoint)
    policy = FrozenTrotPolicy.from_checkpoint(
        checkpoint,
        spec.actor_cfg,
        spec.obs_names,
        spec.obs_dims,
        spec.action_names,
        spec.action_dim,
        device,
    )
    return policy, spec


def _batch(
    data: dict[str, torch.Tensor],
    indices: torch.Tensor,
    *,
    obs_mean: torch.Tensor,
    obs_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    actor_obs = data["actor_obs"][indices]
    next_actor_obs = data["next_actor_obs"][indices]
    return {
        "q_obs": (actor_obs - obs_mean) / obs_std,
        "q_next_obs": (next_actor_obs - obs_mean) / obs_std,
        "policy_next_obs": next_actor_obs,
        "action": data["action"][indices],
        "reward": data["reward"][indices],
        "terminated": data["terminated"][indices],
    }


def train_height_q(
    dataset: TransitionDataset,
    policy: FrozenTrotPolicy,
    policy_spec: PolicySpec,
    *,
    spec: TrainingSpec,
    output_dir: str | Path,
    subset: dict,
    source_revision: str,
) -> Path:
    if "reward" not in dataset.data:
        raise ValueError("dataset must be relabeled before training")
    if dataset.manifest.get("reward_spec") is None:
        raise ValueError("dataset manifest is missing reward_spec")
    if len(dataset) == 0:
        raise ValueError("training dataset is empty")

    device = torch.device(spec.device)
    data = {name: value.to(device) for name, value in dataset.data.items()}
    actor_obs = data["actor_obs"]
    if spec.normalize_observations:
        obs_mean = actor_obs.mean(dim=0)
        obs_std = actor_obs.std(dim=0) + 1e-3
    else:
        obs_mean = torch.zeros(actor_obs.shape[1], device=device, dtype=actor_obs.dtype)
        obs_std = torch.ones(actor_obs.shape[1], device=device, dtype=actor_obs.dtype)

    q_network = QNetwork(
        int(actor_obs.shape[1]),
        int(data["action"].shape[1]),
        spec.hidden_dims,
        spec.activation,
    )
    failure_value = dataset.manifest["reward_spec"]["outside_reward"] / (
        1.0 - spec.gamma
    )
    algorithm = FQE(
        q_network,
        policy,
        gamma=spec.gamma,
        failure_value=failure_value,
        target_tau=spec.target_tau,
        learning_rate=spec.learning_rate,
        device=device,
        target_action_samples=spec.target_action_samples,
    )
    generator = torch.Generator(device=device).manual_seed(spec.seed)
    size = len(dataset)
    batch_size = min(spec.batch_size, size)
    log_every = max(1, spec.updates // 10)
    losses: list[float] = []
    for update in range(spec.updates):
        indices = torch.randint(
            0,
            size,
            (batch_size,),
            device=device,
            generator=generator,
        )
        batch = _batch(
            data,
            indices,
            obs_mean=obs_mean,
            obs_std=obs_std,
        )
        loss = algorithm.update(batch, generator)
        losses.append(loss)
        if update == 0 or (update + 1) % log_every == 0:
            print(f"update={update + 1}/{spec.updates} loss={loss:.6f}")

    artifact = save_height_q(
        output_dir,
        algorithm.q,
        obs_mean=obs_mean,
        obs_std=obs_std,
        gamma=spec.gamma,
        failure_value=failure_value,
        reward_spec=dataset.manifest["reward_spec"],
        policy_spec=policy_spec,
        training_spec=spec,
        dataset_manifest={
            **dataset.manifest,
            "source": dataset.source,
        },
        subset=subset,
        source_revision=source_revision,
    )
    mean_loss = sum(losses) / len(losses)
    print(
        f"artifact={artifact} mean_loss={mean_loss:.6f} "
        f"transitions={size} updates={spec.updates}"
    )
    return artifact
