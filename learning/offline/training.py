"""Orchestrate fixed-policy Q fitting and artifact export."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from learning.algorithms.fqe import FQE, QNetwork
from learning.offline.contracts import (
    AngularVelocityRewardSpec,
    PolicySpec,
    TrainingSpec,
    jsonable,
    sha256_file,
    utc_timestamp,
)
from learning.offline.dataset import TransitionDataset
from learning.offline.nominal_policy import FrozenTrotPolicy
from learning.offline.q_artifact import save_angular_velocity_q, save_height_q


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


@torch.no_grad()
def _bellman_metrics(
    data: dict[str, torch.Tensor],
    indices: torch.Tensor,
    policy: FrozenTrotPolicy,
    algorithm: FQE,
    obs_mean: torch.Tensor,
    obs_std: torch.Tensor,
    generator: torch.Generator,
    chunk_size: int = 16_384,
) -> dict:
    """One-step Bellman residual of the current Q on an explicit row set.

    The target uses the frozen policy's sampled successor action and the same
    terminal convention as training. This is a TD residual on fixed data, not a
    held-out return estimate; it reports whether the fit explains its own
    bootstrap, not whether the Q is accurate off-distribution.
    """
    squared = 0.0
    absolute = 0.0
    bias = 0.0
    prediction_sum = 0.0
    target_sum = 0.0
    rows = int(indices.numel())
    for start in range(0, rows, chunk_size):
        chunk = indices[start : start + chunk_size]
        batch = _batch(data, chunk, obs_mean=obs_mean, obs_std=obs_std)
        target = algorithm.compute_target(batch, generator)
        prediction = algorithm.q(batch["q_obs"], batch["action"])
        difference = prediction - target
        squared += float(difference.square().sum())
        absolute += float(difference.abs().sum())
        bias += float(difference.sum())
        prediction_sum += float(prediction.sum())
        target_sum += float(target.sum())
    return {
        "rows": rows,
        "mse": squared / rows,
        "rmse": (squared / rows) ** 0.5,
        "mae": absolute / rows,
        "bias": bias / rows,
        "prediction_mean": prediction_sum / rows,
        "target_mean": target_sum / rows,
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


def train_angular_velocity_q(
    dataset: TransitionDataset,
    policy: FrozenTrotPolicy,
    policy_spec: PolicySpec,
    *,
    spec: TrainingSpec,
    output_dir: str | Path,
    subset: dict,
    source_revision: str,
    validation_dataset: TransitionDataset | None = None,
    validation_every: int = 0,
    report_path: str | Path | None = None,
) -> Path:
    """Fit ``Q_{r'}^{pi_r}`` for the yaw tracking term.

    Same FQE loop as ``train_height_q``; only the reward contract differs: the
    recorded spec must be a zero-after-terminal angular-velocity reward and the
    terminal target is the observed (zero) reward with no absorbing tail.

    ``validation_dataset`` must be an episode-disjoint split; it is never used
    for optimization or observation normalization. Its one-step Bellman residual
    is the convergence evidence written to ``report_path``.
    """
    if "reward" not in dataset.data:
        raise ValueError("dataset must be relabeled before training")
    reward_spec = dataset.manifest.get("reward_spec")
    if reward_spec is None:
        raise ValueError("dataset manifest is missing reward_spec")
    reward_spec = AngularVelocityRewardSpec(**reward_spec)
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
    # Normalization stays fitted on the training split; the held-out split is
    # only read for diagnostics.
    validation_data = (
        None
        if validation_dataset is None
        else {name: value.to(device) for name, value in validation_dataset.data.items()}
    )

    q_network = QNetwork(
        int(actor_obs.shape[1]),
        int(data["action"].shape[1]),
        spec.hidden_dims,
        spec.activation,
    )
    failure_value = reward_spec.failure_value(spec.gamma)
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
    diagnostic_generator = torch.Generator(device=device).manual_seed(spec.seed + 7)
    results: dict = {}
    if validation_data is not None:
        validation_indices = torch.arange(
            len(validation_dataset), device=device, dtype=torch.int64
        )
        results["validation_curve"] = [
            {
                "update": 0,
                **_bellman_metrics(
                    validation_data,
                    validation_indices,
                    policy,
                    algorithm,
                    obs_mean,
                    obs_std,
                    diagnostic_generator,
                ),
            }
        ]
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
        if (
            validation_data is not None
            and validation_every > 0
            and (update + 1) % validation_every == 0
        ):
            results["validation_curve"].append(
                {
                    "update": update + 1,
                    **_bellman_metrics(
                        validation_data,
                        validation_indices,
                        policy,
                        algorithm,
                        obs_mean,
                        obs_std,
                        diagnostic_generator,
                    ),
                }
            )

    artifact = save_angular_velocity_q(
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
    if report_path is not None:
        if validation_data is not None:
            curve = results["validation_curve"]
            if curve[-1]["update"] != spec.updates:
                curve.append(
                    {
                        "update": spec.updates,
                        **_bellman_metrics(
                            validation_data,
                            validation_indices,
                            policy,
                            algorithm,
                            obs_mean,
                            obs_std,
                            diagnostic_generator,
                        ),
                    }
                )
        loss_tensor = torch.tensor(losses, dtype=torch.float64)
        window = min(1000, len(losses))
        train_indices = torch.randperm(
            size, device=device, generator=diagnostic_generator
        )[: min(size, 8192)]
        report = {
            "created_at": utc_timestamp(),
            "artifact": str(artifact),
            "transitions": size,
            "updates": spec.updates,
            "batch_size": batch_size,
            "learning_rate": spec.learning_rate,
            "target_action_samples": spec.target_action_samples,
            "target_tau": spec.target_tau,
            "hidden_dims": list(spec.hidden_dims),
            "activation": spec.activation,
            "gamma": spec.gamma,
            "seed": spec.seed,
            "device": spec.device,
            "normalize_observations": spec.normalize_observations,
            "reward_spec": dataset.manifest["reward_spec"],
            "loss_curve": {
                "first": losses[0],
                "last": losses[-1],
                "mean": mean_loss,
                "min": float(loss_tensor.min()),
                "max": float(loss_tensor.max()),
                "first_1000_mean": float(loss_tensor[:window].mean()),
                "last_1000_mean": float(loss_tensor[-window:].mean()),
                "below_1_fraction_last_1000": float(
                    (loss_tensor[-window:] < 1.0).double().mean()
                ),
            },
            "train_sample": _bellman_metrics(
                data,
                train_indices,
                policy,
                algorithm,
                obs_mean,
                obs_std,
                diagnostic_generator,
            ),
            "validation": None
            if validation_dataset is None
            else {
                "rows": len(validation_dataset),
                "fraction": validation_dataset.manifest.get("validation_fraction"),
                "split_seed": validation_dataset.manifest.get("split_seed"),
                "curve": results["validation_curve"],
                "initial": results["validation_curve"][0],
                "final": results["validation_curve"][-1],
            },
        }
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(
            json.dumps(jsonable(report), indent=2, sort_keys=True) + "\n"
        )
        print(
            f"report={report_path} "
            f"loss_last1000={report['loss_curve']['last_1000_mean']:.4f}"
        )
    return artifact
