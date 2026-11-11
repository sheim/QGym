"""Bounded transition recording with immutable disk snapshots."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from learning.offline.contracts import (
    ACTION_CONVENTION,
    REPLAY_FORMAT,
    REWARD_TIMING,
    SCHEMA_VERSION,
    TRANSITION_FIELDS,
    jsonable,
    replay_config_fingerprint,
    sample_observation_noise,
    utc_timestamp,
)


class TransitionRecorder:
    """Retain the most recent transitions and publish coherent snapshots.

    The ring stores host tensors so collection does not grow device memory.
    Recording is independent of PPO storage and must not consume training RNG.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        obs_names: tuple[str, ...],
        obs_dims: tuple[int, ...],
        action_names: tuple[str, ...],
        action_dims: tuple[int, ...],
        noise: dict,
        scales: dict,
        device: str,
        output_dir: str,
        capacity: int,
        snapshot_interval: int,
        seed: int,
        task_name: str,
        backend: str,
        source_revision: str,
        log_dir: str | None,
        actor_cfg: dict,
        env_cfg: dict,
        runner_cfg: dict,
        episode_length_s: float,
        n_policy_steps: int,
        control_dt: float,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if snapshot_interval < 0:
            raise ValueError("snapshot_interval must be nonnegative")
        if len(obs_names) != len(obs_dims):
            raise ValueError("obs_names and obs_dims must have equal length")
        if len(action_names) != len(action_dims):
            raise ValueError("action_names and action_dims must have equal length")
        if not output_dir:
            raise ValueError("output_dir is required")
        if n_policy_steps <= 0:
            raise ValueError("n_policy_steps must be positive")
        if n_policy_steps != 1:
            raise ValueError("replay recording currently requires n_policy_steps == 1")
        if control_dt <= 0:
            raise ValueError("control_dt must be positive")
        if episode_length_s <= 0:
            raise ValueError("episode_length_s must be positive")

        self.num_envs = num_envs
        self.obs_names = tuple(obs_names)
        self.obs_dims = tuple(obs_dims)
        self.obs_dim = sum(self.obs_dims)
        self.action_names = tuple(action_names)
        self.action_dims = tuple(action_dims)
        self.action_dim = sum(self.action_dims)
        self.noise = dict(noise)
        self.scales = dict(scales)
        self.device = device
        self.output_dir = Path(output_dir)
        self.capacity = capacity
        self.snapshot_interval = snapshot_interval
        self.seed = seed
        self.task_name = task_name
        self.backend = backend
        self.source_revision = source_revision
        self.log_dir = log_dir
        self.actor_cfg = actor_cfg
        self.env_cfg = env_cfg
        self.runner_cfg = runner_cfg
        self.episode_length_s = episode_length_s
        self.n_policy_steps = n_policy_steps
        self.control_dt = control_dt
        self.run_id = (
            Path(log_dir).name
            if log_dir
            else f"unlogged-{utc_timestamp().replace(':', '-')}"
        )

        self.write_index = 0
        self.total_appended = 0
        self.valid_count = 0
        self.last_snapshot_total = -1

        self._buffers: dict[str, torch.Tensor] = {
            "actor_obs": torch.zeros(capacity, self.obs_dim, dtype=torch.float32),
            "action": torch.zeros(capacity, self.action_dim, dtype=torch.float32),
            "next_actor_obs": torch.zeros(capacity, self.obs_dim, dtype=torch.float32),
            "successor_height_m": torch.zeros(capacity, 1, dtype=torch.float32),
            "terminated": torch.zeros(capacity, dtype=torch.bool),
            "timed_out": torch.zeros(capacity, dtype=torch.bool),
            "episode_id": torch.zeros(capacity, dtype=torch.int64),
            "step_id": torch.zeros(capacity, dtype=torch.int64),
            "env_id": torch.zeros(capacity, dtype=torch.int64),
            "behavior_iteration": torch.zeros(capacity, dtype=torch.int64),
        }
        self._episode_ids = torch.zeros(num_envs, dtype=torch.int64)
        self._step_ids = torch.zeros(num_envs, dtype=torch.int64)
        self._env_ids = torch.arange(num_envs, dtype=torch.int64)
        self._generator = torch.Generator(device=device)
        self._generator.manual_seed(seed)

    @property
    def snapshot_interval(self) -> int:
        return self._snapshot_interval

    @snapshot_interval.setter
    def snapshot_interval(self, value: int) -> None:
        if value < 0:
            raise ValueError("snapshot_interval must be nonnegative")
        self._snapshot_interval = value

    def _validate_tensor(
        self, name: str, tensor: torch.Tensor, shape: tuple[int, ...]
    ) -> None:
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"{name} has shape {tuple(tensor.shape)}, expected {shape}"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains non-finite values")

    def append(
        self,
        *,
        actor_obs: torch.Tensor,
        action: torch.Tensor,
        post_reset_obs: torch.Tensor,
        pre_reset_clean_obs: torch.Tensor,
        successor_height_m: torch.Tensor,
        terminated: torch.Tensor,
        timed_out: torch.Tensor,
        behavior_iteration: int,
    ) -> None:
        """Record one policy transition per environment."""
        self._validate_tensor("actor_obs", actor_obs, (self.num_envs, self.obs_dim))
        self._validate_tensor("action", action, (self.num_envs, self.action_dim))
        self._validate_tensor(
            "post_reset_obs", post_reset_obs, (self.num_envs, self.obs_dim)
        )
        self._validate_tensor(
            "pre_reset_clean_obs",
            pre_reset_clean_obs,
            (self.num_envs, self.obs_dim),
        )
        self._validate_tensor(
            "successor_height_m",
            successor_height_m,
            (self.num_envs, 1),
        )
        if terminated.dtype != torch.bool or tuple(terminated.shape) != (
            self.num_envs,
        ):
            raise ValueError("terminated must be a bool vector of length num_envs")
        if timed_out.dtype != torch.bool or tuple(timed_out.shape) != (self.num_envs,):
            raise ValueError("timed_out must be a bool vector of length num_envs")
        if behavior_iteration < 0:
            raise ValueError("behavior_iteration must be nonnegative")

        dones = terminated | timed_out
        noise = sample_observation_noise(
            self.obs_names,
            self.obs_dims,
            self.noise,
            self.scales,
            self.num_envs,
            self.device,
            self._generator,
        )
        # Ended transitions keep the pre-reset successor. Non-ended transitions
        # reuse the policy observation that the runner already collected.
        next_actor_obs = torch.where(
            dones.unsqueeze(1),
            pre_reset_clean_obs + noise,
            post_reset_obs,
        )

        episode_ids = self._episode_ids.clone()
        step_ids = self._step_ids.clone()
        dones_cpu = dones.detach().to("cpu")
        self._episode_ids = torch.where(
            dones_cpu, self._episode_ids + 1, self._episode_ids
        )
        self._step_ids = torch.where(
            dones_cpu, torch.zeros_like(self._step_ids), self._step_ids + 1
        )

        values = {
            "actor_obs": actor_obs,
            "action": action,
            "next_actor_obs": next_actor_obs,
            "successor_height_m": successor_height_m,
            "terminated": terminated,
            "timed_out": timed_out,
            "episode_id": episode_ids,
            "step_id": step_ids,
            "env_id": self._env_ids,
            "behavior_iteration": torch.full(
                (self.num_envs,),
                int(behavior_iteration),
                dtype=torch.int64,
            ),
        }
        self._write(values)

    def _write(self, values: dict[str, torch.Tensor]) -> None:
        batch_size = values["actor_obs"].shape[0]
        if batch_size != self.num_envs:
            raise ValueError("append batch must match num_envs")
        start = self.write_index
        first = min(batch_size, self.capacity - start)
        for name in TRANSITION_FIELDS:
            value = (
                values[name].detach().to(device="cpu", dtype=self._buffers[name].dtype)
            )
            self._buffers[name][start : start + first] = value[:first]
            if first < batch_size:
                self._buffers[name][: batch_size - first] = value[first:]
        self.write_index = (start + batch_size) % self.capacity
        self.total_appended += batch_size
        self.valid_count = min(self.valid_count + batch_size, self.capacity)

    def _chronological_indices(self) -> torch.Tensor:
        if self.valid_count < self.capacity:
            return torch.arange(self.valid_count)
        if self.write_index == 0:
            return torch.arange(self.capacity)
        return torch.cat(
            (
                torch.arange(self.write_index, self.capacity),
                torch.arange(0, self.write_index),
            )
        )

    def _manifest(self, iteration: int) -> dict:
        return {
            "format": REPLAY_FORMAT,
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_timestamp(),
            "snapshot_id": f"replay_iter_{iteration:07d}",
            "run_id": self.run_id,
            "log_dir": self.log_dir,
            "iteration": int(iteration),
            "capacity": self.capacity,
            "valid_count": self.valid_count,
            "total_appended": self.total_appended,
            "num_envs": self.num_envs,
            "obs_names": list(self.obs_names),
            "obs_dims": list(self.obs_dims),
            "obs_dim": self.obs_dim,
            "action_names": list(self.action_names),
            "action_dims": list(self.action_dims),
            "action_dim": self.action_dim,
            "noise": self.noise,
            "scales": self.scales,
            "actor_cfg": self.actor_cfg,
            "env_cfg": self.env_cfg,
            "config_fingerprint": replay_config_fingerprint(self.env_cfg),
            "runner_cfg": self.runner_cfg,
            "task_name": self.task_name,
            "backend": self.backend,
            "device": self.device,
            "seed": self.seed,
            "source_revision": self.source_revision,
            "control_dt": self.control_dt,
            "n_policy_steps": self.n_policy_steps,
            "episode_length_s": self.episode_length_s,
            "reward_timing": REWARD_TIMING,
            "observation_noise_convention": "recorded_policy_observation",
            "action_convention": ACTION_CONVENTION,
        }

    def snapshot(self, iteration: int) -> Path | None:
        """Publish the current ring contents atomically; skip if unchanged."""
        if self.valid_count == 0:
            raise ValueError("cannot snapshot an empty replay buffer")
        if self.total_appended == self.last_snapshot_total:
            return None
        path = self.output_dir / f"replay_iter_{iteration:07d}"
        if path.exists():
            raise FileExistsError(f"snapshot already exists: {path}")
        temporary = path.with_name(path.name + ".tmp")
        if temporary.exists():
            raise FileExistsError(f"temporary snapshot already exists: {temporary}")

        indices = self._chronological_indices()
        data = {
            name: self._buffers[name][indices].clone() for name in TRANSITION_FIELDS
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        temporary.mkdir()
        torch.save(data, temporary / "data.pt")
        manifest = self._manifest(iteration)
        (temporary / "manifest.json").write_text(
            json.dumps(jsonable(manifest), indent=2, sort_keys=True) + "\n"
        )
        temporary.rename(path)
        self.last_snapshot_total = self.total_appended
        return path
