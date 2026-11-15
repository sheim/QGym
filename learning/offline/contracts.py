"""Versioned contracts shared by offline replay, relabeling and Q artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .replay_fields import resolve_replay_fields


SCHEMA_VERSION = 1
# Legacy core-only replay snapshots keep this format identifier so existing
# snapshots stay loadable. Recordings that add selectable raw-state features use
# REPLAY_FORMAT_V2 and declare their exact field set in the manifest.
REPLAY_FORMAT = "brg-offline-replay-v1"
REPLAY_FORMAT_V2 = "brg-offline-replay-v2"
HEIGHT_Q_FORMAT = "brg-height-q-v1"
ANGULAR_VEL_Q_FORMAT = "brg-angular-vel-q-v1"
CONTACT_Q_FORMAT = "brg-contact-q-v1"
RETURN_CONVENTION = "discounted_continuing_absorbing_failure"
ZERO_AFTER_TERMINAL_RETURN_CONVENTION = "discounted_episodic_zero_after_terminal"
REWARD_TIMING = "post_action_pre_reset"
TERMINAL_CONTINUATION = "absorbing_failure"
ZERO_AFTER_TERMINAL = "zero_after_terminal"
ACTION_CONVENTION = "brg_policy_units"

TRANSITION_FIELDS = (
    "actor_obs",
    "action",
    "next_actor_obs",
    "successor_height_m",
    "terminated",
    "timed_out",
    "episode_id",
    "step_id",
    "env_id",
    "behavior_iteration",
)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replay_config_fingerprint(env_cfg: dict) -> str:
    """Hash task dynamics/observation/action settings, excluding run-only fields."""
    selected = {
        "env": {
            key: value
            for key, value in env_cfg.get("env", {}).items()
            if key != "num_envs"
        },
        "control": env_cfg.get("control"),
        "asset": env_cfg.get("asset"),
        "init_state": env_cfg.get("init_state"),
        "domain_randomization": env_cfg.get("domain_randomization"),
        "terrain": env_cfg.get("terrain"),
        "commands": env_cfg.get("commands"),
        "push_robots": env_cfg.get("push_robots"),
        "scaling": env_cfg.get("scaling"),
        "reward_settings": env_cfg.get("reward_settings"),
        "mujoco": env_cfg.get("mujoco"),
    }
    payload = json.dumps(
        jsonable(selected),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def jsonable(value: Any) -> Any:
    """Convert configs and tensors into deterministic JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if callable(value):
        return None
    return repr(value)


def _check_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


@dataclass(frozen=True)
class RecorderConfig:
    enabled: bool = False
    capacity: int = 1_000_000
    snapshot_interval: int = 0
    output_dir: str = ""
    seed: int = 0
    task_name: str = ""
    backend: str = ""
    source_revision: str = ""
    # Extra raw-state features, already expanded from group names. Importing
    # the registry here would create a cycle only if the registry imported
    # contracts; it does not, so the names are validated in __post_init__.
    fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("replay capacity must be positive")
        if self.snapshot_interval < 0:
            raise ValueError("replay snapshot_interval must be nonnegative")
        if self.seed < 0:
            raise ValueError("replay seed must be nonnegative")
        if not isinstance(self.fields, tuple):
            raise ValueError("replay fields must be a tuple of field names")
        if self.enabled:
            if not self.task_name:
                raise ValueError("replay task_name is required when enabled")
            if not self.backend:
                raise ValueError("replay backend is required when enabled")
            if self.fields:
                if resolve_replay_fields(self.fields) != self.fields:
                    raise ValueError(
                        "replay fields must be canonical, ordered, and unique; "
                        "use resolve_replay_fields()"
                    )


@dataclass(frozen=True)
class HeightRewardSpec:
    reference_height_m: float
    bound_m: float
    inside_reward: float = 0.0
    outside_reward: float = -1.0
    timing: str = REWARD_TIMING
    terminal_continuation: str = TERMINAL_CONTINUATION

    def __post_init__(self) -> None:
        _check_finite("reference_height_m", self.reference_height_m)
        _check_finite("bound_m", self.bound_m)
        _check_finite("inside_reward", self.inside_reward)
        _check_finite("outside_reward", self.outside_reward)
        if self.bound_m < 0:
            raise ValueError("height bound_m must be nonnegative")
        if self.outside_reward >= self.inside_reward:
            raise ValueError("outside_reward must be strictly below inside_reward")
        if self.timing != REWARD_TIMING:
            raise ValueError(f"unsupported reward timing {self.timing!r}")
        if self.terminal_continuation != TERMINAL_CONTINUATION:
            raise ValueError(
                f"unsupported terminal continuation {self.terminal_continuation!r}"
            )

    @property
    def min_reward(self) -> float:
        return min(self.inside_reward, self.outside_reward)

    @property
    def max_reward(self) -> float:
        return max(self.inside_reward, self.outside_reward)

    def failure_value(self, gamma: float) -> float:
        if not 0.0 <= gamma < 1.0:
            raise ValueError("gamma must satisfy 0 <= gamma < 1")
        return self.outside_reward / (1.0 - gamma)


@dataclass(frozen=True)
class AngularVelocityRewardSpec:
    """Yaw angular-velocity tracking term of ``Go2Trot._reward_tracking_ang_vel``.

    The task computes ``exp(-((cmd_yaw - ang_vel_yaw) / error_scale)**2 /
    tracking_sigma)``: the yaw error is divided by a hard-coded 2.5 rad/s scale
    before ``_sqrdexp`` applies ``tracking_sigma``. Reproduce both constants; a
    single ``tracking_sigma`` alone would label a different reward. The reward is
    already positive and bounded in [0, 1], so true termination ends the episode
    with a zero return rather than an absorbing failure tail.
    """

    tracking_sigma: float
    error_scale: float = 2.5
    timing: str = REWARD_TIMING
    terminal_continuation: str = ZERO_AFTER_TERMINAL

    def __post_init__(self) -> None:
        _check_finite("tracking_sigma", self.tracking_sigma)
        _check_finite("error_scale", self.error_scale)
        if self.tracking_sigma <= 0:
            raise ValueError("tracking_sigma must be positive")
        if self.error_scale <= 0:
            raise ValueError("error_scale must be positive")
        if self.timing != REWARD_TIMING:
            raise ValueError(f"unsupported reward timing {self.timing!r}")
        if self.terminal_continuation != ZERO_AFTER_TERMINAL:
            raise ValueError(
                f"unsupported terminal continuation {self.terminal_continuation!r}"
            )

    @property
    def min_reward(self) -> float:
        return 0.0

    @property
    def max_reward(self) -> float:
        return 1.0

    def failure_value(self, gamma: float) -> float:
        """Zero: the terminal return ends at the last observed reward."""
        if not 0.0 <= gamma < 1.0:
            raise ValueError("gamma must satisfy 0 <= gamma < 1")
        return 0.0


@dataclass(frozen=True)
class ContactForceRewardSpec:
    """Per-foot vertical ground-force cap: -1 above it, 0 otherwise.

    The reward is a per-step indicator of a *peak-load* constraint: each foot
    has its own threshold in bodyweight units, and the label fires when the
    instantaneous post-action load of any foot exceeds it. Thresholds come from
    that foot's per-touchdown peak distribution under the recorded behaviour
    policy (``percentile``), so the label is relative to that policy and must be
    carried by the artifact rather than re-derived at eval time.

    The reward is bounded in [-1, 0] and the episode ends with a zero return, so
    true termination contributes no absorbing tail.
    """

    foot_names: tuple[str, ...]
    thresholds_bw: tuple[float, ...]
    percentile: float
    timing: str = REWARD_TIMING
    terminal_continuation: str = ZERO_AFTER_TERMINAL

    def __post_init__(self) -> None:
        if not self.foot_names:
            raise ValueError("at least one foot name is required")
        if len(set(self.foot_names)) != len(self.foot_names):
            raise ValueError("foot_names must be unique")
        if len(self.thresholds_bw) != len(self.foot_names):
            raise ValueError("thresholds_bw must match foot_names in length")
        for name, threshold in zip(self.foot_names, self.thresholds_bw, strict=True):
            _check_finite(f"threshold for {name}", threshold)
            if threshold <= 0:
                raise ValueError(f"threshold for {name} must be positive")
        _check_finite("percentile", self.percentile)
        if not 0.0 < self.percentile < 100.0:
            raise ValueError("percentile must be strictly between 0 and 100")
        if self.timing != REWARD_TIMING:
            raise ValueError(f"unsupported reward timing {self.timing!r}")
        if self.terminal_continuation != ZERO_AFTER_TERMINAL:
            raise ValueError(
                f"unsupported terminal continuation {self.terminal_continuation!r}"
            )

    @property
    def min_reward(self) -> float:
        return -1.0

    @property
    def max_reward(self) -> float:
        return 0.0

    def failure_value(self, gamma: float) -> float:
        """Zero: the return ends at the last observed reward."""
        if not 0.0 <= gamma < 1.0:
            raise ValueError("gamma must satisfy 0 <= gamma < 1")
        return 0.0


@dataclass(frozen=True)
class PolicySpec:
    checkpoint: str
    checkpoint_sha256: str
    actor_cfg: dict
    obs_names: tuple[str, ...]
    obs_dims: tuple[int, ...]
    action_names: tuple[str, ...]
    action_dim: int
    noise: dict
    scales: dict
    control_dt: float
    n_policy_steps: int
    episode_length_s: float
    smooth_exploration: bool
    action_convention: str = ACTION_CONVENTION

    def __post_init__(self) -> None:
        if not self.checkpoint:
            raise ValueError("policy checkpoint path is required")
        if not self.checkpoint_sha256:
            raise ValueError("policy checkpoint hash is required")
        if not self.actor_cfg:
            raise ValueError("actor_cfg is required")
        if not self.obs_names:
            raise ValueError("obs_names is required")
        if len(self.obs_names) != len(self.obs_dims):
            raise ValueError("obs_names and obs_dims must have equal length")
        if any(dim <= 0 for dim in self.obs_dims):
            raise ValueError("observation dimensions must be positive")
        if not self.action_names:
            raise ValueError("action_names is required")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        _check_finite("control_dt", self.control_dt)
        if self.control_dt <= 0:
            raise ValueError("control_dt must be positive")
        if self.n_policy_steps <= 0:
            raise ValueError("n_policy_steps must be positive")
        _check_finite("episode_length_s", self.episode_length_s)
        if self.episode_length_s <= 0:
            raise ValueError("episode_length_s must be positive")
        if self.smooth_exploration:
            raise ValueError("this pipeline supports the standard Actor only")
        if self.action_convention != ACTION_CONVENTION:
            raise ValueError(
                f"unsupported action convention {self.action_convention!r}"
            )


@dataclass(frozen=True)
class TrainingSpec:
    gamma: float
    batch_size: int = 256
    updates: int = 100_000
    learning_rate: float = 3e-4
    hidden_dims: tuple[int, ...] = (256, 256)
    activation: str = "relu"
    target_tau: float = 0.005
    seed: int = 0
    device: str = "cpu"
    target_action_samples: int = 1
    normalize_observations: bool = True

    def __post_init__(self) -> None:
        _check_finite("gamma", self.gamma)
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must satisfy 0 <= gamma < 1")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.updates <= 0:
            raise ValueError("updates must be positive")
        _check_finite("learning_rate", self.learning_rate)
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not self.hidden_dims or any(dim <= 0 for dim in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        if not self.activation:
            raise ValueError("activation is required")
        _check_finite("target_tau", self.target_tau)
        if not 0.0 <= self.target_tau <= 1.0:
            raise ValueError("target_tau must be in [0, 1]")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if not self.device:
            raise ValueError("device is required")
        if self.target_action_samples <= 0:
            raise ValueError("target_action_samples must be positive")


@dataclass(frozen=True)
class EvaluationSpec:
    trials: int
    repetitions: int
    horizon: int
    thresholds: tuple[float, ...]
    seed: int
    device: str

    def __post_init__(self) -> None:
        if self.trials <= 0:
            raise ValueError("trials must be positive")
        if self.repetitions <= 0:
            raise ValueError("repetitions must be positive")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if not self.device:
            raise ValueError("device is required")
        for threshold in self.thresholds:
            _check_finite("threshold", threshold)


def sample_observation_noise(
    obs_names: tuple[str, ...],
    obs_dims: tuple[int, ...],
    noise: dict,
    scales: dict,
    num_envs: int,
    device: str | torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """Match BaseRunner.get_noise while using a caller-owned generator."""
    if len(obs_names) != len(obs_dims):
        raise ValueError("obs_names and obs_dims must have equal length")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    total_dim = sum(obs_dims)
    noise_vec = torch.zeros(total_dim, device=device, dtype=torch.float32)
    offset = 0
    for name, dim in zip(obs_names, obs_dims, strict=True):
        if name in noise:
            value = noise[name]
            if isinstance(value, (list, tuple)):
                if len(value) != dim:
                    raise ValueError(
                        f"noise for {name!r} has {len(value)} values, expected {dim}"
                    )
                scale = torch.tensor(value, device=device, dtype=torch.float32)
            else:
                scale = torch.full(
                    (dim,), float(value), device=device, dtype=torch.float32
                )
            if name in scales:
                scale = scale / torch.as_tensor(
                    scales[name], device=device, dtype=torch.float32
                )
            noise_vec[offset : offset + dim] = scale
        offset += dim
    return noise_vec * torch.randn(
        num_envs,
        total_dim,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
