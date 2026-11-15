"""Runtime helpers for single-environment height-Q intervened playback.

The controller executes one of two per-decision interventions over the frozen
nominal policy:

* ``maxq``: eight stochastic policy draws, execute the highest-Q draw;
* ``rejection``: eight stochastic draws in fixed order, execute the first draw
  with ``Q >= threshold``, otherwise execute the highest-Q draw.

The policy mean is not a candidate in either mode. Selection is delegated to
the verified helpers in :mod:`learning.offline.persistent_intervention`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

from gym.utils.helpers import class_to_dict
from learning.offline.contracts import (
    ACTION_CONVENTION,
    replay_config_fingerprint,
    sha256_file,
)
from learning.offline.persistent_intervention import (
    sample_candidates,
    sample_rejection_candidates,
    select_action,
    select_rejection_action,
)

PLAYBACK_MODES = ("maxq", "rejection")
AUTO_RESET_MODES = ("off", "termination", "timeout", "both")

# Actor observations only. Height, base linear velocity, and contact/root state
# are critic-side or simulator-only measurements and would make Q privileged.
PRIVILEGED_Q_INPUTS = frozenset(
    {
        "base_height",
        "base_lin_vel",
        "bodyweight",
        "contact_forces",
        "contacts",
        "rigid_body_states",
        "root_states",
    }
)


@dataclass(frozen=True)
class PlaybackDecision:
    """One controller decision plus diagnostics for playback logging."""

    action: torch.Tensor
    indices: torch.Tensor
    fallback: torch.Tensor
    selected_q: torch.Tensor | None
    gain: torch.Tensor
    attempts: torch.Tensor


class HeightQPlaybackController:
    """Sample-eight height-Q controller for one environment."""

    def __init__(
        self,
        policy,
        critic,
        *,
        mode: str,
        threshold: float | None = None,
        generator: torch.Generator,
    ) -> None:
        if mode not in PLAYBACK_MODES:
            raise ValueError(f"unknown playback mode {mode!r}")
        if mode == "rejection":
            if threshold is None:
                raise ValueError("rejection mode requires a threshold")
            threshold = float(threshold)
            if not math.isfinite(threshold):
                raise ValueError("rejection threshold must be finite")
        elif threshold is not None:
            raise ValueError("maxq mode does not accept a rejection threshold")
        if generator.device != policy.device:
            raise ValueError("action generator must be on the policy device")
        self.policy = policy
        self.critic = critic
        self.mode = mode
        self.threshold = threshold
        self.generator = generator

    @torch.no_grad()
    def act(self, observation: torch.Tensor) -> PlaybackDecision:
        if observation.ndim != 2 or observation.shape[0] != 1:
            raise ValueError("playback controller requires one observation row")
        if self.mode == "maxq":
            candidates = sample_candidates(self.policy, observation, self.generator)
            action, indices, gain = select_action(observation, candidates, self.critic)
            fallback = torch.zeros_like(indices, dtype=torch.bool)
            selected_q = None
            attempts = torch.full_like(indices, 8)
        else:
            candidates, mean = sample_rejection_candidates(
                self.policy, observation, self.generator
            )
            result = select_rejection_action(
                observation,
                candidates,
                mean,
                self.critic,
                self.threshold,
            )
            action = result["action"]
            indices = result["indices"]
            fallback = result["fallback"]
            selected_q = result["selected_q"]
            gain = result["gain"]
            attempts = result["attempts"]
        if not torch.isfinite(action).all():
            raise RuntimeError("policy selected a non-finite action")
        return PlaybackDecision(
            action=action,
            indices=indices,
            fallback=fallback,
            selected_q=selected_q,
            gain=gain,
            attempts=attempts,
        )


def _values_match(left, right) -> bool:
    """Compare config scaling values to JSON round-tripped float32 values."""
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
            return False
        return len(left) == len(right) and all(
            _values_match(a, b) for a, b in zip(left, right, strict=True)
        )
    return math.isclose(
        float(left),
        float(right),
        rel_tol=1e-6,
        abs_tol=1e-8,
    )


def _scales_match(config_scales: dict, recorded_scales: dict) -> bool:
    if set(config_scales) != set(recorded_scales):
        return False
    return all(
        _values_match(config_scales[name], recorded_scales[name])
        for name in config_scales
    )


def validate_playback_contract(
    env_cfg,
    train_cfg,
    artifact,
    spec,
    checkpoint: str | Path,
) -> str:
    """Fail fast unless the task, frozen actor, and Q artifact agree.

    The config fingerprint is taken before any playback-only override (num_envs,
    viewer UI, pushes, seed). ``spec`` is the artifact policy contract; the
    checkpoint hash is checked against that contract, not against the path.
    Returns the recorded environment fingerprint for the startup summary.
    """
    obs_names = tuple(spec.obs_names)
    obs_dims = tuple(int(dim) for dim in spec.obs_dims)
    action_names = tuple(spec.action_names)

    privileged = PRIVILEGED_Q_INPUTS.intersection(obs_names)
    if privileged:
        raise ValueError(
            "Q observation contract contains privileged inputs: "
            + ", ".join(sorted(privileged))
        )

    if sum(obs_dims) != int(artifact.obs_dim):
        raise ValueError("Q observation width does not match policy observation dims")
    if int(spec.action_dim) != int(artifact.action_dim):
        raise ValueError("Q action width does not match policy action dim")

    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise ValueError(f"policy checkpoint does not exist: {checkpoint}")
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != spec.checkpoint_sha256:
        raise ValueError("policy checkpoint hash does not match the Q artifact")
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "actor_state_dict" not in loaded:
        raise ValueError("policy checkpoint does not contain actor_state_dict")

    cfg_dict = class_to_dict(env_cfg)
    recorded_fingerprint = artifact.manifest["dataset"]["config_fingerprint"]
    if not recorded_fingerprint:
        raise ValueError("Q artifact does not record a config fingerprint")
    actual_fingerprint = replay_config_fingerprint(cfg_dict)
    if actual_fingerprint != recorded_fingerprint:
        raise ValueError(
            "environment configuration fingerprint does not match the Q artifact"
        )

    if tuple(train_cfg.actor.obs) != obs_names:
        raise ValueError("policy observation names do not match the task actor config")
    if tuple(train_cfg.actor.actions) != action_names:
        raise ValueError("policy action names do not match the task actor config")
    if train_cfg.actor.normalize_obs is not False:
        raise ValueError("playback requires actor normalize_obs == False")
    if spec.actor_cfg["normalize_obs"] is not False:
        raise ValueError("artifact actor_cfg requires normalize_obs == False")
    if spec.actor_cfg["smooth_exploration"] is not False:
        raise ValueError("artifact actor_cfg requires smooth_exploration == False")
    if spec.smooth_exploration is not False:
        raise ValueError("playback supports the standard stochastic actor only")
    if int(spec.n_policy_steps) != 1:
        raise ValueError("playback requires n_policy_steps == 1")
    if spec.action_convention != ACTION_CONVENTION:
        raise ValueError("artifact action convention is not brg_policy_units")
    if abs(float(spec.control_dt) - float(env_cfg.control.ctrl_dt)) > 1e-12:
        raise ValueError("artifact control dt does not match the task")

    task_noise = class_to_dict(train_cfg.actor.noise)
    if spec.noise != task_noise:
        raise ValueError("artifact observation noise does not match the task")
    if spec.actor_cfg["noise"] != task_noise:
        raise ValueError("artifact actor_cfg noise does not match the task")

    task_scales = class_to_dict(env_cfg.scaling)
    if not _scales_match(task_scales, spec.scales):
        raise ValueError("artifact observation/action scales do not match the task")

    return recorded_fingerprint


def validate_env_widths(env, spec) -> None:
    """Check live task tensor widths after environment construction."""
    for name, expected in zip(spec.obs_names, spec.obs_dims, strict=True):
        actual = env.get_states([name]).shape[1]
        if actual != expected:
            raise ValueError(
                f"task observation {name!r} has width {actual}, expected {expected}"
            )
    action_width = env.get_states(list(spec.action_names)).shape[1]
    if action_width != int(spec.action_dim):
        raise ValueError(
            f"task action width is {action_width}, expected {spec.action_dim}"
        )


def disable_command_resampling(env):
    """Freeze task command resampling without changing shared config defaults.

    Returns the original bound method for :func:`restore_command_resampling`.
    This is an instance-local override, so playback can hold user commands
    indefinitely while normal training and ``scripts/play.py`` keep resampling.
    """
    original = env._resample_commands

    def _hold_user_commands(_mask):
        return None

    env._resample_commands = _hold_user_commands
    return original


def restore_command_resampling(env, original) -> None:
    env._resample_commands = original


def auto_reset_mask(env, mode: str) -> torch.Tensor | None:
    if mode not in AUTO_RESET_MODES:
        raise ValueError(f"unknown auto-reset mode {mode!r}")
    if mode == "off":
        return None
    if mode == "termination":
        return env.terminated.clone()
    if mode == "timeout":
        return env.timed_out.clone()
    return env.terminated | env.timed_out


def apply_auto_reset(env, mode: str) -> int:
    """Reset flagged environments and zero their commands.

    Returns the number of reset environments. A held command is preserved when
    no environment is flagged; every actual reset path ends at zero commands.
    """
    mask = auto_reset_mask(env, mode)
    if mask is None or not bool(mask.any()):
        return 0
    count = int(mask.sum().item())
    env._reset_idx(mask)
    env.commands.zero_()
    return count
