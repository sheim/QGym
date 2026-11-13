"""Selectable raw-state feature registry for replay recording.

The core transition fields in :data:`learning.offline.contracts.TRANSITION_FIELDS`
are always recorded. Everything here is opt-in through ``--replay_fields`` so
one recorder can produce a narrow actor-observation dataset or a wide snapshot
of every simulator signal, without changing the core contract.

Two time points are stored for each feature:

* the plain name (``dof_pos``) is read from the environment *before* the policy
  action is applied, so it matches the observation that chose that action;
* the ``next_`` name (``next_dof_pos``) is read after the action and *before*
  the episode reset, so it is the true successor state, never the reset state.

``reward_terms`` is the only non-tensor feature: it is the runner's weighted
per-term reward vector for the completed transition, in the order recorded by
``reward_term_names`` in the snapshot manifest.
"""

from __future__ import annotations

from collections.abc import Sequence

# Recorded feature name -> environment attribute read before the action.
STEP_FEATURE_ATTRS: dict[str, str] = {
    "root_states": "root_states",
    "base_lin_vel": "base_lin_vel",
    "base_ang_vel": "base_ang_vel",
    "projected_gravity": "projected_gravity",
    "base_height": "base_height",
    "dof_pos": "dof_pos",
    "dof_vel": "dof_vel",
    "dof_pos_obs": "dof_pos_obs",
    "dof_pos_target": "dof_pos_target",
    "dof_pos_history": "dof_pos_history",
    "dof_vel_target": "dof_vel_target",
    "torques": "torques",
    "tau_ff": "tau_ff",
    "p_gains": "p_gains",
    "d_gains": "d_gains",
    "contact_forces": "contact_forces",
    "rigid_body_states": "_rigid_body_state",
    "commands": "commands",
    "phase": "phase",
    "phase_obs": "phase_obs",
    "phase_frequency": "phase_frequency",
    "gait_reference": "gait_reference",
    "episode_length": "episode_length_buf",
}

# Recorded feature name -> environment attribute read after the action and
# before the reset. ``next_base_height`` is omitted: the core field
# ``successor_height_m`` already carries the same measurement.
NEXT_FEATURE_ATTRS: dict[str, str] = {
    "next_root_states": "root_states",
    "next_base_lin_vel": "base_lin_vel",
    "next_base_ang_vel": "base_ang_vel",
    "next_projected_gravity": "projected_gravity",
    "next_dof_pos": "dof_pos",
    "next_dof_vel": "dof_vel",
    "next_dof_pos_obs": "dof_pos_obs",
    "next_dof_pos_target": "dof_pos_target",
    "next_torques": "torques",
    "next_contact_forces": "contact_forces",
    "next_rigid_body_states": "_rigid_body_state",
    "next_commands": "commands",
    "next_phase": "phase",
    "next_phase_frequency": "phase_frequency",
    "next_gait_reference": "gait_reference",
    "next_episode_length": "episode_length_buf",
}

# Observation and reward features are built by the runner, not by getattr().
REWARD = "reward_terms"
RUNNER_FEATURES: tuple[str, ...] = (
    "critic_obs",
    "next_critic_obs",
    REWARD,
)

# Group order defines the canonical field order inside a snapshot manifest.
REPLAY_FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "critic": ("critic_obs", "next_critic_obs"),
    "root": ("root_states", "next_root_states"),
    "base": (
        "base_lin_vel",
        "base_ang_vel",
        "projected_gravity",
        "base_height",
        "next_base_lin_vel",
        "next_base_ang_vel",
        "next_projected_gravity",
    ),
    "joints": (
        "dof_pos",
        "dof_vel",
        "dof_pos_obs",
        "dof_pos_target",
        "dof_pos_history",
        "dof_vel_target",
        "torques",
        "tau_ff",
        "p_gains",
        "d_gains",
        "next_dof_pos",
        "next_dof_vel",
        "next_dof_pos_obs",
        "next_dof_pos_target",
        "next_torques",
    ),
    "contacts": ("contact_forces", "next_contact_forces"),
    "bodies": ("rigid_body_states", "next_rigid_body_states"),
    "commands": ("commands", "next_commands"),
    "gait": (
        "phase",
        "phase_frequency",
        "phase_obs",
        "gait_reference",
        "next_phase",
        "next_phase_frequency",
        "next_gait_reference",
    ),
    "rewards": ("reward_terms",),
    "episode": ("episode_length", "next_episode_length"),
}

ALL_REPLAY_FIELDS: tuple[str, ...] = tuple(
    name for group in REPLAY_FEATURE_GROUPS.values() for name in group
)

STEP_FEATURE_NAMES: tuple[str, ...] = tuple(STEP_FEATURE_ATTRS)
NEXT_FEATURE_NAMES: tuple[str, ...] = tuple(NEXT_FEATURE_ATTRS)


def _unknown_tokens_message(tokens: Sequence[str]) -> str:
    choices = ", ".join(sorted(REPLAY_FEATURE_GROUPS))
    return (
        f"unknown replay field selection {list(tokens)!r}; use 'all', group names "
        f"({choices}), or field names ({', '.join(ALL_REPLAY_FIELDS)})"
    )


def resolve_replay_fields(selection: str | Sequence[str] | None) -> tuple[str, ...]:
    """Expand a selection of groups/fields into ordered unique field names.

    ``""``, ``"none"``, ``None`` and an empty sequence select no extra fields.
    ``"all"`` selects every registered feature. Unknown tokens fail instead of
    being dropped, so a typo cannot silently produce a narrower dataset.
    """
    if selection is None:
        return ()
    if isinstance(selection, str):
        tokens = [token.strip() for token in selection.split(",") if token.strip()]
    else:
        tokens = [str(token).strip() for token in selection if str(token).strip()]
    if not tokens:
        return ()

    lowered = [token.lower() for token in tokens]
    if "all" in lowered:
        if any(token != "all" for token in lowered):
            raise ValueError("'all' cannot be combined with other replay fields")
        return ALL_REPLAY_FIELDS

    selected: set[str] = set()
    unknown: list[str] = []
    for token in lowered:
        if token in ("none", "core"):
            continue
        if token in REPLAY_FEATURE_GROUPS:
            selected.update(REPLAY_FEATURE_GROUPS[token])
        elif token in STEP_FEATURE_ATTRS or token in NEXT_FEATURE_ATTRS:
            selected.add(token)
        elif token in RUNNER_FEATURES:
            selected.add(token)
        else:
            unknown.append(token)
    if unknown:
        raise ValueError(_unknown_tokens_message(unknown))
    return tuple(name for name in ALL_REPLAY_FIELDS if name in selected)


def step_feature_names(fields: Sequence[str]) -> tuple[str, ...]:
    """Features read before the action (everything except successor features)."""
    return tuple(
        name for name in fields if not name.startswith("next_") and name != REWARD
    )


def next_feature_names(fields: Sequence[str]) -> tuple[str, ...]:
    """Features read after the action and before the reset."""
    return tuple(name for name in fields if name.startswith("next_") or name == REWARD)
