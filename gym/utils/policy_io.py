"""Metadata and analysis helpers for recorded policy observations and actions."""

import numpy as np


_NAMED_COMPONENTS = {
    "base_lin_vel": ("x", "y", "z"),
    "base_ang_vel": ("x", "y", "z"),
    "projected_gravity": ("x", "y", "z"),
    "commands": ("vx", "vy", "yaw"),
    "phase_obs": ("sin", "cos"),
}


def state_component_names(env, state_names):
    """Expand configured state fields into flattened, human-readable names."""
    names = []
    actuated_names = list(getattr(env, "actuated_dof_names", ()))
    dof_names = list(getattr(env, "dof_names", ()))
    for state_name in state_names:
        state = getattr(env, state_name)
        if state.ndim != 2:
            raise ValueError(
                f"state {state_name!r} must be [num_envs, width], got {state.shape}"
            )
        width = state.shape[1]
        named_components = _NAMED_COMPONENTS.get(state_name)
        if named_components is not None and len(named_components) == width:
            components = named_components
        elif "dof" in state_name or state_name.startswith("tau_"):
            if width == len(dof_names):
                joint_names = dof_names
            elif width == len(actuated_names):
                joint_names = actuated_names
            elif (
                "history" in state_name
                and actuated_names
                and width % len(actuated_names) == 0
            ):
                joint_names = actuated_names
            else:
                joint_names = []
            if joint_names:
                components = tuple(
                    (
                        joint_name
                        if width == len(joint_names)
                        else f"history_{slot}.{joint_name}"
                    )
                    for slot in range(width // len(joint_names))
                    for joint_name in joint_names
                )
            else:
                components = tuple(str(index) for index in range(width))
        elif width == 1:
            components = ("",)
        else:
            components = tuple(str(index) for index in range(width))
        names.extend(
            state_name if not component else f"{state_name}.{component}"
            for component in components
        )
    return names


def state_component_scales(env, state_names, scales):
    """Expand task scaling values into the same flattened order as states."""
    component_scales = []
    for state_name in state_names:
        state = getattr(env, state_name)
        if state.ndim != 2:
            raise ValueError(
                f"state {state_name!r} must be [num_envs, width], got {state.shape}"
            )
        component_scales.extend(_expand_field_scale(state_name, state.shape[1], scales))
    return component_scales


def component_scales_from_names(state_names, component_names, scales):
    """Recover flattened scales from artifact field and component names."""
    component_names = [str(name) for name in component_names]
    offset = 0
    component_scales = []
    for state_name in (str(name) for name in state_names):
        width = 0
        while offset + width < len(component_names):
            component_name = component_names[offset + width]
            if component_name != state_name and not component_name.startswith(
                f"{state_name}."
            ):
                break
            width += 1
        if width == 0:
            raise ValueError(f"no components found for state field {state_name!r}")
        component_scales.extend(_expand_field_scale(state_name, width, scales))
        offset += width
    if offset != len(component_names):
        raise ValueError(
            f"{len(component_names) - offset} component names do not belong to "
            "the configured state fields"
        )
    return component_scales


def _expand_field_scale(state_name, width, scales):
    scale = np.asarray(scales.get(state_name, 1.0), dtype=np.float32)
    if scale.ndim == 0:
        return [float(scale)] * width
    scale = scale.reshape(-1)
    if len(scale) != width:
        raise ValueError(
            f"scale for state {state_name!r} has {len(scale)} components, "
            f"expected {width}"
        )
    return scale.tolist()


def policy_io_in_space(values, scale, source_space, target_space):
    """Convert logged policy I/O between task-normalized and task units."""
    valid_spaces = {"normalized", "unnormalized"}
    if source_space not in valid_spaces:
        raise ValueError(f"unknown source space {source_space!r}")
    if target_space not in valid_spaces:
        raise ValueError(f"unknown target space {target_space!r}")

    values = np.asarray(values)
    scale = np.asarray(scale, dtype=values.dtype)
    if np.any(scale == 0):
        raise ValueError("policy I/O scale values must be nonzero")
    if source_space == target_space:
        return values
    if target_space == "unnormalized":
        return values * scale
    return values / scale


def first_episode_mask(terminated):
    """Select samples strictly before each environment's first termination."""
    terminated = np.asarray(terminated, dtype=bool)
    if terminated.ndim != 2:
        raise ValueError(f"terminated must be [time, env], got {terminated.shape}")
    return np.cumsum(terminated, axis=0) == 0


def phase_binned_stats(phase, values, valid, num_bins=32):
    """Calculate mean and standard deviation of samples in oscillator-phase bins."""
    phase = np.mod(np.asarray(phase), 2.0 * np.pi)
    values = np.asarray(values)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(phase) & np.isfinite(values)
    if phase.shape != values.shape or phase.shape != valid.shape:
        raise ValueError(
            "phase, values, and valid must have the same [time, env] shape"
        )
    edges = np.linspace(0.0, 2.0 * np.pi, num_bins + 1)
    means = np.full(num_bins, np.nan, dtype=np.float32)
    stds = np.full(num_bins, np.nan, dtype=np.float32)
    counts = np.zeros(num_bins, dtype=np.int64)
    indices = np.clip(np.digitize(phase, edges) - 1, 0, num_bins - 1)
    for bin_index in range(num_bins):
        selected = valid & (indices == bin_index)
        counts[bin_index] = np.count_nonzero(selected)
        if counts[bin_index]:
            means[bin_index] = np.mean(values[selected])
            stds[bin_index] = np.std(values[selected])
    return 0.5 * (edges[:-1] + edges[1:]), means, stds, counts
