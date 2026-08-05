"""Post-process legged-policy trajectories into height and gait metrics."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


IDEAL_GAIT_RPD = {
    "trot": np.asarray([np.pi, np.pi, 0.0]),
    "pace": np.asarray([np.pi, 0.0, np.pi]),
    "bound": np.asarray([0.0, np.pi, np.pi]),
    "pronk": np.asarray([0.0, 0.0, 0.0]),
}


def urdf_total_mass(path):
    """Return nominal total link mass from a URDF."""
    root = ET.parse(Path(path)).getroot()
    masses = [
        float(mass.attrib["value"]) for mass in root.findall("./link/inertial/mass")
    ]
    if not masses:
        raise ValueError(f"no link masses found in {path}")
    return sum(masses)


def _trajectory_end(alive, env_index):
    dead = np.flatnonzero(~alive[:, env_index])
    return int(dead[0]) if len(dead) else alive.shape[0]


def _circular_difference(angle, reference):
    return np.angle(np.exp(1j * (angle - reference)))


def analyze_base_height(
    base_height_history,
    alive_history,
    sample_rate_hz,
    settle_steps,
):
    """Measure steadiness without requiring a particular absolute height."""
    height = np.asarray(base_height_history)
    alive = np.asarray(alive_history)
    num_envs = height.shape[1]
    metrics = {
        name: np.full(num_envs, np.nan, dtype=np.float32)
        for name in (
            "base_height_mean",
            "base_height_std",
            "base_height_range",
            "base_height_drift_abs",
        )
    }
    for env_index in range(num_envs):
        end = _trajectory_end(alive, env_index)
        values = height[settle_steps:end, env_index]
        if len(values) < 2:
            continue
        metrics["base_height_mean"][env_index] = np.mean(values)
        metrics["base_height_std"][env_index] = np.std(values)
        metrics["base_height_range"][env_index] = np.ptp(values)
        duration = (len(values) - 1) / sample_rate_hz
        metrics["base_height_drift_abs"][env_index] = (
            abs(float(values[-1] - values[0])) / duration
        )
    return metrics


def _debounced_touchdowns(contact, minimum_separation_steps):
    rising = np.flatnonzero(contact[1:] & ~contact[:-1]) + 1
    retained = []
    for index in rising:
        if not retained or index - retained[-1] >= minimum_separation_steps:
            retained.append(int(index))
    return np.asarray(retained, dtype=np.int64)


def _gait_features(
    force_norm,
    sample_rate_hz,
    contact_threshold_n,
    gait_frequency_hz,
):
    contact = force_norm > contact_threshold_n
    minimum_separation = max(1, int(round(0.5 * sample_rate_hz / gait_frequency_hz)))
    events = [
        _debounced_touchdowns(contact[:, foot], minimum_separation)
        for foot in range(contact.shape[1])
    ]
    rf_events = events[0]
    nominal_period = sample_rate_hz / gait_frequency_hz
    cycle_bounds = [
        (start, stop)
        for start, stop in zip(rf_events[:-1], rf_events[1:])
        if 0.5 * nominal_period <= stop - start <= 2.0 * nominal_period
    ]
    frequencies = np.asarray(
        [sample_rate_hz / (stop - start) for start, stop in cycle_bounds]
    )
    cycle_rpd = []
    for start, stop in cycle_bounds:
        phases = []
        for foot_events in events[1:]:
            candidates = foot_events[(foot_events >= start) & (foot_events < stop)]
            if not len(candidates):
                break
            phases.append(
                2.0 * np.pi * float(candidates[0] - start) / float(stop - start)
            )
        if len(phases) == 3:
            cycle_rpd.append(phases)

    result = {
        "gait_cycle_frequency_mean": (
            float(np.mean(frequencies)) if len(frequencies) else np.nan
        ),
        "gait_cycle_frequency_std": (
            float(np.std(frequencies)) if len(frequencies) else np.nan
        ),
        "gait_complete_cycle_fraction": (
            float(len(cycle_rpd) / len(cycle_bounds)) if cycle_bounds else np.nan
        ),
        "gait_rpd_trot_error": np.nan,
        "gait_rpd_cycle_consistency": np.nan,
        "gait_trot_classified": np.nan,
    }
    mean_rpd = np.full(3, np.nan)
    gait_class = "transition"
    if cycle_rpd:
        cycle_rpd = np.asarray(cycle_rpd)
        mean_rpd = np.mod(
            np.angle(np.mean(np.exp(1j * cycle_rpd), axis=0)),
            2.0 * np.pi,
        )
        trot_difference = _circular_difference(
            mean_rpd,
            IDEAL_GAIT_RPD["trot"],
        )
        result["gait_rpd_trot_error"] = float(
            np.sqrt(np.mean(np.square(trot_difference)))
        )
        deviations = _circular_difference(cycle_rpd, mean_rpd)
        result["gait_rpd_cycle_consistency"] = float(
            np.sqrt(np.mean(np.square(deviations)))
        )
        distances = {
            name: float(np.linalg.vector_norm(_circular_difference(mean_rpd, ideal)))
            for name, ideal in IDEAL_GAIT_RPD.items()
        }
        gait_class = min(distances, key=distances.get)
        if distances[gait_class] > 2.0:
            gait_class = "transition"
        result["gait_trot_classified"] = float(gait_class == "trot")
    return result, mean_rpd, gait_class


def analyze_gait_and_grf(
    foot_force_norm_history,
    foot_force_z_history,
    alive_history,
    moving,
    sample_rate_hz,
    settle_steps,
    contact_threshold_n,
    gait_frequency_hz,
    robot_mass_kg,
):
    """Calculate touchdown RPD and Fig.-3-style normalized leg loading."""
    force_norm = np.asarray(foot_force_norm_history)
    force_z = np.maximum(np.asarray(foot_force_z_history), 0.0)
    alive = np.asarray(alive_history)
    moving = np.asarray(moving, dtype=bool)
    num_envs = force_norm.shape[1]
    metric_names = (
        "gait_cycle_frequency_mean",
        "gait_cycle_frequency_std",
        "gait_complete_cycle_fraction",
        "gait_rpd_trot_error",
        "gait_rpd_cycle_consistency",
        "gait_trot_classified",
        "grf_balance_std",
        "grf_balance_cv",
        "grf_total_body_weight",
        "grf_min_leg_mean",
        "grf_max_leg_mean",
    )
    metrics = {
        name: np.full(num_envs, np.nan, dtype=np.float32) for name in metric_names
    }
    rpd_mean = np.full((num_envs, 3), np.nan, dtype=np.float32)
    gait_class = np.full(num_envs, "stationary", dtype="<U10")
    grf_by_foot = np.full((num_envs, force_norm.shape[2]), np.nan, dtype=np.float32)
    body_weight = robot_mass_kg * 9.81

    for env_index in range(num_envs):
        end = _trajectory_end(alive, env_index)
        if end - settle_steps < 2:
            continue
        normalized_grf = (
            np.mean(
                force_z[settle_steps:end, env_index],
                axis=0,
            )
            / body_weight
        )
        grf_by_foot[env_index] = normalized_grf
        metrics["grf_balance_std"][env_index] = np.std(normalized_grf)
        metrics["grf_balance_cv"][env_index] = np.std(normalized_grf) / max(
            float(np.mean(normalized_grf)),
            np.finfo(float).eps,
        )
        metrics["grf_total_body_weight"][env_index] = np.sum(normalized_grf)
        metrics["grf_min_leg_mean"][env_index] = np.min(normalized_grf)
        metrics["grf_max_leg_mean"][env_index] = np.max(normalized_grf)

        if not moving[env_index]:
            continue
        features, mean_rpd, classification = _gait_features(
            force_norm[settle_steps:end, env_index],
            sample_rate_hz,
            contact_threshold_n,
            gait_frequency_hz,
        )
        for name, value in features.items():
            metrics[name][env_index] = value
        rpd_mean[env_index] = mean_rpd
        gait_class[env_index] = classification

    artifacts = {
        "gait_rpd_mean": rpd_mean,
        "gait_class": gait_class,
        "grf_body_weight_by_foot": grf_by_foot,
    }
    return metrics, artifacts
