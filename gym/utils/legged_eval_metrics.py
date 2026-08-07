"""Hardware-oriented evaluation metrics for legged policies.

The training reward is deliberately not reused here.  These metrics retain
physical units so policies trained with different rewards or physics backends
can be compared without changing the evaluation objective.
"""

from dataclasses import asdict, dataclass

import numpy as np
import torch

from gym.utils.legged_signal_analysis import (
    analyze_base_height,
    analyze_foot_clearance_by_phase,
    analyze_gait_and_grf,
)


@dataclass(frozen=True)
class MetricDefinition:
    unit: str
    direction: str
    description: str


METRIC_DEFINITIONS = {
    "survival": MetricDefinition(
        "ratio", "higher", "Fraction of environments reaching the evaluation end."
    ),
    "episode_duration": MetricDefinition(
        "s", "higher", "Time before termination, capped at evaluation duration."
    ),
    "disturbance_failure": MetricDefinition(
        "ratio",
        "lower",
        "Post-impulse failure, conditional on surviving until perturbation.",
    ),
    "disturbance_pre_impulse_failure": MetricDefinition(
        "ratio", "lower", "Episode terminated before its scheduled perturbation."
    ),
    "disturbance_total_failure": MetricDefinition(
        "ratio", "lower", "Episode terminated before the evaluation ended."
    ),
    "tracking_vx_rmse": MetricDefinition(
        "m/s", "lower", "Forward command-tracking RMSE."
    ),
    "tracking_vy_rmse": MetricDefinition(
        "m/s", "lower", "Lateral command-tracking RMSE."
    ),
    "tracking_yaw_rmse": MetricDefinition(
        "rad/s", "lower", "Yaw-rate command-tracking RMSE."
    ),
    "base_height_error_rms": MetricDefinition(
        "m", "lower", "RMS base-height error relative to the task target."
    ),
    "base_height_mean": MetricDefinition(
        "m", "context", "Mean base height, without assuming the training target."
    ),
    "base_height_std": MetricDefinition(
        "m", "lower", "Base-height standard deviation after settling."
    ),
    "base_height_range": MetricDefinition(
        "m", "lower", "Peak-to-peak base-height range after settling."
    ),
    "base_height_drift_abs": MetricDefinition(
        "m/s", "lower", "Absolute endpoint base-height drift rate."
    ),
    "base_tilt_rms": MetricDefinition(
        "deg", "lower", "RMS angle between the base up axis and world up."
    ),
    "base_vertical_velocity_rms": MetricDefinition(
        "m/s", "lower", "RMS vertical base velocity."
    ),
    "base_vertical_acceleration_rms": MetricDefinition(
        "m/s^2", "lower", "Finite-difference RMS vertical base acceleration."
    ),
    "base_roll_pitch_rate_rms": MetricDefinition(
        "rad/s", "lower", "RMS roll/pitch angular velocity."
    ),
    "reference_joint_rmse": MetricDefinition(
        "rad", "lower", "Joint-position RMSE relative to the gait reference."
    ),
    "swing_clearance_p95_mean": MetricDefinition(
        "m",
        "context",
        "Mean across feet of 95th-percentile swing clearance above stance height.",
    ),
    "swing_clearance_p95_min": MetricDefinition(
        "m",
        "context",
        "Lowest per-foot 95th-percentile swing clearance above stance height.",
    ),
    "joint_velocity_rms": MetricDefinition(
        "rad/s", "lower", "RMS actuated-joint velocity."
    ),
    "joint_velocity_utilization_rms": MetricDefinition(
        "ratio", "lower", "RMS joint speed divided by the URDF velocity limit."
    ),
    "joint_velocity_limit_fraction": MetricDefinition(
        "ratio", "lower", "Fraction of joint samples at the URDF velocity limit."
    ),
    "joint_acceleration_rms": MetricDefinition(
        "rad/s^2", "lower", "Finite-difference RMS actuated-joint acceleration."
    ),
    "target_velocity_rms": MetricDefinition(
        "rad/s", "lower", "RMS rate of policy joint-position targets."
    ),
    "target_acceleration_rms": MetricDefinition(
        "rad/s^2",
        "lower",
        "RMS second derivative of policy joint-position targets.",
    ),
    "torque_rms": MetricDefinition("N m", "lower", "RMS actuator torque."),
    "torque_rate_rms": MetricDefinition(
        "N m/s", "lower", "Finite-difference RMS actuator torque rate."
    ),
    "torque_utilization_rms": MetricDefinition(
        "ratio", "lower", "RMS torque divided by the URDF effort limit."
    ),
    "torque_utilization_peak": MetricDefinition(
        "ratio", "lower", "Peak torque divided by the URDF effort limit."
    ),
    "torque_saturation_fraction": MetricDefinition(
        "ratio", "lower", "Fraction of actuator samples above 98% effort."
    ),
    "mechanical_power_mean": MetricDefinition(
        "W", "lower", "Mean sum of absolute actuator mechanical power."
    ),
    "joint_limit_margin_min": MetricDefinition(
        "ratio",
        "higher",
        "Minimum joint-limit margin as a fraction of total joint range.",
    ),
    "target_joint_limit_margin_min": MetricDefinition(
        "ratio",
        "higher",
        "Minimum commanded PD-target margin as a fraction of total joint range.",
    ),
    "foot_slip_speed_rms": MetricDefinition(
        "m/s", "lower", "RMS horizontal foot speed while in contact."
    ),
    "foot_contact_phase_match": MetricDefinition(
        "ratio", "higher", "Fraction of feet whose contact matches reference phase."
    ),
    "swing_contact_fraction": MetricDefinition(
        "ratio", "lower", "Fraction of reference-swing feet in contact."
    ),
    "stance_miss_fraction": MetricDefinition(
        "ratio", "lower", "Fraction of reference-stance feet without contact."
    ),
    "foot_force_peak": MetricDefinition(
        "N", "lower", "Peak measured net contact force on any foot."
    ),
    "unsafe_contact_fraction": MetricDefinition(
        "ratio",
        "lower",
        "Fraction of control steps with contact on a penalized body.",
    ),
    "gait_cycle_frequency_mean": MetricDefinition(
        "Hz", "context", "Mean RF touchdown-cycle frequency."
    ),
    "gait_cycle_frequency_std": MetricDefinition(
        "Hz", "lower", "Standard deviation of RF touchdown-cycle frequency."
    ),
    "gait_complete_cycle_fraction": MetricDefinition(
        "ratio", "higher", "RF cycles containing a touchdown from every leg."
    ),
    "gait_rpd_trot_error": MetricDefinition(
        "rad",
        "lower",
        "RMS circular error from trot RPD (LF, RH, LH) = (pi, pi, 0).",
    ),
    "gait_rpd_cycle_consistency": MetricDefinition(
        "rad", "lower", "Cycle-to-cycle RMS variation in touchdown RPD."
    ),
    "gait_trot_classified": MetricDefinition(
        "ratio", "higher", "Paper-style nearest-symmetric-gait trot indicator."
    ),
    "grf_balance_std": MetricDefinition(
        "body weight",
        "lower",
        "Across-leg standard deviation of mean vertical GRF.",
    ),
    "grf_balance_cv": MetricDefinition(
        "ratio", "lower", "Across-leg coefficient of variation of mean GRF."
    ),
    "grf_total_body_weight": MetricDefinition(
        "body weight", "closer_to_one", "Mean summed vertical foot GRF."
    ),
    "grf_min_leg_mean": MetricDefinition(
        "body weight", "context", "Lowest mean vertical GRF among the four legs."
    ),
    "grf_max_leg_mean": MetricDefinition(
        "body weight", "context", "Highest mean vertical GRF among the four legs."
    ),
}


HARDWARE_COMMAND_CASES = (
    ("stand", (0.0, 0.0, 0.0)),
    ("forward_0p5", (0.5, 0.0, 0.0)),
    ("forward_1p0", (1.0, 0.0, 0.0)),
    ("backward_0p5", (-0.5, 0.0, 0.0)),
    ("left_0p4", (0.0, 0.4, 0.0)),
    ("right_0p4", (0.0, -0.4, 0.0)),
    ("yaw_left_0p75", (0.0, 0.0, 0.75)),
    ("yaw_right_0p75", (0.0, 0.0, -0.75)),
    ("combined", (0.75, 0.25, 0.5)),
)

GO2_COMMAND_CASES = (
    ("stand", (0.0, 0.0, 0.0)),
    ("backward_0p5", (-0.5, 0.0, 0.0)),
    ("forward_0p5", (0.5, 0.0, 0.0)),
    ("forward_1p0", (1.0, 0.0, 0.0)),
    ("forward_3p0", (3.0, 0.0, 0.0)),
    ("left_0p4", (0.0, 0.4, 0.0)),
    ("right_0p4", (0.0, -0.4, 0.0)),
    ("yaw_left_0p75", (0.0, 0.0, 0.75)),
    ("yaw_right_0p75", (0.0, 0.0, -0.75)),
    ("combined", (0.75, 0.25, 0.5)),
)


def metric_metadata():
    return {name: asdict(definition) for name, definition in METRIC_DEFINITIONS.items()}


def apply_command_profile(env, profile):
    """Apply a repeatable command suite and return one case label per env."""
    if profile == "sampled":
        return np.full(env.num_envs, "sampled", dtype="<U16")
    if profile == "forward_3p0":
        env.commands[:, :3] = torch.tensor(
            [3.0, 0.0, 0.0],
            dtype=env.commands.dtype,
            device=env.device,
        )
        if hasattr(env, "_update_cmd_switch"):
            env._update_cmd_switch()
        return np.full(env.num_envs, "forward_3p0", dtype="<U16")
    command_cases = {
        "hardware": HARDWARE_COMMAND_CASES,
        "go2": GO2_COMMAND_CASES,
    }.get(profile)
    if command_cases is None:
        raise ValueError(f"unknown command profile {profile!r}")

    case_indices = np.arange(env.num_envs) % len(command_cases)
    command_values = torch.tensor(
        [command_cases[index][1] for index in case_indices],
        dtype=env.commands.dtype,
        device=env.device,
    )
    env.commands[:, :3] = command_values
    if hasattr(env, "_update_cmd_switch"):
        env._update_cmd_switch()
    return np.asarray(
        [command_cases[index][0] for index in case_indices],
        dtype="<U24",
    )


def _select_actuated(env, values):
    if values.shape[1] == len(env.actuated_dof_indices):
        return values
    return values.index_select(1, env.actuated_dof_indices)


def actuated_position_reference(env):
    """Return a relative gait reference in actuated-DOF order, if one exists."""
    if hasattr(env, "gait_reference"):
        return _select_actuated(env, env.gait_reference)
    if hasattr(env, "_get_ref"):
        return _select_actuated(env, env._get_ref())
    return None


def actuated_position_target(env):
    """Return the actual absolute PD target in actuated-DOF order."""
    default = env.default_dof_pos.index_select(1, env.actuated_dof_indices)
    target = default + env.dof_pos_target
    if hasattr(env, "gait_reference"):
        target = target + _select_actuated(env, env.gait_reference)
    return target


def velocity_impulse_schedule(
    num_envs,
    sample_rate_hz,
    start_time_s,
    stagger_time_s,
    magnitude,
    num_directions=36,
):
    """Build the phase-staggered planar velocity perturbations from Table I."""
    num_stagger_steps = max(1, int(round(stagger_time_s * sample_rate_hz)))
    env_indices = np.arange(num_envs)
    direction_indices = env_indices % num_directions
    group_indices = env_indices // num_directions
    num_groups = int(np.ceil(num_envs / num_directions))
    if num_groups == 1:
        stagger_indices = np.zeros(num_envs, dtype=np.int64)
    else:
        stagger_indices = np.rint(
            group_indices * (num_stagger_steps - 1) / (num_groups - 1)
        ).astype(np.int64)
    angles = 2.0 * np.pi * direction_indices / num_directions
    steps = int(round(start_time_s * sample_rate_hz)) + stagger_indices
    delta_velocity = magnitude * np.stack(
        (np.cos(angles), np.sin(angles)),
        axis=1,
    )
    return (
        steps.astype(np.int64),
        angles.astype(np.float32),
        delta_velocity.astype(np.float32),
    )


class LeggedMetricAccumulator:
    """Accumulate per-environment metrics without storing full trajectories."""

    def __init__(
        self,
        env,
        settle_steps=50,
        contact_threshold=20.0,
        num_steps=None,
        robot_mass_kg=None,
    ):
        self.env = env
        self.settle_steps = settle_steps
        self.contact_threshold = contact_threshold
        self.robot_mass_kg = robot_mass_kg
        self.dt = float(env.dt)
        self.num_envs = env.num_envs
        self.device = env.device
        self.evaluation_commands = env.commands[:, :3].detach().clone()
        self.moving = torch.linalg.vector_norm(self.evaluation_commands, dim=1) > 0.1
        self._sum = {}
        self._count = {}
        self._minimum = {}
        self._maximum = {}
        self._previous_target = None
        self._previous_target_rate = None
        self._previous_dof_velocity = None
        self._previous_base_velocity_z = None
        self._previous_torque = None
        self.artifacts = {}
        self._histories = None
        self._has_phase_reference = hasattr(env, "_leg_phases")
        if num_steps is not None:
            num_feet = len(env.feet_indices)
            self._histories = {
                "base_height": np.empty(
                    (num_steps, self.num_envs),
                    dtype=np.float32,
                ),
                "foot_force_norm": np.empty(
                    (num_steps, self.num_envs, num_feet),
                    dtype=np.float32,
                ),
                "foot_force_z": np.empty(
                    (num_steps, self.num_envs, num_feet),
                    dtype=np.float32,
                ),
                "foot_height": np.empty(
                    (num_steps, self.num_envs, num_feet),
                    dtype=np.float32,
                ),
                "alive": np.empty(
                    (num_steps, self.num_envs),
                    dtype=np.bool_,
                ),
            }
            if self._has_phase_reference:
                self._histories["leg_phase"] = np.empty(
                    (num_steps, self.num_envs, num_feet),
                    dtype=np.float32,
                )
                self._histories["expected_stance"] = np.empty(
                    (num_steps, self.num_envs, num_feet),
                    dtype=np.bool_,
                )

    def _zeros(self):
        return torch.zeros(self.num_envs, dtype=torch.float64, device=self.device)

    def _valid_for(self, values, valid):
        if values.ndim == 1:
            values = values.unsqueeze(1)
        if valid.ndim > values.ndim:
            raise ValueError(
                f"valid mask shape {valid.shape} cannot index values {values.shape}"
            )
        expanded = valid.reshape(
            *valid.shape,
            *([1] * (values.ndim - valid.ndim)),
        )
        return expanded.expand_as(values)

    def _add_mean(self, name, values, valid):
        values = values.to(torch.float64)
        if values.ndim == 1:
            values = values.unsqueeze(1)
        sample_valid = self._valid_for(values, valid)
        self._sum.setdefault(name, self._zeros()).add_(
            torch.where(sample_valid, values, 0.0).flatten(1).sum(dim=1)
        )
        self._count.setdefault(name, self._zeros()).add_(
            sample_valid.flatten(1).sum(dim=1)
        )

    def _add_rms(self, name, values, valid):
        self._add_mean(name, torch.square(values), valid)

    def _add_min(self, name, values, valid):
        values = values.to(torch.float64)
        if values.ndim == 1:
            values = values.unsqueeze(1)
        sample_valid = self._valid_for(values, valid)
        sample = (
            torch.where(sample_valid, values, torch.inf).flatten(1).min(dim=1).values
        )
        self._minimum.setdefault(
            name,
            torch.full(
                (self.num_envs,),
                torch.inf,
                dtype=torch.float64,
                device=self.device,
            ),
        )
        self._minimum[name] = torch.minimum(self._minimum[name], sample)

    def _add_max(self, name, values, valid):
        values = values.to(torch.float64)
        if values.ndim == 1:
            values = values.unsqueeze(1)
        sample_valid = self._valid_for(values, valid)
        sample = (
            torch.where(sample_valid, values, -torch.inf).flatten(1).max(dim=1).values
        )
        self._maximum.setdefault(
            name,
            torch.full(
                (self.num_envs,),
                -torch.inf,
                dtype=torch.float64,
                device=self.device,
            ),
        )
        self._maximum[name] = torch.maximum(self._maximum[name], sample)

    def update(self, step, alive):
        """Record the state after one control step.

        ``alive`` excludes environments terminated by this step because the task
        has already reset their state before returning to the evaluator.
        """
        env = self.env
        settled = step >= self.settle_steps
        valid = alive & settled
        actuated = env.actuated_dof_indices
        dof_position = env.dof_pos.index_select(1, actuated)
        dof_velocity = env.dof_vel.index_select(1, actuated)
        target = env.dof_pos_target
        torque = env.torques
        if self._histories is not None:
            foot_forces = env.contact_forces[:, env.feet_indices, :]
            self._histories["base_height"][step] = (
                env.base_height.flatten().detach().cpu().numpy()
            )
            self._histories["foot_force_norm"][step] = (
                torch.linalg.vector_norm(foot_forces, dim=-1).detach().cpu().numpy()
            )
            self._histories["foot_force_z"][step] = (
                foot_forces[:, :, 2].detach().cpu().numpy()
            )
            self._histories["foot_height"][step] = (
                env._rigid_body_pos[:, env.feet_indices, 2].detach().cpu().numpy()
            )
            self._histories["alive"][step] = alive.detach().cpu().numpy()
            if self._has_phase_reference:
                leg_phase = env._leg_phases()
                expected_stance = (
                    env._expected_stance()
                    if hasattr(env, "_expected_stance")
                    else leg_phase > torch.pi
                )
                self._histories["leg_phase"][step] = leg_phase.detach().cpu().numpy()
                self._histories["expected_stance"][step] = (
                    expected_stance.detach().cpu().numpy()
                )

        self._add_rms(
            "tracking_vx_rmse",
            env.base_lin_vel[:, 0] - self.evaluation_commands[:, 0],
            valid,
        )
        self._add_rms(
            "tracking_vy_rmse",
            env.base_lin_vel[:, 1] - self.evaluation_commands[:, 1],
            valid,
        )
        self._add_rms(
            "tracking_yaw_rmse",
            env.base_ang_vel[:, 2] - self.evaluation_commands[:, 2],
            valid,
        )
        self._add_rms(
            "base_height_error_rms",
            env.base_height.flatten() - env.cfg.reward_settings.base_height_target,
            valid,
        )
        tilt = torch.rad2deg(
            torch.acos(torch.clamp(-env.projected_gravity[:, 2], -1.0, 1.0))
        )
        self._add_rms("base_tilt_rms", tilt, valid)
        self._add_rms("base_vertical_velocity_rms", env.base_lin_vel[:, 2], valid)
        self._add_rms("base_roll_pitch_rate_rms", env.base_ang_vel[:, :2], valid)
        self._add_rms("joint_velocity_rms", dof_velocity, valid)
        self._add_rms("torque_rms", torque, valid)

        torque_limits = env.actuated_torque_limits.to(torque.device).clamp_min(1e-9)
        torque_utilization = torch.abs(torque) / torque_limits
        self._add_rms("torque_utilization_rms", torque_utilization, valid)
        self._add_max("torque_utilization_peak", torque_utilization, valid)
        self._add_mean(
            "torque_saturation_fraction",
            (torque_utilization >= 0.98).float(),
            valid,
        )
        self._add_mean(
            "mechanical_power_mean",
            torch.sum(torch.abs(torque * dof_velocity), dim=1),
            valid,
        )

        velocity_limits = env.dof_vel_limits.index_select(0, actuated).clamp_min(1e-9)
        velocity_utilization = torch.abs(dof_velocity) / velocity_limits
        self._add_rms(
            "joint_velocity_utilization_rms",
            velocity_utilization,
            valid,
        )
        self._add_mean(
            "joint_velocity_limit_fraction",
            (velocity_utilization >= 1.0).float(),
            valid,
        )

        position_limits = env.dof_pos_limits.index_select(0, actuated)
        joint_range = (position_limits[:, 1] - position_limits[:, 0]).clamp_min(1e-9)
        limit_margin = (
            torch.minimum(
                dof_position - position_limits[:, 0],
                position_limits[:, 1] - dof_position,
            )
            / joint_range
        )
        self._add_min("joint_limit_margin_min", limit_margin, valid)
        target_position = actuated_position_target(env)
        target_limit_margin = (
            torch.minimum(
                target_position - position_limits[:, 0],
                position_limits[:, 1] - target_position,
            )
            / joint_range
        )
        self._add_min(
            "target_joint_limit_margin_min",
            target_limit_margin,
            valid,
        )

        reference = actuated_position_reference(env)
        if reference is not None:
            reference = reference + env.default_dof_pos.index_select(1, actuated)
            self._add_rms(
                "reference_joint_rmse",
                dof_position - reference,
                valid & self.moving,
            )

        if self._previous_target is not None:
            target_rate = (target - self._previous_target) / self.dt
            self._add_rms("target_velocity_rms", target_rate, valid)
            if self._previous_target_rate is not None:
                target_acceleration = (
                    target_rate - self._previous_target_rate
                ) / self.dt
                self._add_rms("target_acceleration_rms", target_acceleration, valid)
            self._previous_target_rate = target_rate.detach().clone()
        if self._previous_dof_velocity is not None:
            joint_acceleration = (dof_velocity - self._previous_dof_velocity) / self.dt
            self._add_rms("joint_acceleration_rms", joint_acceleration, valid)
        if self._previous_base_velocity_z is not None:
            vertical_acceleration = (
                env.base_lin_vel[:, 2] - self._previous_base_velocity_z
            ) / self.dt
            self._add_rms(
                "base_vertical_acceleration_rms", vertical_acceleration, valid
            )
        if self._previous_torque is not None:
            torque_rate = (torque - self._previous_torque) / self.dt
            self._add_rms("torque_rate_rms", torque_rate, valid)

        self._previous_target = target.detach().clone()
        self._previous_dof_velocity = dof_velocity.detach().clone()
        self._previous_base_velocity_z = env.base_lin_vel[:, 2].detach().clone()
        self._previous_torque = torque.detach().clone()

        self._record_contacts(valid)

    def _record_contacts(self, valid):
        env = self.env
        foot_forces = torch.linalg.vector_norm(
            env.contact_forces[:, env.feet_indices, :],
            dim=-1,
        )
        in_contact = foot_forces > self.contact_threshold
        feet_valid = valid.unsqueeze(1).expand_as(in_contact)
        self._add_max("foot_force_peak", foot_forces, feet_valid)

        foot_velocity = env._rigid_body_lin_vel[:, env.feet_indices, :2]
        slip_valid = feet_valid & in_contact
        self._add_rms("foot_slip_speed_rms", foot_velocity, slip_valid)

        if hasattr(env, "_leg_phases"):
            expected_stance = (
                env._expected_stance()
                if hasattr(env, "_expected_stance")
                else env._leg_phases() > torch.pi
            )
            if expected_stance.shape == in_contact.shape:
                gait_valid = feet_valid & self.moving.unsqueeze(1)
                self._add_mean(
                    "foot_contact_phase_match",
                    (in_contact == expected_stance).float(),
                    gait_valid,
                )
                self._add_mean(
                    "swing_contact_fraction",
                    in_contact.float(),
                    gait_valid & ~expected_stance,
                )
                self._add_mean(
                    "stance_miss_fraction",
                    (~in_contact).float(),
                    gait_valid & expected_stance,
                )

        penalized = env.penalised_contact_indices
        if len(penalized):
            unsafe_force = torch.linalg.vector_norm(
                env.contact_forces[:, penalized, :],
                dim=-1,
            )
            unsafe_step = torch.any(
                unsafe_force > self.contact_threshold,
                dim=1,
            )
            self._add_mean("unsafe_contact_fraction", unsafe_step.float(), valid)

    def finalize(self, survived=None):
        metrics = {}
        rms_names = {
            "tracking_vx_rmse",
            "tracking_vy_rmse",
            "tracking_yaw_rmse",
            "base_height_error_rms",
            "base_tilt_rms",
            "base_vertical_velocity_rms",
            "base_vertical_acceleration_rms",
            "base_roll_pitch_rate_rms",
            "reference_joint_rmse",
            "joint_velocity_rms",
            "joint_velocity_utilization_rms",
            "joint_acceleration_rms",
            "target_velocity_rms",
            "target_acceleration_rms",
            "torque_rms",
            "torque_rate_rms",
            "torque_utilization_rms",
            "foot_slip_speed_rms",
        }
        for name, total in self._sum.items():
            count = self._count[name]
            value = torch.full_like(total, torch.nan)
            populated = count > 0
            value[populated] = total[populated] / count[populated]
            if name in rms_names:
                value = torch.sqrt(value)
            metrics[name] = value.cpu().numpy().astype(np.float32)
        for name, value in self._minimum.items():
            value[torch.isinf(value)] = torch.nan
            metrics[name] = value.cpu().numpy().astype(np.float32)
        for name, value in self._maximum.items():
            value[torch.isinf(value)] = torch.nan
            metrics[name] = value.cpu().numpy().astype(np.float32)
        if self._histories is not None:
            if survived is None:
                raise ValueError("survived mask is required for signal analysis")
            if self.robot_mass_kg is None:
                raise ValueError("robot_mass_kg is required for GRF analysis")
            sample_rate_hz = 1.0 / self.dt
            if hasattr(self.env, "phase_frequency"):
                gait_frequency_hz = (
                    self.env.phase_frequency.flatten().detach().cpu().numpy()
                )
            else:
                configured_frequency = getattr(self.env.cfg.control, "gait_freq", 1.0)
                gait_frequency_hz = float(np.mean(configured_frequency))
            metrics.update(
                analyze_base_height(
                    self._histories["base_height"],
                    self._histories["alive"],
                    sample_rate_hz,
                    self.settle_steps,
                )
            )
            moving = self.moving.clone()
            if not hasattr(self.env, "_leg_phases"):
                moving[:] = False
            gait_metrics, gait_artifacts = analyze_gait_and_grf(
                self._histories["foot_force_norm"],
                self._histories["foot_force_z"],
                self._histories["alive"],
                moving.detach().cpu().numpy(),
                sample_rate_hz,
                self.settle_steps,
                self.contact_threshold,
                gait_frequency_hz,
                self.robot_mass_kg,
            )
            metrics.update(gait_metrics)
            self.artifacts.update(gait_artifacts)
            self.artifacts["reference_gait_frequency_hz"] = np.broadcast_to(
                gait_frequency_hz, (self.num_envs,)
            ).astype(np.float32)
            if self._has_phase_reference:
                clearance_metrics, clearance_artifacts = (
                    analyze_foot_clearance_by_phase(
                        self._histories["foot_height"],
                        self._histories["foot_force_norm"],
                        self._histories["leg_phase"],
                        self._histories["expected_stance"],
                        self._histories["alive"],
                        self.moving.detach().cpu().numpy(),
                        self.settle_steps,
                        self.contact_threshold,
                    )
                )
                metrics.update(clearance_metrics)
                self.artifacts.update(clearance_artifacts)
        return metrics


def summarize_metrics(metrics, command_cases):
    """Return JSON-serializable overall and per-command distribution summaries."""

    def _summarize(values):
        finite = np.asarray(values)[np.isfinite(values)]
        if not len(finite):
            return {"mean": None, "median": None, "p10": None, "p90": None}
        return {
            "mean": float(np.mean(finite)),
            "median": float(np.median(finite)),
            "p10": float(np.quantile(finite, 0.1)),
            "p90": float(np.quantile(finite, 0.9)),
        }

    command_cases = np.asarray(command_cases)
    groups = {"overall": np.ones(len(command_cases), dtype=bool)}
    groups.update(
        {
            str(case): command_cases == case
            for case in dict.fromkeys(command_cases.tolist())
        }
    )
    return {
        group: {name: _summarize(values[mask]) for name, values in metrics.items()}
        for group, mask in groups.items()
    }
