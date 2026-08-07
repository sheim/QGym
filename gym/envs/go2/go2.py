import torch

from gym.utils.sampling import torch_rand_float

from gym.envs.base.legged_robot import LeggedRobot
# from learning.utils.logger.SaveStates import (
#     init_env_log_buffers,
# )


class Go2(LeggedRobot):
    def __init__(self, cfg, device, headless, backend):
        super().__init__(cfg, device, headless, backend)

    def _init_buffers(self):
        super()._init_buffers()

        self.phase = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device
        )
        self.phase_obs = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device
        )
        self.phase_frequency = torch.ones(
            self.num_envs, 1, dtype=torch.float, device=self.device
        )
        self.gait_reference = torch.zeros_like(self.dof_pos_target)
        foot_names = self.robot_layout.body_groups["feet"]
        phase_offsets = self.cfg.control.gait_phase_offsets
        self._gait_phase_offsets = (
            2
            * torch.pi
            * torch.tensor(
                [phase_offsets[name] for name in foot_names],
                dtype=torch.float,
                device=self.device,
            )
        )
        self._gait_dof_phase_offsets = self._gait_phase_offsets.repeat_interleave(3)
        self._gait_joint_offsets = torch.tensor(
            self.cfg.control.gait_joint_offsets,
            dtype=torch.float,
            device=self.device,
        )
        self._gait_joint_amplitudes = torch.tensor(
            self.cfg.control.gait_joint_amplitudes,
            dtype=torch.float,
            device=self.device,
        )
        self._update_phase_observation()
        self._update_gait_reference()

    def _update_phase_observation(self):
        self.phase_obs[:, 0:1] = torch.sin(self.phase)
        self.phase_obs[:, 1:2] = torch.cos(self.phase)

    def _update_gait_reference(self):
        joint_phase = self.phase + self._gait_dof_phase_offsets.unsqueeze(0)
        self.gait_reference[:] = (
            self._gait_joint_offsets
            + self._gait_joint_amplitudes * torch.sin(joint_phase)
        )

    def _pre_decimation_step(self):
        self._update_gait_reference()

    def _compute_torques(self):
        pos = self.dof_pos.index_select(1, self.actuated_dof_indices)
        vel = self.dof_vel.index_select(1, self.actuated_dof_indices)
        default_pos = self.default_dof_pos.index_select(1, self.actuated_dof_indices)
        torques = (
            self.p_gains
            * (self.gait_reference + self.dof_pos_target + default_pos - pos)
            + self.d_gains * (self.dof_vel_target - vel)
            + self.tau_ff
        )
        return torch.clip(
            torques, -self.actuated_torque_limits, self.actuated_torque_limits
        ).view(self.torques.shape)

    def _reset_system(self, env_ids):
        super()._reset_system(env_ids)
        self.phase[env_ids] = torch_rand_float(
            0, 2 * torch.pi, shape=self.phase[env_ids].shape, device=self.device
        )
        self.phase_frequency[env_ids] = torch_rand_float(
            self.cfg.control.gait_freq[0],
            self.cfg.control.gait_freq[1],
            shape=self.phase_frequency[env_ids].shape,
            device=self.device,
        )

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        self.dof_pos_target[env_ids] = 0.0
        self._update_gait_reference()
        self.dof_pos_history[env_ids] = self.dof_pos_target[env_ids].tile(3)
        self._update_phase_observation()

    def _post_physics_step(self):
        super()._post_physics_step()
        self._advance_phase()

    def _advance_phase(self):
        # phase_frequency is in cycles/s; _post_physics_step runs once per
        # physics substep, so convert cycles to radians and use the simulation dt.
        self.phase.add_(
            2 * torch.pi * self.dt * self.phase_frequency / self.cfg.control.decimation
        ).remainder_(2 * torch.pi)

    def _post_decimation_step(self):
        super()._post_decimation_step()
        self._update_phase_observation()

    def _reward_trot_contact(self):
        """Reward contact in the positive half-cycle of each foot's phase."""
        contact = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.reward_settings.gait_contact_force_threshold
        ).float()
        gait_phase = self.phase + self._gait_phase_offsets.unsqueeze(0)
        return torch.mean(torch.sin(gait_phase) * contact, dim=1)

    def _reward_lin_vel_z(self):
        """Penalize z axis base linear velocity with squared exp"""
        return self._sqrdexp(self.base_lin_vel[:, 2] / self.scales["base_lin_vel"])

    def _reward_ang_vel_xy(self):
        """Penalize xy axes base angular velocity"""
        error = self._sqrdexp(self.base_ang_vel[:, :2] / self.scales["base_ang_vel"])
        return torch.sum(error, dim=1)

    def _reward_orientation(self):
        """Penalize non-flat base orientation"""
        error = (
            torch.square(self.projected_gravity[:, :2])
            / self.cfg.reward_settings.tracking_sigma
        )
        return torch.sum(torch.exp(-error), dim=1)

    def _reward_min_base_height(self):
        """Squared exponential saturating at base_height target"""
        error = self.base_height - self.cfg.reward_settings.base_height_target
        error /= self.scales["base_height"]
        error = torch.clamp(error, max=0, min=None).flatten()
        return self._sqrdexp(error)

    def _reward_tracking_lin_vel(self):
        """Tracking of linear velocity commands (xy axes)"""
        # just use lin_vel?
        error = self.commands[:, :2] - self.base_lin_vel[:, :2]
        # * scale by (1+|cmd|): if cmd=0, no scaling.
        error *= 1.0 / (1.0 + torch.abs(self.commands[:, :2]))
        error = torch.sum(torch.square(error), dim=1)
        return torch.exp(-error / self.cfg.reward_settings.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        """Tracking of angular velocity commands (yaw)"""
        ang_vel_error = torch.square(
            (self.commands[:, 2] - self.base_ang_vel[:, 2]) / 2.5
        )
        return self._sqrdexp(ang_vel_error)

    def _reward_dof_vel(self):
        """Penalize dof velocities"""
        return torch.sum(self._sqrdexp(self.dof_vel / self.scales["dof_vel"]), dim=1)

    def _reward_dof_near_home(self):
        return torch.sum(
            self._sqrdexp(
                (self.dof_pos - self.default_dof_pos) / self.scales["dof_pos_obs"]
            ),
            dim=1,
        )
