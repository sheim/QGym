import torch

from gym.utils.sampling import torch_rand_float
from gym.utils.sampling import masked_update

from gym.envs.base.legged_robot import LeggedRobot


class Go2Trot(LeggedRobot):
    def __init__(self, cfg, device, headless, backend):
        super().__init__(cfg, device, headless, backend)

    def _init_buffers(self):
        super()._init_buffers()

        self._actuated_dof_pos_limits = self.dof_pos_limits.index_select(
            0, self.actuated_dof_indices
        )
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

    def _leg_phases(self):
        return torch.remainder(
            self.phase + self._gait_phase_offsets.unsqueeze(0), 2 * torch.pi
        )

    def _expected_stance(self):
        return torch.sin(self._leg_phases()) > 0

    def capture_state(self) -> dict:
        """Snapshot the task state needed to resume this exact rollout.

        Only task-owned buffers are included; physics contact warm start is not,
        so a restored world is self-consistent but not bit-identical to an
        uninterrupted trajectory.
        """
        return {
            "dof_pos": self.dof_pos.clone(),
            "dof_vel": self.dof_vel.clone(),
            "root_states": self.root_states.clone(),
            "dof_pos_target": self.dof_pos_target.clone(),
            "commands": self.commands.clone(),
            "phase": self.phase.clone(),
            "phase_frequency": self.phase_frequency.clone(),
            "episode_length_buf": self.episode_length_buf.clone(),
            "common_step_counter": int(self.common_step_counter),
        }

    def restore_state(self, state: dict) -> None:
        """Restore a snapshot into every world and clear per-episode flags.

        ``reset_state`` clamps joint positions to the asset limits, so a snapshot
        taken outside those limits is restored at the clamped value. The clamp is
        deterministic, so repeated restores of one snapshot agree exactly.
        """
        mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.dof_pos[:] = state["dof_pos"]
        self.dof_vel[:] = state["dof_vel"]
        self.root_states[:] = state["root_states"]
        self.dof_pos_target[:] = state["dof_pos_target"]
        self.commands[:] = state["commands"]
        self.phase[:] = state["phase"]
        self.phase_frequency[:] = state["phase_frequency"]
        self.episode_length_buf[:] = state["episode_length_buf"]
        self.common_step_counter = state["common_step_counter"]
        self._backend.reset_state(mask)
        self._refresh_base_observations(mask)
        self.dof_pos_obs[:] = self.dof_pos - self.default_dof_pos
        self._update_phase_observation()
        self._update_gait_reference()
        self._reset_buffers()

    def _reset_system(self, reset_mask):
        super()._reset_system(reset_mask)
        phase = torch_rand_float(
            0,
            2 * torch.pi,
            shape=self.phase.shape,
            device=self.device,
        )
        phase_frequency = torch_rand_float(
            self.cfg.control.gait_freq[0],
            self.cfg.control.gait_freq[1],
            shape=self.phase_frequency.shape,
            device=self.device,
        )
        masked_update(self.phase, phase, reset_mask)
        masked_update(self.phase_frequency, phase_frequency, reset_mask)

    def _resample_commands(self, command_mask):
        super()._resample_commands(command_mask)

        forward_commands = torch.tensor(
            self.command_ranges["lin_vel_x"], device=self.device
        )
        command_indices = torch.randint(
            len(forward_commands),
            (self.num_envs,),
            device=self.device,
        )
        forward = forward_commands[command_indices] + (
            torch.randn(self.num_envs, device=self.device) * self.cfg.commands.var
        )
        masked_update(self.commands[:, 0], forward, command_mask)

        if 0 in self.cfg.commands.ranges.lin_vel_x:
            # Include forward-only, rotation-only, and stopped examples.
            drop = command_mask.unsqueeze(1) & (
                torch.rand(self.num_envs, 1, device=self.device) >= 0.8
            )
            self.commands[:, 1:].masked_fill_(drop, 0.0)
            drop = command_mask.unsqueeze(1) & (
                torch.rand(self.num_envs, 1, device=self.device) >= 0.8
            )
            self.commands[:, :2].masked_fill_(drop, 0.0)
            drop = command_mask.unsqueeze(1) & (
                torch.rand(self.num_envs, 1, device=self.device) >= 0.9
            )
            self.commands.masked_fill_(drop, 0.0)

    def _reset_idx(self, reset_mask):
        super()._reset_idx(reset_mask)
        self.dof_pos_target.masked_fill_(reset_mask.unsqueeze(1), 0.0)
        self._update_gait_reference()
        masked_update(
            self.dof_pos_history,
            self.dof_pos_target.tile(1, 3),
            reset_mask,
        )
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
        self._update_gait_reference()

    def _foot_contact_strength(self):
        """Map upward foot load smoothly from zero to nominal body-weight."""
        load = torch.clamp(
            self.contact_forces[:, self.feet_indices, 2]
            / (9.81 * self._backend.link_mass.sum(dim=1, keepdim=True)),
            min=0.0,
            max=1.0,
        )
        return load.square() * (3.0 - 2.0 * load)

    def _reward_trot_support(self):
        """Require both feet in the scheduled trot diagonal to support the body."""
        phase = torch.sin(self._leg_phases())
        contact_strength = self._foot_contact_strength()
        stance_strength = torch.where(
            phase > 0.0,
            contact_strength,
            torch.ones_like(contact_strength),
        )
        paired_support = torch.clamp(
            2.0 * torch.sqrt(torch.prod(stance_strength, dim=1)),
            max=1.0,
        )
        return torch.relu(phase).amax(dim=1) * paired_support

    def _reward_swing_contact(self):
        """Penalize load carried by feet during their scheduled swing phase."""
        phase = torch.sin(self._leg_phases())
        return -torch.mean(torch.relu(-phase) * self._foot_contact_strength(), dim=1)

    def _reward_reference_traj(self):
        pos = self.dof_pos.index_select(1, self.actuated_dof_indices)
        default_pos = self.default_dof_pos.index_select(1, self.actuated_dof_indices)
        scale = self.scales["dof_pos"].index_select(0, self.actuated_dof_indices)
        error = (self.gait_reference + default_pos - pos) / scale
        return (self._sqrdexp(error) - 0.2 * error.abs()).mean(dim=1)

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
        ang_vel_error = (self.commands[:, 2] - self.base_ang_vel[:, 2]) / 2.5
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
