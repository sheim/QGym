import torch
import pandas as pd


from gym.utils.gym_math_wrappers import torch_rand_float
from gym.utils.torch_quat import to_torch

from gym import LEGGED_GYM_ROOT_DIR
from gym.envs.mini_cheetah.mini_cheetah import MiniCheetah


class MiniCheetahRef(MiniCheetah):
    def _init_buffers(self):
        super()._init_buffers()
        self._init_reference_trajectory_buffers()
        self._switch = torch.zeros(self.num_envs, 1, device=self.device)
        self.phase = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device
        )
        self.phase_obs = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device
        )

    def _init_reference_trajectory_buffers(self):
        csv_path = self.cfg.init_state.ref_traj.format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR
        )
        self.leg_ref = 3 * to_torch(
            pd.read_csv(csv_path).to_numpy(),
            device=self.device,
        )
        self.omega = 2 * torch.pi * self.cfg.control.gait_freq
        leg_groups = list(self.cfg.control.reference_leg_groups)
        phase_offsets = self.cfg.control.gait_phase_offsets
        if set(leg_groups) != set(phase_offsets):
            raise ValueError(
                "reference_leg_groups and gait_phase_offsets must have the same keys"
            )
        self._reference_leg_indices = []
        for group_name in leg_groups:
            actuator_indices = [
                self.actuated_dof_names.index(name)
                for name in self.robot_layout.dof_groups[group_name]
            ]
            if len(actuator_indices) != self.leg_ref.shape[1]:
                raise ValueError(
                    f"reference group {group_name!r} has {len(actuator_indices)} "
                    f"DOFs, trajectory has {self.leg_ref.shape[1]} columns"
                )
            self._reference_leg_indices.append(
                torch.tensor(actuator_indices, dtype=torch.long, device=self.device)
            )
        self._gait_phase_offsets = (
            2
            * torch.pi
            * torch.tensor(
                [phase_offsets[name] for name in leg_groups],
                dtype=torch.float,
                device=self.device,
            )
        )
        # if len(self.feet_indices) != len(self._reference_leg_indices):
        #     raise ValueError(
        #         "feet body group and reference leg groups must have the same length"
        #     )

    def _reset_system(self, env_ids):
        super()._reset_system(env_ids)
        self.phase[env_ids] = torch_rand_float(
            0, 2 * torch.pi, shape=self.phase[env_ids].shape, device=self.device
        )

    def _post_physics_step(self):
        super()._post_physics_step()
        self.phase = (
            self.phase + self.dt * self.omega / self.cfg.control.decimation
        ).fmod(2 * torch.pi)

    def _post_decimation_step(self):
        super()._post_decimation_step()
        self.phase_obs = torch.cat(
            (torch.sin(self.phase), torch.cos(self.phase)), dim=1
        )
        self._update_cmd_switch()

    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        axis_aligned_fraction = getattr(self.cfg.commands, "axis_aligned_fraction", 0.0)
        if axis_aligned_fraction:
            axis_aligned = (
                torch.rand(len(env_ids), device=self.device) < axis_aligned_fraction
            )
            selected_ids = env_ids[axis_aligned]
            selected_axes = torch.randint(
                0,
                3,
                (len(selected_ids), 1),
                device=self.device,
            )
            command_mask = torch.zeros(
                len(selected_ids),
                3,
                dtype=self.commands.dtype,
                device=self.device,
            )
            command_mask.scatter_(1, selected_axes, 1.0)
            self.commands[selected_ids, :3] *= command_mask
        # * with 10% chance, reset to 0 commands
        rand_ids = torch_rand_float(
            0, 1, (len(env_ids), 1), device=self.device
        ).squeeze(1)
        self.commands[env_ids, :3] *= (rand_ids < 0.9).unsqueeze(1)

    def _check_terminations_and_timeouts(self):
        """Check if environments need to be reset"""
        contact_forces = self.contact_forces[:, self.termination_contact_indices, :]
        self.terminated |= torch.any(torch.norm(contact_forces, dim=-1) > 1.0, dim=1)
        self.timed_out = self.episode_length_buf >= self.max_episode_length
        # self.to_be_reset = self.timed_out | self.terminated

    # ---

    def _update_cmd_switch(self):
        c_vel = torch.linalg.norm(self.commands, dim=1)
        self._switch = torch.exp(
            -torch.square(torch.max(torch.zeros_like(c_vel), c_vel - 0.1))
            / self.cfg.reward_settings.switch_scale
        )

    def _reward_swing_grf(self):
        """Reward non-zero grf during swing (0 to pi)"""
        in_contact = torch.gt(
            torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1),
            50.0,
        )
        swing = self._leg_phases() < torch.pi
        rew = in_contact * swing
        return -torch.sum(rew.float(), dim=1) * (1 - self._switch)

    def _reward_stance_grf(self):
        """Reward non-zero grf during stance (pi to 2pi)"""
        in_contact = torch.gt(
            torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1),
            50.0,
        )
        stance = self._leg_phases() > torch.pi
        rew = in_contact * stance

        return torch.sum(rew.float(), dim=1) * (1 - self._switch)

    def _reward_reference_traj(self):
        """REWARDS EACH LEG INDIVIDUALLY BASED ON ITS POSITION IN THE CYCLE"""
        # * dof position error
        error = self._get_ref() + self.default_dof_pos - self.dof_pos
        error /= self.scales["dof_pos"]
        reward = (self._sqrdexp(error) - torch.abs(error) * 0.2).mean(dim=1)
        # * only when commanded velocity is higher
        return reward * (1 - self._switch)

    def _get_ref(self):
        leg_frame = torch.zeros_like(self.torques)
        phases = self._leg_phases()
        num_trajectory_samples = self.leg_ref.size(dim=0)
        trajectory_samples_per_radian = num_trajectory_samples / (2 * torch.pi)
        for leg_index, actuator_indices in enumerate(self._reference_leg_indices):
            sample_indices = (
                torch.round(phases[:, leg_index] * trajectory_samples_per_radian).long()
                % num_trajectory_samples
            )
            leg_frame[:, actuator_indices] = self.leg_ref[sample_indices]
        return leg_frame

    def _leg_phases(self):
        return torch.remainder(
            self.phase + self._gait_phase_offsets.unsqueeze(0), 2 * torch.pi
        )

    def _reward_stand_still(self):
        """Penalize motion at zero commands"""
        # * normalize angles so we care about being within 5 deg
        rew_pos = torch.mean(
            self._sqrdexp((self.dof_pos - self.default_dof_pos) / torch.pi * 36),
            dim=1,
        )
        rew_vel = torch.mean(self._sqrdexp(self.dof_vel), dim=1)
        rew_base_vel = torch.mean(torch.square(self.base_lin_vel), dim=1)
        rew_base_vel += torch.mean(torch.square(self.base_ang_vel), dim=1)
        return (rew_vel + rew_pos - rew_base_vel) * self._switch

    def _reward_tracking_lin_vel(self):
        """Tracking linear velocity commands (xy axes)"""
        # just use lin_vel?
        reward = super()._reward_tracking_lin_vel()
        return reward * (1 - self._switch)
