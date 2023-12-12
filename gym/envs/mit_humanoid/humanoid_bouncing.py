import torch

from isaacgym.torch_utils import quat_rotate_inverse
from gym.envs import LeggedRobot
from isaacgym import gymutil
from isaacgym import gymapi


class HumanoidBouncing(LeggedRobot):
    def __init__(self, gym, sim, cfg, sim_params, sim_device, headless):
        super().__init__(gym, sim, cfg, sim_params, sim_device, headless)

    def _init_buffers(self):
        super()._init_buffers()

        # * get the body_name to body_index dict
        body_dict = self.gym.get_actor_rigid_body_dict(
            self.envs[0], self.actor_handles[0]
        )
        # * extract a list of body_names where the index is the id number
        body_names = [
            body_tuple[0]
            for body_tuple in sorted(
                body_dict.items(), key=lambda body_tuple: body_tuple[1]
            )
        ]
        # * construct a list of id numbers corresponding to end_effectors
        self.end_effector_ids = []
        for end_effector_name in self.cfg.asset.end_effector_names:
            self.end_effector_ids.extend(
                [
                    body_names.index(body_name)
                    for body_name in body_names
                    if end_effector_name in body_name
                ]
            )

        # * end_effector_pos is world-frame and converted to env_origin
        self.end_effector_pos = self._rigid_body_pos[
            :, self.end_effector_ids
        ] - self.env_origins.unsqueeze(dim=1).expand(
            self.num_envs, len(self.end_effector_ids), 3
        )
        self.end_effector_quat = self._rigid_body_quat[:, self.end_effector_ids]

        self.end_effector_lin_vel = torch.zeros(
            self.num_envs,
            len(self.end_effector_ids),
            3,
            dtype=torch.float,
            device=self.device,
        )
        self.end_effector_ang_vel = torch.zeros(
            self.num_envs,
            len(self.end_effector_ids),
            3,
            dtype=torch.float,
            device=self.device,
        )

        # * end_effector vels are body-relative like body vels above
        for index in range(len(self.end_effector_ids)):
            self.end_effector_lin_vel[:, index, :] = quat_rotate_inverse(
                self.base_quat,
                self._rigid_body_lin_vel[:, self.end_effector_ids][:, index, :],
            )
            self.end_effector_ang_vel[:, index, :] = quat_rotate_inverse(
                self.base_quat,
                self._rigid_body_ang_vel[:, self.end_effector_ids][:, index, :],
            )

        # * separate legs and arms
        self.dof_pos_target_legs = torch.zeros(
            self.num_envs, 10, dtype=torch.float, device=self.device
        )
        self.dof_pos_target_arms = torch.zeros(
            self.num_envs, 8, dtype=torch.float, device=self.device
        )
        self.dof_pos_legs = torch.zeros(
            self.num_envs, 10, dtype=torch.float, device=self.device
        )
        self.dof_pos_arms = torch.zeros(
            self.num_envs, 8, dtype=torch.float, device=self.device
        )
        self.dof_vel_legs = torch.zeros(
            self.num_envs, 10, dtype=torch.float, device=self.device
        )
        self.dof_vel_arms = torch.zeros(
            self.num_envs, 8, dtype=torch.float, device=self.device
        )

        # * other
        self.base_pos = self.root_states[:, 0:3]
        self.phase = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device
        )
        self.phase_freq = 1.0

        # * high level
        self.hl_impulses = torch.zeros(
            self.num_envs, 4, 5, dtype=torch.float, device=self.device
        )
        self.hl_impulses_flat = self.hl_impulses.flatten(start_dim=1)
        self.hl_ix = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device
        )
        self.hl_commands = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device
        )
        self.time_since_hl_query = torch.zeros(
            self.num_envs, 1, device=self.device, dtype=torch.float
        )

    def _pre_physics_step(self):
        super()._pre_physics_step()
        self.dof_pos_target[:, :10] = self.dof_pos_target_legs
        self.dof_pos_target[:, 10:] = self.dof_pos_target_arms

        envs_to_resample = torch.where(
            self.time_since_hl_query == self.cfg.high_level.interval,
            True,
            False,
        )
        if envs_to_resample.any().item():
            self._resample_high_level(
                envs_to_resample.nonzero(as_tuple=False).flatten()
            )

        self.hl_impulses_flat = self.hl_impulses.flatten(start_dim=1)

    def _reset_system(self, env_ids):
        super()._reset_system(env_ids)
        # if self.cfg.commands.resampling_time == -1:
        #     self.commands[env_ids, :] = 0.
        self.phase[env_ids, 0] = torch.rand(
            (torch.numel(env_ids),), requires_grad=False, device=self.device
        )
        self._resample_high_level(env_ids)
        self.hl_impulses_flat = self.hl_impulses.flatten(start_dim=1)

    def _post_physics_step(self):
        super()._post_physics_step()
        self.time_since_hl_query += 1  # self.dt
        self.phase_sin = torch.sin(2 * torch.pi * self.phase)
        self.phase_cos = torch.cos(2 * torch.pi * self.phase)

        self.in_contact = self.contact_forces[:, self.end_effector_ids, 2].gt(0.0)
        self.phase = torch.fmod(self.phase + self.dt, 1.0)

        self.dof_pos_legs = self.dof_pos[:, :10]
        self.dof_pos_arms = self.dof_pos[:, 10:]
        self.dof_vel_legs = self.dof_vel[:, :10]
        self.dof_vel_arms = self.dof_vel[:, 10:]

        # * end_effector_pos is world-frame and converted to env_origin
        self.end_effector_pos = self._rigid_body_pos[
            :, self.end_effector_ids
        ] - self.env_origins.unsqueeze(dim=1).expand(
            self.num_envs, len(self.end_effector_ids), 3
        )
        self.end_effector_quat = self._rigid_body_quat[:, self.end_effector_ids]

        # * end_effector vels are body-relative like body vels above
        for index in range(len(self.end_effector_ids)):
            self.end_effector_lin_vel[:, index, :] = quat_rotate_inverse(
                self.base_quat,
                self._rigid_body_lin_vel[:, self.end_effector_ids][:, index, :],
            )
            self.end_effector_ang_vel[:, index, :] = quat_rotate_inverse(
                self.base_quat,
                self._rigid_body_ang_vel[:, self.end_effector_ids][:, index, :],
            )

        # * update HL indices
        delta_hl_now = self.hl_impulses[:, 0, :] - self.time_since_hl_query.view(
            self.num_envs, 1
        ).expand(self.num_envs, 5)
        self.hl_ix = torch.argmin(
            torch.where(
                delta_hl_now >= 0,
                delta_hl_now,
                float("inf") * torch.ones_like(delta_hl_now),
            ),
            dim=1,
        ).unsqueeze(1)
        self._update_hl_commands()
        # roll out next time step HL command
        self.hl_commands[:, :3] += self.hl_commands[:, 3:] * self.dt
        self.hl_commands[:, 5] -= 9.81 * self.dt

    def _resample_high_level(self, env_ids):
        """
        Updates impulse sequences for envs its passed s.t. the first impulse compensates for
        tracking error and the rest enforce the same bouncing ball trajectory on the COM
        """
        self.time_since_hl_query[env_ids] = 0.0
        delta_touchdown = self._time_to_touchdown(
            (
                self.root_states[env_ids, 2:3]
                - self.cfg.reward_settings.base_height_target
            ),
            self.base_lin_vel[env_ids, 2].unsqueeze(1),
            -9.81,
        )

        step_time = self.cfg.high_level.sec_per_gait

        first_xy = torch.tensor(
            [[[1], [0]]], dtype=torch.float, device=self.device
        ).repeat(env_ids.shape[0], 1, 1) - (
            self.base_lin_vel[env_ids, :2] * delta_touchdown.repeat(1, 2)
        ).unsqueeze(2)
        first_z = torch.tensor(
            [[[step_time * 9.81 / 2]]], dtype=torch.float, device=self.device
        ).repeat(env_ids.shape[0], 1, 1) - (
            self.base_lin_vel[env_ids, 2] - 9.81 * delta_touchdown.squeeze(1)
        ).view(env_ids.shape[0], 1, 1)
        first_impulse = torch.cat((first_xy, first_z), dim=1)
        remaining_impulses = torch.tensor(
            [[0], [0], [step_time * 9.81]], dtype=torch.float, device=self.device
        ).repeat(env_ids.shape[0], 1, 4)
        impulse_mag_buf = torch.cat((first_impulse, remaining_impulses), dim=2)

        time_rollout = torch.cat(
            (
                delta_touchdown,
                delta_touchdown + (1.0 * step_time),
                delta_touchdown + (2.0 * step_time),
                delta_touchdown + (3.0 * step_time),
                delta_touchdown + (4.0 * step_time),
            ),
            dim=1,
        ).unsqueeze(1)

        time_rollout = time_rollout / self.dt
        time_rollout = time_rollout.round().int()

        self.hl_impulses[env_ids] = torch.cat((time_rollout, impulse_mag_buf), dim=1)
        self.hl_ix[env_ids] = 0

        # seed the root states for roll out via post physics
        self.hl_commands[env_ids, :3] = self.base_pos[env_ids, :]
        self.hl_commands[env_ids, 3:] = self.base_lin_vel[env_ids, :]

        if self.cfg.viewer.record:
            # draw new high level target trajectories
            self.gym.clear_lines(self.viewer)
            sphere_geom = gymutil.WireframeSphereGeometry(
                0.1, 64, 64, None, color=(1, 1, 0)
            )

            for i in range(self.num_envs):
                base_pos = (self.root_states[i, :3]).cpu().numpy()
                sphere_pose = gymapi.Transform(
                    gymapi.Vec3(base_pos[0], base_pos[1], base_pos[2]), r=None
                )
                gymutil.draw_lines(
                    sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose
                )
                line = [
                    base_pos[0],
                    base_pos[1],
                    self.cfg.reward_settings.base_height_target,
                    base_pos[0] + 100,
                    base_pos[1],
                    self.cfg.reward_settings.base_height_target,
                ]
                self.gym.add_lines(self.viewer, self.envs[i], 1, line, [0.85, 0.1, 0.1])

    def _time_to_touchdown(self, pos, vel, acc):
        """
        Assumes robot COM in projectile motion to calculate next touchdown
        """
        determinant = torch.square(vel) - 4 * pos * 0.5 * acc
        solution = torch.where(determinant < 0.0, False, True)
        no_solution = torch.eq(solution, False).nonzero(as_tuple=True)[0]
        t1 = (-vel + torch.sqrt(determinant)) / (2 * 0.5 * acc)
        t2 = (-vel - torch.sqrt(determinant)) / (2 * 0.5 * acc)
        result = torch.where(t2 > t1, t2, t1)
        result[no_solution] = 0.0
        return result

    def _update_hl_commands(self):
        """
        Checks which environments should have an impulse imparted on them and
        updates HL command per-step rollout's velocities accordingly
        """
        vel_update_indices = torch.argmax(
            (self.hl_impulses[:, 0, :] == self.time_since_hl_query).float(), dim=1
        ).unsqueeze(1)
        vel_update_indices[
            (vel_update_indices == 0)
            & (self.hl_impulses[:, 0, 0].unsqueeze(1) != self.time_since_hl_query)
        ] = -1
        vel_update_indices = vel_update_indices.squeeze(1)
        vel_update_envs = torch.nonzero(vel_update_indices != -1)
        self.hl_commands[vel_update_envs, 3:] += self.hl_impulses[
            vel_update_envs, 1:, vel_update_indices[vel_update_envs]
        ]

    def _check_terminations_and_timeouts(self):
        """Check if environments need to be reset"""
        super()._check_terminations_and_timeouts()

        # * Termination for velocities, orientation, and low height
        self.terminated |= (self.base_lin_vel.norm(dim=-1, keepdim=True) > 10).any(
            dim=1
        )
        self.terminated |= (self.base_ang_vel.norm(dim=-1, keepdim=True) > 5).any(dim=1)
        self.terminated |= (self.projected_gravity[:, 0:1].abs() > 0.7).any(dim=1)
        self.terminated |= (self.projected_gravity[:, 1:2].abs() > 0.7).any(dim=1)
        self.terminated |= (self.base_pos[:, 2:3] < 0.3).any(dim=1)

        self.to_be_reset = self.timed_out | self.terminated

    # ########################## REWARDS ######################## #

    # * Task rewards * #

    def _reward_tracking_hl_pos(self):
        # error = self.hl_commands[:, 2] - self.base_pos[:, 2]
        # error /= self.scales["hl_pos"]
        # error = error.flatten()
        # return self._sqrdexp(error)
        error = self.hl_commands[:, :3] - self.base_pos[:, :3]
        error /= self.scales["hl_pos"]
        return self._sqrdexp(error).sum(dim=1)

    def _reward_tracking_hl_vel(self):
        error = self.hl_commands[:, 3:] - self.base_lin_vel
        error /= self.scales["hl_vel"]
        return self._sqrdexp(error).sum(dim=1)

    def _reward_tracking_ang_vel(self):
        ang_vel_error = 0.0 - self.base_ang_vel[:, 2]
        ang_vel_error /= self.scales["base_ang_vel"]
        return self._sqrdexp(ang_vel_error)

    # * Shaping rewards * #

    def _reward_base_height(self):
        error = self.base_height - self.cfg.reward_settings.base_height_target
        error /= self.scales["base_height"]
        error = error.flatten()
        return self._sqrdexp(error)

    def _reward_orientation(self):
        return self._sqrdexp(self.projected_gravity[:, 2] + 1)

    def _reward_joint_regularization_legs(self):
        # * Reward joint poses and symmetry
        reward = self._reward_hip_yaw_zero()
        reward += self._reward_hip_abad_symmetry()
        reward += self._reward_hip_pitch_symmetry()
        return reward / 3.0

    def _reward_hip_yaw_zero(self):
        error = self.dof_pos[:, 0] - self.default_dof_pos[:, 0]
        reward = self._sqrdexp(error / self.scales["dof_pos"][0]) / 2.0
        error = self.dof_pos[:, 5] - self.default_dof_pos[:, 5]
        reward += self._sqrdexp(error / self.scales["dof_pos"][5]) / 2.0
        return reward

    def _reward_hip_abad_symmetry(self):
        error = (
            self.dof_pos[:, 1] / self.scales["dof_pos"][1]
            - self.dof_pos[:, 6] / self.scales["dof_pos"][6]
        )
        return self._sqrdexp(error)

    def _reward_hip_pitch_symmetry(self):
        error = (
            self.dof_pos[:, 2] / self.scales["dof_pos"][2]
            + self.dof_pos[:, 7] / self.scales["dof_pos"][7]
        )
        return self._sqrdexp(error)

    def _reward_joint_regularization_arms(self):
        reward = 0
        reward += self._reward_arm_yaw_symmetry()
        reward += self._reward_arm_yaw_zero()
        reward += self._reward_arm_abad_zero()
        reward += self._reward_arm_abad_symmetry()
        reward += self._reward_arm_pitch_symmetry()
        reward += self._reward_arm_pitch_zero()
        reward += self._reward_elbow_zero()
        return reward / 6.0

    def _reward_arm_pitch_symmetry(self):
        error = (
            self.dof_pos[:, 10] / self.scales["dof_pos"][10]
            + self.dof_pos[:, 14] / self.scales["dof_pos"][14]
        )
        return self._sqrdexp(error)

    def _reward_arm_pitch_zero(self):
        error = self.dof_pos[:, 10] - self.default_dof_pos[:, 10]
        reward = self._sqrdexp(error / self.scales["dof_pos"][10])
        error = self.dof_pos[:, 14] - self.default_dof_pos[:, 14]
        reward += self._sqrdexp(error / self.scales["dof_pos"][14])
        return reward / 2.0

    def _reward_elbow_symmetry(self):
        error = (
            self.dof_pos[:, 13] / self.scales["dof_pos"][13]
            + self.dof_pos[:, 17] / self.scales["dof_pos"][17]
        )
        return self._sqrdexp(error)

    def _reward_elbow_zero(self):
        error = self.dof_pos[:, 13] - self.default_dof_pos[:, 13]
        reward = self._sqrdexp(error / self.scales["dof_pos"][13])
        error = self.dof_pos[:, 17] - self.default_dof_pos[:, 17]
        reward += self._sqrdexp(error / self.scales["dof_pos"][17])
        return reward / 2.0

    def _reward_arm_yaw_symmetry(self):
        error = (
            self.dof_pos[:, 12] / self.scales["dof_pos"][12]
            - self.dof_pos[:, 16] / self.scales["dof_pos"][16]
        )
        return self._sqrdexp(error)

    def _reward_arm_yaw_zero(self):
        error = self.dof_pos[:, 12] - self.default_dof_pos[:, 12]
        reward = self._sqrdexp(error / self.scales["dof_pos"][12])
        error = self.dof_pos[:, 16] - self.default_dof_pos[:, 16]
        reward += self._sqrdexp(error / self.scales["dof_pos"][16])
        return reward / 2.0

    def _reward_arm_abad_symmetry(self):
        error = (
            self.dof_pos[:, 11] / self.scales["dof_pos"][11]
            - self.dof_pos[:, 15] / self.scales["dof_pos"][15]
        )
        return self._sqrdexp(error)

    def _reward_arm_abad_zero(self):
        error = self.dof_pos[:, 11] - self.default_dof_pos[:, 11]
        reward = self._sqrdexp(error / self.scales["dof_pos"][11])
        error = self.dof_pos[:, 15] - self.default_dof_pos[:, 15]
        reward += self._sqrdexp(error / self.scales["dof_pos"][15])
        return reward / 2.0
