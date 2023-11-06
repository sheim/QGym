import torch

from isaacgym.torch_utils import quat_rotate_inverse
from gym.envs import LeggedRobot


class HumanoidBouncing(LeggedRobot):
    def __init__(self, gym, sim, cfg, sim_params, sim_device, headless):
        super().__init__(gym, sim, cfg, sim_params, sim_device, headless)

    def _init_buffers(self):
        super()._init_buffers()

        # * get the body_name to body_index dict
        body_dict = self.gym.get_actor_rigid_body_dict(
            self.envs[0], self.actor_handles[0])
        # * extract a list of body_names where the index is the id number
        body_names = [body_tuple[0] for body_tuple in
                      sorted(body_dict.items(),
                             key=lambda body_tuple: body_tuple[1])]
        # * construct a list of id numbers corresponding to end_effectors
        self.end_effector_ids = []
        for end_effector_name in self.cfg.asset.end_effector_names:
            self.end_effector_ids.extend([
                body_names.index(body_name)
                for body_name in body_names
                if end_effector_name in body_name])

        # * end_effector_pos is world-frame and converted to env_origin
        self.end_effector_pos = (
            self._rigid_body_pos[:, self.end_effector_ids]
            - self.env_origins.unsqueeze(dim=1).expand(
                self.num_envs, len(self.end_effector_ids), 3))
        self.end_effector_quat = \
            self._rigid_body_quat[:, self.end_effector_ids]

        self.end_effector_lin_vel = torch.zeros(
            self.num_envs, len(self.end_effector_ids), 3,
            dtype=torch.float, device=self.device)
        self.end_effector_ang_vel = torch.zeros(
            self.num_envs, len(self.end_effector_ids), 3,
            dtype=torch.float, device=self.device)

        # * end_effector vels are body-relative like body vels above
        for index in range(len(self.end_effector_ids)):
            self.end_effector_lin_vel[:, index, :] = quat_rotate_inverse(
                self.base_quat,
                self._rigid_body_lin_vel[:, self.end_effector_ids]
                                        [:, index, :])
            self.end_effector_ang_vel[:, index, :] = quat_rotate_inverse(
                self.base_quat,
                self._rigid_body_ang_vel[:, self.end_effector_ids]
                                        [:, index, :])

        # * separate legs and arms
        self.dof_pos_target_legs = torch.zeros(self.num_envs, 10,
                                               dtype=torch.float,
                                               device=self.device)
        self.dof_pos_target_arms = torch.zeros(self.num_envs, 8,
                                               dtype=torch.float,
                                               device=self.device)
        self.dof_pos_legs = torch.zeros(self.num_envs, 10,
                                        dtype=torch.float, device=self.device)
        self.dof_pos_arms = torch.zeros(self.num_envs, 8,
                                        dtype=torch.float, device=self.device)
        self.dof_vel_legs = torch.zeros(self.num_envs, 10,
                                        dtype=torch.float, device=self.device)
        self.dof_vel_arms = torch.zeros(self.num_envs, 8,
                                        dtype=torch.float, device=self.device)

        # * other
        self.base_pos = self.root_states[:, 0:3]
        self.phase = torch.zeros(self.num_envs, 1,
                                 dtype=torch.float, device=self.device)
        self.phase_freq = 1.

        # * high level
        self.hl_impulses = torch.zeros(self.num_envs, 4, 5,
            dtype=torch.float, device=self.device)
        self.hl_ix = torch.zeros(self.num_envs, 1,
                                 dtype=torch.int64,
                                 device=self.device)
        self.hl_commands = torch.zeros(self.num_envs, 6,
            dtype=torch.float, device=self.device)

    def _pre_physics_step(self):
        super()._pre_physics_step()
        self.dof_pos_target[:, :10] += self.dof_pos_target_legs
        self.dof_pos_target[:, 10:] += self.dof_pos_target_arms

    def _reset_system(self, env_ids):
        super()._reset_system(env_ids)
        # if self.cfg.commands.resampling_time == -1:
        #     self.commands[env_ids, :] = 0.
        self.phase[env_ids, 0] = torch.rand(
            (torch.numel(env_ids),), requires_grad=False, device=self.device)

    def _post_physics_step(self):
        super()._post_physics_step()
        self.phase_sin = torch.sin(2 * torch.pi * self.phase)
        self.phase_cos = torch.cos(2 * torch.pi * self.phase)

        self.in_contact = \
            self.contact_forces[:, self.end_effector_ids, 2].gt(0.)
        self.phase = torch.fmod(self.phase + self.dt, 1.0)

        self.dof_pos_legs = self.dof_pos[:, :10]
        self.dof_pos_arms = self.dof_pos[:, 10:]
        self.dof_vel_legs = self.dof_vel[:, :10]
        self.dof_vel_arms = self.dof_vel[:, 10:]

        # * end_effector_pos is world-frame and converted to env_origin
        self.end_effector_pos = (
            self._rigid_body_pos[:, self.end_effector_ids]
            - self.env_origins.unsqueeze(dim=1).expand(
                self.num_envs, len(self.end_effector_ids), 3))
        self.end_effector_quat = \
            self._rigid_body_quat[:, self.end_effector_ids]

        # * end_effector vels are body-relative like body vels above
        for index in range(len(self.end_effector_ids)):
            self.end_effector_lin_vel[:, index, :] = quat_rotate_inverse(
                self.base_quat,
                self._rigid_body_lin_vel[:, self.end_effector_ids]
                                        [:, index, :])
            self.end_effector_ang_vel[:, index, :] = quat_rotate_inverse(
                self.base_quat,
                self._rigid_body_ang_vel[:, self.end_effector_ids]
                                        [:, index, :])
            
        # * update hl command based on current robot state and time
        envs_to_resample = torch.where(self.episode_length_buf % 5 == 0, True, False) # replace magic # w/ config ref
        if envs_to_resample.any().item():
            self._resample_high_level(envs_to_resample.nonzero(as_tuple=False).flatten())
        ix_next_impulse = torch.clamp(self.hl_ix + 1, min = 0, max = self.hl_impulses[0, 0, :].shape[0] - 1)
        t_next_impulse = torch.gather(self.hl_impulses[:, 0, :], dim=1, index=ix_next_impulse)
        ix_to_increment = torch.ge(self.episode_length_buf.view(-1, 1), t_next_impulse)
        self.hl_ix += ix_to_increment
        delta_t = self.episode_length_buf.view(-1, 1) - torch.gather(self.hl_impulses[:, 0, :], dim=1, index=self.hl_ix)
        
        
        hl_vel_xy = self.hl_impulses[:, 1:3, :].gather(dim=2, index=self.hl_ix.expand(-1, 2).unsqueeze(2))
        self.hl_commands[:, 3:5] = self.hl_commands[:, 3:5] + hl_vel_xy.squeeze(2)

        hl_vel_z = self.hl_impulses[:, -1, :].gather(dim=1, index=self.hl_ix)
        self.hl_commands[:, 5] = self.hl_commands[:, 5] - 9.81*delta_t.squeeze(1) + hl_vel_z.squeeze(1)

        self.hl_commands[:, 0:2] += delta_t.expand(-1, 2) *self.hl_commands[:, 3:5]
        print("self.hl_commands shape", self.hl_commands[:, -1].shape)
        print("delta_t.shape", delta_t.squeeze(1).shape)
        self.hl_commands[:, 2] += -0.5*9.81*torch.square(delta_t.squeeze(1)) + delta_t.squeeze(1)*self.hl_commands[:, -1]
        print(self.hl_commands[0, :])
        
    def _resample_high_level(self, envs):
        impulse_mag = torch.cat((torch.tensor([[1], [0], [5]],
                               device=self.device),
                               torch.tensor([[1], [0], [10]],
                               device=self.device).expand(-1, 4)),
                               dim=1).expand(self.num_envs, -1, -1)
        
        print('base height shape', self.base_height.shape)
        print("base lin vel ", self.base_lin_vel[0, :])
        
        delta_t = self.time_to_touchdown(self.base_height, self.base_lin_vel[:, 2].unsqueeze(1), -0.5*9.81)
        print("delta t shape", delta_t.shape)

        print("delta_t", delta_t)
        exit()
        
        impulse_mag[:, :2, 0] += self.base_lin_vel[:, :2] * delta_t.expand(-1, 2) + self.root_states[:, :2]
        impulse_mag[:, 2, 0] += -0.5*9.81*torch.square(delta_t) + self.base_lin_vel[:, 2]*delta_t + self.base_height

        self.hl_impulses = torch.cat((torch.arange(delta_t, delta_t + (5.0/self.dt), device=self.device).expand(envs.sum().item(), -1, -1), impulse_mag), dim=2)
        print(self.hl_impulses.shape)
        exit()
        self.hl_ix = 0
        self.hl_commands = torch.zeros(self.num_envs, 6,
            dtype=torch.float, device=self.device)
    
    def time_to_touchdown(self, pos, vel, acc):
        """ Assumes robot COM in projectile motion to calculate next touchdown
        """
        print("pos shape", pos.shape)
        print("vel shape", vel.shape)
        determinant = torch.square(vel) + 4*pos*acc
        print("determinant", determinant)
        solution = torch.where(determinant <= 0., False, True)
        print("solution", solution)
        print("solution any", solution.any())
        # assert torch.all(solution).item(), "No solution determinant in projectile t_final search" # reconsider assert
        t1 = (-vel + torch.sqrt(determinant)) / (2*acc)
        t2 = (-vel - torch.sqrt(determinant)) / (2*acc)
        print("t1 ", t1)
        print("t2 ", t2)
        return torch.where(t2 > t1, t2, t1)

    def _check_terminations_and_timeouts(self):
        """ Check if environments need to be reset
        """
        super()._check_terminations_and_timeouts()

        # * Termination for velocities, orientation, and low height
        self.terminated |= \
            (self.base_lin_vel.norm(dim=-1, keepdim=True) > 10).any(dim=1)
        self.terminated |= \
            (self.base_ang_vel.norm(dim=-1, keepdim=True) > 5).any(dim=1)
        self.terminated |= \
            (self.projected_gravity[:, 0:1].abs() > 0.7).any(dim=1)
        self.terminated |= \
            (self.projected_gravity[:, 1:2].abs() > 0.7).any(dim=1)
        self.terminated |= (self.base_pos[:, 2:3] < 0.3).any(dim=1)

        self.to_be_reset = self.timed_out | self.terminated

# ########################## REWARDS ######################## #

    # * Task rewards * #

    def _reward_tracking_lin_vel(self):
        error = self.commands[:, :2] - self.base_lin_vel[:, :2]
        error *= 2. / (1. + torch.abs(self.commands[:, :2]))
        return self._sqrdexp(error).sum(dim=1)

    def _reward_tracking_ang_vel(self):
        ang_vel_error = (self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error /= self.scales["base_ang_vel"]
        return self._sqrdexp(ang_vel_error / torch.pi)

    # * Shaping rewards * #

    def _reward_base_height(self):
        error = self.base_height - self.cfg.reward_settings.base_height_target
        error /= self.scales['base_height']
        error = error.flatten()
        return self._sqrdexp(error)

    def _reward_orientation(self):
        return self._sqrdexp(self.projected_gravity[:, 2] + 1)

    def _reward_joint_regularization_legs(self):
        # * Reward joint poses and symmetry
        reward = self._reward_hip_yaw_zero()
        reward += self._reward_hip_abad_symmetry()
        reward += self._reward_hip_pitch_symmetry()
        return reward / 3.

    def _reward_hip_yaw_zero(self):
        error = self.dof_pos[:, 0] - self.default_dof_pos[:, 0]
        reward = self._sqrdexp(error / self.scales['dof_pos'][0]) / 2.
        error = self.dof_pos[:, 5] - self.default_dof_pos[:, 5]
        reward += self._sqrdexp(error / self.scales['dof_pos'][5]) / 2.
        return reward

    def _reward_hip_abad_symmetry(self):
        error = (self.dof_pos[:, 1] / self.scales['dof_pos'][1]
                 - self.dof_pos[:, 6] / self.scales['dof_pos'][6])
        return self._sqrdexp(error)

    def _reward_hip_pitch_symmetry(self):
        error = (self.dof_pos[:, 2] / self.scales['dof_pos'][2]
                 + self.dof_pos[:, 7] / self.scales['dof_pos'][7])
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
        return reward / 6.

    def _reward_arm_pitch_symmetry(self):
        error = (self.dof_pos[:, 10] / self.scales['dof_pos'][10]
                 + self.dof_pos[:, 14] / self.scales['dof_pos'][14])
        return self._sqrdexp(error)

    def _reward_arm_pitch_zero(self):
        error = self.dof_pos[:, 10] - self.default_dof_pos[:, 10]
        reward = self._sqrdexp(error / self.scales['dof_pos'][10])
        error = self.dof_pos[:, 14] - self.default_dof_pos[:, 14]
        reward += self._sqrdexp(error / self.scales['dof_pos'][14])
        return reward / 2.

    def _reward_elbow_symmetry(self):
        error = (self.dof_pos[:, 13] / self.scales['dof_pos'][13]
                 + self.dof_pos[:, 17] / self.scales['dof_pos'][17])
        return self._sqrdexp(error)

    def _reward_elbow_zero(self):
        error = self.dof_pos[:, 13] - self.default_dof_pos[:, 13]
        reward = self._sqrdexp(error / self.scales['dof_pos'][13])
        error = self.dof_pos[:, 17] - self.default_dof_pos[:, 17]
        reward += self._sqrdexp(error / self.scales['dof_pos'][17])
        return reward / 2.

    def _reward_arm_yaw_symmetry(self):
        error = (self.dof_pos[:, 12] / self.scales['dof_pos'][12]
                 - self.dof_pos[:, 16] / self.scales['dof_pos'][16])
        return self._sqrdexp(error)

    def _reward_arm_yaw_zero(self):
        error = self.dof_pos[:, 12] - self.default_dof_pos[:, 12]
        reward = self._sqrdexp(error / self.scales['dof_pos'][12])
        error = self.dof_pos[:, 16] - self.default_dof_pos[:, 16]
        reward += self._sqrdexp(error / self.scales['dof_pos'][16])
        return reward / 2.

    def _reward_arm_abad_symmetry(self):
        error = (self.dof_pos[:, 11] / self.scales['dof_pos'][11]
                 - self.dof_pos[:, 15] / self.scales['dof_pos'][15])
        return self._sqrdexp(error)

    def _reward_arm_abad_zero(self):
        error = self.dof_pos[:, 11] - self.default_dof_pos[:, 11]
        reward = self._sqrdexp(error / self.scales['dof_pos'][11])
        error = self.dof_pos[:, 15] - self.default_dof_pos[:, 15]
        reward += self._sqrdexp(error / self.scales['dof_pos'][15])
        return reward / 2.
