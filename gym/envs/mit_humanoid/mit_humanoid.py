import torch

from gym.envs.base.legged_robot import LeggedRobot
from .jacobian import _apply_coupling
from gym.utils import exp_avg_filter


class MIT_Humanoid(LeggedRobot):
    def __init__(self, gym, sim, cfg, sim_params, sim_device, headless):
        super().__init__(gym, sim, cfg, sim_params, sim_device, headless)

    def _init_buffers(self):
        super()._init_buffers()
        self.oscillators = torch.zeros(self.num_envs, 2, device=self.device)
        self.oscillator_obs = torch.zeros(self.num_envs, 4, device=self.device)
        self.oscillator_freq = torch.zeros(self.num_envs, 2, device=self.device)
        self._init_sampled_history_buffers()

    def _init_sampled_history_buffers(self):
        self.sampled_history_dof_pos_target = torch.zeros(
            (self.num_envs, self.num_dof * self.cfg.env.sampled_history_length),
            device=self.device,
        )
        self.sampled_history_dof_pos = torch.zeros(
            self.num_envs,
            self.num_dof * self.cfg.env.sampled_history_length,
            device=self.device,
        )
        self.sampled_history_dof_vel = torch.zeros(
            self.num_envs,
            self.num_dof * self.cfg.env.sampled_history_length,
            device=self.device,
        )
        self.sampled_history_counter = torch.zeros(
            self.num_envs, dtype=int, device=self.device
        )
        self.sampled_history_threshold = int(
            self.cfg.control.ctrl_frequency / self.cfg.env.sampled_history_frequency
        )

    def _reset_system(self, env_ids):
        if len(env_ids) == 0:
            return
        super()._reset_system(env_ids)
        # reset oscillators, with a pi phase shift between left and right
        self.oscillators[env_ids, 0] = (
            torch.rand(len(env_ids), device=self.device) * 2 * torch.pi
        )
        self.oscillators[env_ids, 1] = self.oscillators[env_ids, 0] + torch.pi
        self.oscillators = torch.remainder(self.oscillators, 2 * torch.pi)
        # reset oscillator velocities to base freq
        self.oscillator_freq[env_ids] = self.cfg.oscillator.base_frequency
        # recompute oscillator observations
        self.oscillator_obs = torch.cat(
            (self.oscillators.cos(), self.oscillators.sin()), dim=1
        )
        self._reset_sampled_history_buffers(env_ids)
        return

    def _reset_sampled_history_buffers(self, ids):
        n = self.cfg.env.sampled_history_length
        self.sampled_history_dof_pos_target[ids] = self.dof_pos_target[ids].tile(n)
        self.sampled_history_dof_pos[ids] = self.dof_pos_target[ids].tile(n)
        self.sampled_history_dof_vel[ids] = self.dof_pos_target[ids].tile(n)

    # compute_torques accounting for coupling, and filtering torques
    def _compute_torques(self):
        torques = _apply_coupling(
            self.dof_pos,
            self.dof_vel,
            self.dof_pos_target + self.default_dof_pos,
            self.dof_vel_target,
            self.p_gains,
            self.d_gains,
            self.tau_ff,
        )
        torques = torques.clip(-self.torque_limits, self.torque_limits)
        return exp_avg_filter(torques, self.torques, self.cfg.control.filter_gain)

    # oscillator integration

    def _post_decimation_step(self):
        super()._post_decimation_step()
        self._step_oscillators()
        self._update_sampled_history_buffers()

    def _step_oscillators(self, dt=None):
        if dt is None:
            dt = self.dt
        self.oscillators += (self.oscillator_freq * 2 * torch.pi) * dt
        self.oscillators = torch.remainder(self.oscillators, 2 * torch.pi)
        self.oscillator_obs = torch.cat(
            (torch.cos(self.oscillators), torch.sin(self.oscillators)), dim=1
        )

    def _update_sampled_history_buffers(self):
        self.sampled_history_counter += 1

        ids = torch.nonzero(
            self.sampled_history_counter == self.sampled_history_threshold,
            as_tuple=False,
        ).flatten()

        self.sampled_history_dof_pos_target[ids].roll(self.num_dof, dims=1)  # check
        self.sampled_history_dof_pos_target[ids, : self.num_dof] = self.dof_pos_target[
            ids
        ]
        self.sampled_history_dof_pos[ids].roll(self.num_dof, dims=1)  # check
        self.sampled_history_dof_pos[ids, : self.num_dof] = self.dof_pos_target[ids]
        self.sampled_history_dof_vel[ids].roll(self.num_dof, dims=1)  # check
        self.sampled_history_dof_vel[ids, : self.num_dof] = self.dof_pos_target[ids]

        self.sampled_history_counter[ids] = 0

    # --- rewards ---

    def _switch(self, mode=None):
        c_vel = torch.linalg.norm(self.commands, dim=1)
        switch = torch.exp(
            -torch.square(
                torch.max(
                    torch.zeros_like(c_vel),
                    c_vel - self.cfg.reward_settings.switch_scale,
                )
            )
            / self.cfg.reward_settings.switch_scale
        )
        if mode is None or mode == "stand":
            return switch
        elif mode == "move":
            return 1 - switch

    def _reward_lin_vel_xy(self):
        return torch.exp(
            -torch.linalg.norm(self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        )

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity w. squared exp
        return self._sqrdexp(self.base_lin_vel[:, 2] / self.scales["base_lin_vel"])

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(
            self._sqrdexp(torch.square(self.projected_gravity[:, :2])), dim=1
        )

    def _reward_min_base_height(self):
        """Squared exponential saturating at base_height target"""
        error = self.base_height - self.cfg.reward_settings.base_height_target
        error = torch.clamp(error, max=0, min=None).flatten()
        return self._sqrdexp(error)

    def _reward_tracking_lin_vel(self):
        """Tracking of linear velocity commands (xy axes)"""
        # just use lin_vel?
        error = self.commands[:, :2] - self.base_lin_vel[:, :2]
        # * scale by (1+|cmd|): if cmd=0, no scaling.
        error *= 1.0 / (1.0 + torch.abs(self.commands[:, :2]))
        return torch.mean(self._sqrdexp(error), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.mean(
            self._sqrdexp(self.dof_vel / self.scales["dof_vel"]), dim=1
        ) * self._switch("stand")

    def _reward_dof_near_home(self):
        return torch.mean(
            self._sqrdexp(
                (self.dof_pos - self.default_dof_pos) / self.scales["dof_pos"]
            ),
            dim=1,
        )

    def _reward_stand_still(self):
        """Penalize motion at zero commands"""
        # * normalize angles so we care about being within 5 deg
        rew_pos = torch.mean(
            self._sqrdexp((self.dof_pos - self.default_dof_pos) / torch.pi * 36), dim=1
        )
        rew_vel = torch.mean(self._sqrdexp(self.dof_vel), dim=1)
        rew_base_vel = torch.mean(torch.square(self.base_lin_vel), dim=1)
        rew_base_vel += torch.mean(torch.square(self.base_ang_vel), dim=1)
        return rew_vel + rew_pos - rew_base_vel * self._switch("stand")

    def _reward_stance(self):
        # phase = torch.maximum(
        #     torch.zeros_like(self.oscillators), -self.oscillators.sin()
        # )  # positive during swing, negative during stance
        phase = self.smooth_sqr_wave(self.oscillators)
        return (phase * self._compute_grf()).mean(dim=1)

    def _reward_swing(self):
        # phase = torch.maximum(
        #     torch.zeros_like(self.oscillators), self.oscillators.sin()
        # )  # positive during swing, negative during stance
        phase = self.smooth_sqr_wave(self.oscillators + torch.pi)
        return -(phase * self._compute_grf()).mean(dim=1)

    def _compute_grf(self, grf_norm=True):
        grf = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        if grf_norm:
            return torch.clamp_max(grf / self.cfg.asset.total_mass, 1.0)
        else:
            return grf

    def smooth_sqr_wave(self, phase, sigma=0.2):  # sigma=0 is step function
        return phase.sin() / (2 * torch.sqrt(phase.sin() ** 2.0 + sigma**2.0)) + 0.5

    def _reward_walk_freq(self):
        # Penalize deviation from base frequency
        return torch.mean(
            self._sqrdexp(
                (self.oscillator_freq - self.cfg.oscillator.base_frequency)
                / self.cfg.oscillator.base_frequency
            ),
            dim=1,
        ) * self._switch("move")

    def _reward_hips_forward(self):
        # reward hip motors for pointing forward
        hip_yaw_abad = torch.stack((self.dof_pos[:, 0:2], self.dof_pos[:, 5:7]), dim=1)
        hip_yaw_abad -= torch.stack(
            (self.default_dof_pos[:, 0:2], self.default_dof_pos[:, 5:7]), dim=1
        )
        hip_yaw_abad /= torch.stack(
            (self.scales["dof_pos"][0:2], self.scales["dof_pos"][5:7]), dim=1
        )
        return self._sqrdexp(hip_yaw_abad).sum(dim=1).mean(dim=1)
