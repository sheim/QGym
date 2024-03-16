import torch

from gym.envs.base.legged_robot import LeggedRobot
from .jacobian import _apply_coupling
from gym.utils import exp_avg_filter


class MIT_Humanoid(LeggedRobot):
    def __init__(self, gym, sim, cfg, sim_params, sim_device, headless):
        super().__init__(gym, sim, cfg, sim_params, sim_device, headless)

    def _init_buffers(self):
        super()._init_buffers()

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
        return torch.mean(self._sqrdexp(self.dof_vel / self.scales["dof_vel"]), dim=1)

    def _reward_dof_near_home(self):
        return torch.mean(
            self._sqrdexp(
                (self.dof_pos - self.default_dof_pos) / self.scales["dof_pos_obs"]
            ),
            dim=1,
        )
