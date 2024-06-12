import torch
import numpy as np

from gym.envs.base.fixed_robot import FixedRobot


class Pendulum(FixedRobot):
    def _init_buffers(self):
        super()._init_buffers()
        self.dof_pos_obs = torch.zeros(self.num_envs, 2, device=self.device)

    def _post_decimation_step(self):
        super()._post_decimation_step()
        self.dof_pos_obs = torch.cat([self.dof_pos.sin(), self.dof_pos.cos()], dim=1)

    def _reset_system(self, env_ids):
        super()._reset_system(env_ids)
        self.dof_pos_obs[env_ids] = torch.cat(
            [self.dof_pos[env_ids].sin(), self.dof_pos[env_ids].cos()], dim=1
        )

    def _reward_theta(self):
        theta_rwd = torch.cos(self.dof_pos[:, 0])  # no scaling
        return self._sqrdexp(theta_rwd.squeeze(dim=-1))

    def _reward_omega(self):
        omega_rwd = torch.square(self.dof_vel[:, 0] / self.scales["dof_vel"])
        return self._sqrdexp(omega_rwd.squeeze(dim=-1))

    def _reward_equilibrium(self):
        theta_norm = self._normalize_theta()
        omega = self.dof_vel[:, 0]
        error = torch.stack(
            [theta_norm / self.scales["dof_pos"], omega / self.scales["dof_vel"]], dim=1
        )
        return self._sqrdexp(torch.mean(error, dim=1), sigma=0.01)

    def _reward_torques(self):
        """Penalize torques"""
        return self._sqrdexp(torch.mean(torch.square(self.torques), dim=1), sigma=0.2)

    def _reward_energy(self):
        kinetic_energy = (
            0.5
            * self.cfg.asset.mass
            * self.cfg.asset.length**2
            * torch.square(self.dof_vel[:, 0])
        )
        potential_energy = (
            self.cfg.asset.mass
            * 9.81
            * self.cfg.asset.length
            * torch.cos(self.dof_pos[:, 0])
        )
        desired_energy = self.cfg.asset.mass * 9.81 * self.cfg.asset.length
        energy_error = kinetic_energy + potential_energy - desired_energy
        return self._sqrdexp(energy_error / desired_energy)

    def _normalize_theta(self):
        # normalize to range [-pi, pi]
        theta = self.dof_pos[:, 0]
        return ((theta + np.pi) % (2 * np.pi)) - np.pi
