from math import sqrt
import torch

from gym.envs.base.fixed_robot import FixedRobot


class Pendulum(FixedRobot):
    def _post_physics_step(self):
        """Update all states that are not handled in PhysX"""
        super()._post_physics_step()

    def _check_terminations_and_timeouts(self):
        super()._check_terminations_and_timeouts()
        self.terminated = self.timed_out

    def reset_to_uniform(self, env_ids):
        grid_points = int(sqrt(self.num_envs))
        lin_pos = torch.linspace(
            self.dof_pos_range[0, 0],
            self.dof_pos_range[0, 1],
            grid_points,
            device=self.device,
        )
        lin_vel = torch.linspace(
            self.dof_vel_range[0, 0],
            self.dof_vel_range[0, 1],
            grid_points,
            device=self.device,
        )
        grid = torch.cartesian_prod(lin_pos, lin_vel)
        self.dof_pos[env_ids] = grid[:, 0].unsqueeze(-1)
        self.dof_vel[env_ids] = grid[:, 1].unsqueeze(-1)

    def _reward_theta(self):
        theta_rwd = torch.cos(self.dof_pos[:, 0]) / self.scales["dof_pos"]
        return self._sqrdexp(theta_rwd.squeeze(dim=-1))

    def _reward_omega(self):
        omega_rwd = torch.square(self.dof_vel[:, 0] / self.scales["dof_vel"])
        return self._sqrdexp(omega_rwd.squeeze(dim=-1))

    def _reward_equilibrium(self):
        error = torch.abs(self.dof_state)
        error[:, 0] /= self.scales["dof_pos"]
        error[:, 1] /= self.scales["dof_vel"]
        return self._sqrdexp(torch.mean(error, dim=1), sigma=0.01)
        # return torch.exp(
        #     -error.pow(2).sum(dim=1) / self.cfg.reward_settings.tracking_sigma
        # )

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
