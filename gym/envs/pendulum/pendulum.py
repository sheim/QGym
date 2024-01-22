import torch

from gym.envs.base.fixed_robot import FixedRobot


class Pendulum(FixedRobot):
    def _post_physics_step(self):
        """Update all states that are not handled in PhysX"""
        super()._post_physics_step()

    def _compute_torques(self):
        return self.tau_ff

    def _reward_theta(self):
        theta_rwd = torch.cos(self.dof_pos[:, 0]) / self.scales["theta"]
        return theta_rwd.squeeze(dim=-1)

    def _reward_omega(self):
        omega_rwd = -torch.square(self.dof_vel[:, 0]) / self.scales["omega"]
        return omega_rwd.squeeze(dim=-1)

    def _reward_equilibrium(self):
        error = torch.abs(self.dof_state)
        error[:, 0] /= self.scales["theta"]
        error[:, 1] /= self.scales["omega"]
        return self._sqrdexp(torch.mean(error, dim=1))

    def _reward_energy(self):
        m_pendulum = 2.0
        l_pendulum = 0.75
        kinetic_energy = (
            0.5 * m_pendulum * l_pendulum**2 * torch.square(self.dof_vel[:, 0])
        )
        potential_energy = (
            m_pendulum * 9.81 * l_pendulum * torch.cos(self.dof_pos[:, 0])
        )
        desired_energy = m_pendulum * 9.81 * l_pendulum
        energy_error = kinetic_energy + potential_energy - desired_energy
        return self._sqrdexp(energy_error / desired_energy)
