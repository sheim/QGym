import torch

from gym.envs.base.fixed_robot import FixedRobot


class Pendulum(FixedRobot):
    def _post_physics_step(self):
        """Update all states that are not handled in PhysX"""
        super()._post_physics_step()

    def _compute_torques(self):
        return self.tau_ff

    def _reward_theta(self):
        theta_rwd = torch.cos(self.dof_pos[:, 0])
        return theta_rwd.squeeze(dim=-1)

    def _reward_omega(self):
        omega_rwd = -torch.square(self.dof_vel[:, 0])
        return omega_rwd.squeeze(dim=-1)
