import torch
import numpy.linalg as linalg
import scipy

from gym.envs.pendulum.pendulum import Pendulum


class LQRPendulum(Pendulum):
    def _compute_torques(self):
        """Compute torques from actions.
            Actions can be interpreted as position or velocity targets given
            to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs,
                even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # timer = TimeKeeper()
        # timer.tic("Entered compute_torques()")
        actuated_dof_pos = torch.zeros(
            self.num_envs, self.num_actuators, device=self.device
        )
        actuated_dof_vel = torch.zeros(
            self.num_envs, self.num_actuators, device=self.device
        )
        for dof_idx in range(self.num_dof):
            idx = 0
            if self.cfg.control.actuated_joints_mask[dof_idx]:
                actuated_dof_pos[:, idx] = self.dof_pos[:, dof_idx]
                actuated_dof_vel[:, idx] = self.dof_vel[:, dof_idx]
                idx += 1

        # vectorized_lqr = torch.vmap(self.lqr, in_dims=0, out_dims=0)
        x_desired = torch.zeros(self.dof_state.shape, device=self.device)
        u_desired = torch.zeros(self.torques.shape, device=self.device)
        Q = torch.eye(self.dof_state.shape[1], device=self.device)
        Q[0, 0] = 10.0
        Q = Q.repeat(self.num_envs, 1, 1)
        R = torch.eye(self.torques.shape[1], device=self.device).repeat(
            self.num_envs, 1, 1
        )
        torques = torch.zeros_like(self.torques)
        for env in range(self.num_envs):
            u_prime = self.lqr(
                Q[env], R[env], self.dof_state[env], x_desired[env], u_desired[env]
            )
            torques[env] = torch.from_numpy(u_prime)
        return torques.view(self.torques.shape)

    def lqr(self, Q, R, x, x_desired, u_desired):
        A, B = self.linearize_pendulum_dynamics(x_desired)
        A = A.cpu().detach().numpy()
        B = B.cpu().detach().numpy()
        Q = Q.cpu().detach().numpy()
        R = R.cpu().detach().numpy()
        x = x.cpu().detach().numpy()
        x_desired = x_desired.cpu().detach().numpy()
        u_desired = u_desired.cpu().detach().numpy()
        # S = scipy.linalg.solve_discrete_are(A, B, Q, R)
        S = scipy.linalg.solve_continuous_are(A, B, Q, R)
        B_T = B.transpose(-1, -2)
        x_bar = x - x_desired
        u_prime = u_desired - linalg.inv(R) @ B_T @ S @ x_bar
        return u_prime

    def linearize_pendulum_dynamics(self, x_desired):
        m = self.cfg.asset.mass
        b = self.cfg.asset.joint_damping
        length = self.cfg.asset.length
        g = 9.81
        ml2 = m * length**2

        A = torch.tensor([[0.0, 1.0], [g / length * torch.cos(x_desired[0]), -b / ml2]])
        B = torch.tensor([[0.0], [(1.0 / ml2)]])
        return A, B
