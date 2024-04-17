import torch
import numpy as np
import torch.linalg as linalg
import scipy

from torch.func import jacrev
from gym.envs.pendulum.pendulum import Pendulum


class LQRPendulum(Pendulum):

    def dynamics(self, theta, theta_dot, u):
        m = self.cfg.asset.mass
        b = self.cfg.asset.joint_damping
        length = self.cfg.asset.length
        g = 9.81
        theta_ddot = (1.0/(m*length**2))*(u-b*theta_dot - m*g*length*torch.sin(theta))
        return torch.hstack((theta_dot, theta_ddot))

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

        vectorized_lqr = torch.vmap(self.lqr, in_dims=0, out_dims=0)
        Q = torch.eye(self.dof_state.shape[1]).repeat(self.num_envs, 1, 1)
        R = torch.eye(self.torques.shape[1]).repeat(self.num_envs, 1, 1)
        torques = vectorized_lqr(Q.cpu(), R.cpu(),
                                 torch.zeros(self.dof_state.shape).cpu(),
                                 torch.zeros(self.torques.shape).cpu())

        torques = torch.clip(torques, -self.torque_limits, self.torque_limits)
        return torques.view(self.torques.shape)

    def lqr(self, Q, R, x_d, u_d):
        A, B = self.linearize_dynamics(self.dynamics, x_d, u_d)
        A = self.detach_numpy(A)
        B = self.detach_numpy(B)
        Q = self.detach_numpy(Q)
        R = self.detach_numpy(R)
        x_d = self.detach_numpy(x_d)
        u_d = self.detach_numpy(u_d)
        S = scipy.linalg.solve_discrete_are(A, B, Q, R)
        B_T = B.transpose(-1, -2)
        x_bar = (self.dof_state - x_d)
        u_prime = u_d - linalg.inv(R + B_T@S@B)@B_T@S@A@x_bar
        return u_prime

    def linearize_dynamics(self, dynamics, x_d, u_d):
        A = jacrev(dynamics, argnums=0)(*x_d, u_d)
        B = jacrev(dynamics, argnums=-1)(*x_d, u_d)
        return A, B

    def detach_numpy(self, tensor):
        tensor = tensor.detach().cpu()
        # if torch._C._functorch.is_gradtrackingtensor(tensor):
        #     tensor = torch._C._functorch.get_unwrapped(tensor)
        #     return np.array(tensor.storage().tolist()).reshape(tensor.shape)
        tensor = torch._C._functorch.get_unwrapped(tensor)
        tensor = np.array(tensor.storage().tolist()).reshape(tensor.shape)
        return tensor
