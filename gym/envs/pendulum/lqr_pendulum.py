import torch
import numpy as np
import numpy.linalg as linalg
import scipy

from torch.func import jacrev
from gym.envs.pendulum.pendulum import Pendulum

from learning.utils.logger.TimeKeeper import TimeKeeper

class LQRPendulum(Pendulum):

    def dynamics(self, theta, theta_dot, u):
        m = self.cfg.asset.mass
        b = self.cfg.asset.joint_damping
        length = self.cfg.asset.length
        g = 9.81
        ml2 = m*length**2
        theta_ddot = (1.0/ml2)*(u-b*theta_dot - m*g*length*torch.sin(theta + (torch.pi/2.0)))
        # theta_ddot = (1.0/ml2)*(u-b*theta_dot - m*g*length*torch.sin(theta))
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
        timer = TimeKeeper()
        timer.tic("Entered compute_torques()")
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
        x_d = torch.zeros(self.dof_state.shape, device=self.device)
        u_d = torch.zeros(self.torques.shape, device=self.device)
        Q = torch.eye(self.dof_state.shape[1], device=self.device)
        Q[0, 0] = 10.0
        Q = Q.repeat(self.num_envs, 1, 1)
        R = torch.eye(self.torques.shape[1], device=self.device).repeat(self.num_envs, 1, 1)
        torques = torch.zeros_like(self.torques)
        # torques = vectorized_lqr(Q.cpu(), R.cpu(),
        #                          torch.zeros(self.dof_state.shape).cpu(),
        #                          torch.zeros(self.torques.shape).cpu())
        for env in range(self.num_envs):
            u_prime = self.lqr(Q[env],
                               R[env],
                               self.dof_state[env],
                               x_d[env],
                               u_d[env])
            torques[env] = torch.from_numpy(u_prime)

        torques = torch.clip(torques, -self.torque_limits, self.torque_limits)
        if torch.any(torques != torch.zeros_like(torques)):
            print("torques", torques)
        timer.toc("Entered compute_torques()")
        # print(f"Time spent computing torques {timer.get_time('Entered compute_torques()')}")
        return torques.view(self.torques.shape)

    def lqr(self, Q, R, x, x_d, u_d):
        # A, B = self.linearize_dynamics(self.dynamics, x_d, u_d)
        A, B = self.linearize_pendulum_dynamics(x_d)
        A = A.cpu().detach().numpy()
        B = B.cpu().detach().numpy()
        Q = Q.cpu().detach().numpy()
        R = R.cpu().detach().numpy()
        x = x.cpu().detach().numpy()
        x_d = x_d.cpu().detach().numpy()
        u_d = u_d.cpu().detach().numpy()
        S = scipy.linalg.solve_continuous_are(A, B, Q, R)
        # S = scipy.linalg.solve_discrete_are(A, B, Q, R)
        B_T = B.transpose(-1, -2)
        x_bar = (x - x_d)
        u_prime = u_d - linalg.inv(R)@B_T@S@x_bar
        return u_prime

    def linearize_pendulum_dynamics(self, x_d):
        m = self.cfg.asset.mass
        b = self.cfg.asset.joint_damping
        length = self.cfg.asset.length
        g = 9.81
        ml2 = m*length**2
        A = torch.tensor([[0.0, 1.0],
                          [(1.0/ml2)*(-m*g*length*torch.cos(x_d[0] + (torch.pi/2.0))), -(b/ml2)]])
        # A = torch.tensor([[0.0, 1.0],
        #                   [(1.0/ml2)*(-m*g*length*torch.cos(x_d[0])), -(b/ml2)]])
        B = torch.tensor([[0.0],
                          [(1.0/ml2)]])
        return A, B


    # ! address zeroing out
    # def linearize_dynamics(self, dynamics, x_d, u_d):
    #     A = torch.vstack(jacrev(dynamics, argnums=(0, 1))(*x_d, u_d))
    #     B = jacrev(dynamics, argnums=(len(x_d) + len(u_d) - 1))(*x_d, u_d)
    #     return A, B

    # def detach_numpy(self, tensor):
    #     tensor = tensor.detach().cpu()
    #     # if torch._C._functorch.is_gradtrackingtensor(tensor):
    #     #     tensor = torch._C._functorch.get_unwrapped(tensor)
    #     #     return np.array(tensor.storage().tolist()).reshape(tensor.shape)
    #     tensor = torch._C._functorch.get_unwrapped(tensor)
    #     tensor = np.array(tensor.storage().tolist()).reshape(tensor.shape)
    #     return tensor
