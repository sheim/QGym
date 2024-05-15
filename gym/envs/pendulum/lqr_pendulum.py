import torch
import scipy

from gym.envs.pendulum.pendulum import Pendulum


class LQRPendulum(Pendulum):
    def _init_buffers(self):
        super()._init_buffers()
        self.x_desired = torch.tensor([0.0, 0.0], device=self.device)
        self.u_desired = torch.zeros(self.torques.shape, device=self.device)
        self.A, self.B = self.linearize_pendulum_dynamics(self.x_desired)
        self.Q = torch.eye(self.dof_state.shape[1], device=self.device)
        self.Q[0, 0] = 10.0
        self.R = torch.eye(self.torques.shape[1], device=self.device)
        self.S = (
            torch.from_numpy(self.solve_ricatti(self.Q, self.R)).float().to(self.device)
        )

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

        torques = self.lqr_u_prime(
            self.S, self.R, self.dof_state, self.x_desired, self.u_desired
        )
        return torques.view(self.torques.shape)

    def solve_ricatti(self, Q, R):
        A = self.A.cpu().detach().numpy()
        B = self.B.cpu().detach().numpy()
        Q = Q.cpu().detach().numpy()
        R = R.cpu().detach().numpy()
        S = scipy.linalg.solve_continuous_are(A, B, Q, R)
        return S

    def lqr_u_prime(self, S, R, x, x_desired, u_desired):
        batch_envs = x.shape[0]
        B_T = self.B.transpose(-1, -2).expand(batch_envs, -1, -1)
        S = S.expand(batch_envs, -1, -1)
        R_inv = torch.linalg.inv(R.expand(batch_envs, -1, -1))
        x_bar = x - x_desired.expand(batch_envs, -1)
        K = torch.einsum(
            "...ij, ...jk -> ...ik",
            torch.einsum("...ij, ...jk -> ...ik", R_inv, B_T),
            S,
        )
        u_prime = u_desired - torch.einsum(
            "...ij, ...jk -> ...ik", K, x_bar.unsqueeze(-1)
        ).squeeze(-1)
        return u_prime

    def linearize_pendulum_dynamics(self, x_desired):
        m = self.cfg.asset.mass
        b = self.cfg.asset.joint_damping
        length = self.cfg.asset.length
        g = 9.81
        ml2 = m * length**2

        A = torch.tensor(
            [[0.0, 1.0], [g / length * torch.cos(x_desired[0]), -b / ml2]],
            device=self.device,
        )
        B = torch.tensor([[0.0], [(1.0 / ml2)]], device=self.device)
        return A, B
