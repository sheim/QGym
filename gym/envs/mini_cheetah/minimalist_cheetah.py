import torch


class MinimalistCheetah:
    """
    Helper class for computing mini cheetah rewards
    """

    def __init__(self, device="cpu", tracking_sigma=0.25):
        self.device = device
        self.tracking_sigma = tracking_sigma

    def set_states(
        self, base_height, base_lin_vel, base_ang_vel, proj_gravity, commands
    ):
        # Unsqueeze so first dim is batch_size
        self.base_height = torch.tensor(base_height, device=self.device).unsqueeze(0)
        self.base_lin_vel = torch.tensor(base_lin_vel, device=self.device).unsqueeze(0)
        self.base_ang_vel = torch.tensor(base_ang_vel, device=self.device).unsqueeze(0)
        self.proj_gravity = torch.tensor(proj_gravity, device=self.device).unsqueeze(0)
        self.commands = torch.tensor(commands, device=self.device).unsqueeze(0)

    def _reward_tracking_lin_vel(self):
        """Tracking of linear velocity commands (xy axes)"""
        # just use lin_vel?
        error = self.commands[:, :2] - self.base_lin_vel[:, :2]
        # * scale by (1+|cmd|): if cmd=0, no scaling.
        error *= 1.0 / (1.0 + torch.abs(self.commands[:, :2]))
        error = torch.sum(torch.square(error), dim=1)
        return torch.exp(-error / self.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        """Tracking of angular velocity commands (yaw)"""
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.tracking_sigma)

    def _reward_orientation(self):
        """Penalize non-flat base orientation"""
        error = torch.square(self.proj_gravity[:, :2]) / self.tracking_sigma
        return torch.sum(torch.exp(-error), dim=1)
