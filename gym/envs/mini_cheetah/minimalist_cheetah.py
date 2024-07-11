import torch


class MinimalistCheetah:
    """
    Helper class for computing mini cheetah rewards
    """

    def __init__(self, device="cpu", tracking_sigma=0.25):
        self.device = device
        self.tracking_sigma = tracking_sigma

    def set_states(
        self,
        base_height,
        base_lin_vel,
        base_ang_vel,
        proj_gravity,
        commands,
        phase_obs,
        grf,
    ):
        # Unsqueeze so first dim is batch_size
        self.base_height = torch.tensor(base_height, device=self.device).unsqueeze(0)
        self.base_lin_vel = torch.tensor(base_lin_vel, device=self.device).unsqueeze(0)
        self.base_ang_vel = torch.tensor(base_ang_vel, device=self.device).unsqueeze(0)
        self.proj_gravity = torch.tensor(proj_gravity, device=self.device).unsqueeze(0)
        self.commands = torch.tensor(commands, device=self.device).unsqueeze(0)
        self.grf = torch.tensor(grf, device=self.device).unsqueeze(0)

        phase_obs = torch.tensor(phase_obs, device=self.device).unsqueeze(0)
        self.phase_sin = phase_obs[:, 0].unsqueeze(0)

    def _sqrdexp(self, x, scale=1.0):
        """shorthand helper for squared exponential"""
        return torch.exp(-torch.square(x / scale) / self.tracking_sigma)

    def _switch(self, scale=0.5):
        # TODO: Check scale, RS commands are scaled differently than QGym
        c_vel = torch.linalg.norm(self.commands, dim=1)
        return torch.exp(
            -torch.square(torch.max(torch.zeros_like(c_vel), c_vel - scale)) / scale
        )

    def _reward_min_base_height(self, target_height=0.3, scale=0.3):
        """Squared exponential saturating at base_height target"""
        # TODO: Check scale
        error = (self.base_height - target_height) / scale
        error = torch.clamp(error, max=0, min=None).flatten()
        return self._sqrdexp(error)

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

    def _reward_swing_grf(self, contact_thresh=0.5):
        """Reward non-zero grf during swing (0 to pi)"""
        in_contact = torch.gt(self.grf, contact_thresh)
        ph_off = torch.gt(self.phase_sin, 0)  # phase <= pi
        rew = in_contact * torch.cat((ph_off, ~ph_off, ~ph_off, ph_off), dim=1)
        return -torch.sum(rew.float(), dim=1)  # * (1 - self._switch())

    def _reward_stance_grf(self, contact_thresh=0.5):
        """Reward non-zero grf during stance (pi to 2pi)"""
        in_contact = torch.gt(self.grf, contact_thresh)
        ph_off = torch.lt(self.phase_sin, 0)  # phase >= pi
        rew = in_contact * torch.cat((ph_off, ~ph_off, ~ph_off, ph_off), dim=1)
        return torch.sum(rew.float(), dim=1)  # * (1 - self._switch())
