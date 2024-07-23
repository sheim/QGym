import torch


class MinimalistCheetah:
    """
    Helper class for computing mini cheetah rewards
    """

    def __init__(
        self, device="cpu", tracking_sigma=0.25, ctrl_dt=0.01, ctrl_decimation=5
    ):
        self.device = device
        self.tracking_sigma = tracking_sigma

        # Implemented as in legged robot action rate reward
        self.dt = ctrl_dt * ctrl_decimation

        # Default joint angles from mini_cheetah_config.py
        self.default_dof_pos = torch.tensor(
            [0.0, -0.785398, 1.596976], device=self.device
        ).repeat(4)

        # Scales
        self.command_scales = torch.tensor([3.0, 1.0, 3.0]).to(self.device)

        # Previous 2 dof pos targets
        self.dof_target_prev = None
        self.dof_target_prev2 = None

    def set_states(
        self,
        base_height,
        base_lin_vel,
        base_ang_vel,
        proj_gravity,
        commands,
        dof_pos_obs,
        dof_vel,
        phase_obs,
        grf,
        dof_pos_target,
    ):
        # Unsqueeze so first dim is batch_size
        self.base_height = torch.tensor(base_height, device=self.device).unsqueeze(0)
        self.base_lin_vel = torch.tensor(base_lin_vel, device=self.device).unsqueeze(0)
        self.base_ang_vel = torch.tensor(base_ang_vel, device=self.device).unsqueeze(0)
        self.proj_gravity = torch.tensor(proj_gravity, device=self.device).unsqueeze(0)
        self.commands = (
            torch.tensor(commands, device=self.device).unsqueeze(0)
            * self.command_scales
        )
        self.dof_pos_obs = torch.tensor(dof_pos_obs, device=self.device).unsqueeze(0)
        self.dof_vel = torch.tensor(dof_vel, device=self.device).unsqueeze(0)
        self.grf = torch.tensor(grf, device=self.device).unsqueeze(0)

        # Compute phase sin
        phase_obs = torch.tensor(phase_obs, device=self.device).unsqueeze(0)
        self.phase_sin = phase_obs[:, 0].unsqueeze(0)

        # Set targets
        self.dof_pos_target = torch.tensor(
            dof_pos_target, device=self.device
        ).unsqueeze(0)
        if self.dof_target_prev is None:
            self.dof_target_prev = self.dof_pos_target
        if self.dof_target_prev2 is None:
            self.dof_target_prev2 = self.dof_target_prev

    def post_process(self):
        self.dof_target_prev2 = self.dof_target_prev
        self.dof_target_prev = self.dof_pos_target

    def _sqrdexp(self, x, scale=1.0):
        """shorthand helper for squared exponential"""
        return torch.exp(-torch.square(x / scale) / self.tracking_sigma)

    def _switch(self, scale=0.1):
        # TODO: Check scale, RS commands are scaled differently than QGym
        c_vel = torch.linalg.norm(self.commands, dim=1)
        return torch.exp(
            -torch.square(torch.max(torch.zeros_like(c_vel), c_vel - 0.1)) / scale
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
        return torch.exp(-error / self.tracking_sigma) * (1 - self._switch())

    def _reward_tracking_ang_vel(self):
        """Tracking of angular velocity commands (yaw)"""
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.tracking_sigma)

    def _reward_orientation(self):
        """Penalize non-flat base orientation"""
        error = torch.square(self.proj_gravity[:, :2]) / self.tracking_sigma
        return torch.sum(torch.exp(-error), dim=1)

    def _reward_swing_grf(self, contact_thresh=50 / 80):
        """Reward non-zero grf during swing (0 to pi)"""
        in_contact = torch.gt(self.grf, contact_thresh)
        ph_off = torch.gt(self.phase_sin, 0)  # phase <= pi
        rew = in_contact * torch.cat((ph_off, ~ph_off, ~ph_off, ph_off), dim=1)
        return -torch.sum(rew.float(), dim=1) * (1 - self._switch())

    def _reward_stance_grf(self, contact_thresh=50 / 80):
        """Reward non-zero grf during stance (pi to 2pi)"""
        in_contact = torch.gt(self.grf, contact_thresh)
        ph_off = torch.lt(self.phase_sin, 0)  # phase >= pi
        rew = in_contact * torch.cat((ph_off, ~ph_off, ~ph_off, ph_off), dim=1)
        return torch.sum(rew.float(), dim=1) * (1 - self._switch())

    def _reward_stand_still(self):
        """Penalize motion at zero commands"""
        # * normalize angles so we care about being within 5 deg
        rew_pos = torch.mean(
            self._sqrdexp((self.dof_pos_obs) / torch.pi * 36),
            dim=1,
        )
        rew_vel = torch.mean(self._sqrdexp(self.dof_vel), dim=1)
        rew_base_vel = torch.mean(torch.square(self.base_lin_vel), dim=1)
        rew_base_vel += torch.mean(torch.square(self.base_ang_vel), dim=1)
        return (rew_vel + rew_pos - rew_base_vel) * self._switch()

    def _reward_action_rate(self):
        """Penalize changes in actions"""
        # TODO: check this
        error = torch.square(self.dof_pos_target - self.dof_target_prev) / self.dt**2
        return -torch.sum(error, dim=1)

    def _reward_action_rate2(self):
        """Penalize changes in actions"""
        # TODO: check this
        error = (
            torch.square(
                self.dof_pos_target - 2 * self.dof_target_prev + self.dof_target_prev2
            )
            / self.dt**2
        )
        return -torch.sum(error, dim=1)
