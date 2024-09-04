import torch

# from gym.envs.base.legged_robot import LeggedRobot
from gym.envs.mit_humanoid.mit_humanoid import MIT_Humanoid
from isaacgym.torch_utils import torch_rand_float


class Lander(MIT_Humanoid):
    def __init__(self, gym, sim, cfg, sim_params, sim_device, headless):
        super().__init__(gym, sim, cfg, sim_params, sim_device, headless)

    def _resample_commands(self, env_ids):
        """Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        if len(env_ids) == 0:
            return
        super()._resample_commands(env_ids)
        # * with 75% chance, reset to 0
        self.commands[env_ids, :] *= (
            torch_rand_float(0, 1, (len(env_ids), 1), device=self.device).squeeze(1)
            < 0.25
        ).unsqueeze(1)

    # --- rewards ---

    def _switch(self, mode=None):
        c_vel = torch.linalg.norm(self.commands, dim=1)
        switch = torch.exp(
            -torch.square(
                torch.max(
                    torch.zeros_like(c_vel),
                    c_vel - self.cfg.reward_settings.switch_scale,
                )
            )
            / self.cfg.reward_settings.switch_scale
        )
        if mode is None or mode == "stand":
            return switch
        elif mode == "move":
            return 1 - switch

    def _reward_lin_vel_xy(self):
        return torch.exp(
            -torch.linalg.norm(self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        )

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity w. squared exp
        return self._sqrdexp(self.base_lin_vel[:, 2] / self.scales["base_lin_vel"])

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(
            self._sqrdexp(torch.square(self.projected_gravity[:, :2])), dim=1
        )

    def _reward_min_base_height(self):
        """Squared exponential saturating at base_height target"""
        error = self.base_height - self.cfg.reward_settings.base_height_target
        error = torch.clamp(error, max=0, min=None).flatten()
        return self._sqrdexp(error)

    def _reward_tracking_lin_vel(self):
        """Tracking of linear velocity commands (xy axes)"""
        # just use lin_vel?
        error = self.commands[:, :2] - self.base_lin_vel[:, :2]
        # * scale by (1+|cmd|): if cmd=0, no scaling.
        error *= 1.0 / (1.0 + torch.abs(self.commands[:, :2]))
        return torch.mean(self._sqrdexp(error), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.mean(
            self._sqrdexp(self.dof_vel / self.scales["dof_vel"]), dim=1
        ) * self._switch("stand")

    def _reward_dof_near_home(self):
        return self._sqrdexp(
            (self.dof_pos - self.default_dof_pos) / self.scales["dof_pos"]
        ).mean(dim=1)

    def _reward_stand_still(self):
        """Penalize motion at zero commands"""
        # * normalize angles so we care about being within 5 deg
        rew_pos = torch.mean(
            self._sqrdexp((self.dof_pos - self.default_dof_pos) / torch.pi * 36), dim=1
        )
        rew_vel = torch.mean(self._sqrdexp(self.dof_vel), dim=1)
        rew_base_vel = torch.mean(torch.square(self.base_lin_vel), dim=1)
        rew_base_vel += torch.mean(torch.square(self.base_ang_vel), dim=1)
        return rew_vel + rew_pos - rew_base_vel * self._switch("stand")

    def _compute_grf(self, grf_norm=True):
        grf = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        if grf_norm:
            return torch.clamp_max(grf / self.cfg.asset.total_mass, 1.0)
        else:
            return grf

    def smooth_sqr_wave(self, phase, sigma=0.2):  # sigma=0 is step function
        return phase.sin() / (2 * torch.sqrt(phase.sin() ** 2.0 + sigma**2.0)) + 0.5

    def _reward_hips_forward(self):
        # reward hip motors for pointing forward
        hip_yaw_abad = torch.cat((self.dof_pos[:, 0:2], self.dof_pos[:, 5:7]), dim=1)
        hip_yaw_abad -= torch.cat(
            (self.default_dof_pos[:, 0:2], self.default_dof_pos[:, 5:7]), dim=1
        )
        hip_yaw_abad /= torch.cat(
            (self.scales["dof_pos"][0:2], self.scales["dof_pos"][5:7])
        )
        return (hip_yaw_abad).pow(2).mean(dim=1)
        # return self._sqrdexp(hip_yaw_abad).sum(dim=1).mean(dim=1)

    def _reward_power(self):
        power = self.torques * self.dof_vel
        return power.pow(2).mean(dim=1)
