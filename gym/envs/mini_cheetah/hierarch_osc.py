import torch

from gym.envs.mini_cheetah.mini_cheetah_osc import MiniCheetahOsc
from gym.utils import exp_avg_filter

MINI_CHEETAH_WEIGHT = 8.292 * 9.81  # Weight of mini cheetah in Newtons


class HierarchOsc(MiniCheetahOsc):
    def __init__(self, gym, sim, cfg, sim_params, sim_device, headless):
        super().__init__(gym, sim, cfg, sim_params, sim_device, headless)
        self.process_noise_std = self.cfg.osc.process_noise_std

    def _init_buffers(self):
        super()._init_buffers()
        self.osc_action = torch.zeros(self.num_envs, 4, device=self.device)
        self.osc_action_filtered = torch.zeros_like(self.osc_action)
        # self.osc_action_history = torch.zeros(self.num)

    def _pre_decimation_step(self):
        super()._pre_decimation_step()
        self.osc_omega.clamp(min=0, max=self.cfg.osc.omega_max)

    def _step_oscillators(self, dt=None):
        if dt is None:
            dt = self.dt

        local_feedback = self.osc_coupling * (
            torch.cos(self.oscillators) + self.osc_offset
        )
        grf = self._compute_grf()
        # ! only change is here
        self.osc_action_filtered = exp_avg_filter(
            self.osc_action, self.osc_action_filtered, self.cfg.osc.filter_coeff
        )
        self.oscillator_vel = (
            self.osc_omega - grf * local_feedback + self.osc_action_filtered
        )
        self.oscillator_vel += (
            torch.randn(self.oscillator_vel.shape, device=self.device)
            * self.cfg.osc.process_noise_std
        )

        self.oscillator_vel *= 2 * torch.pi
        self.oscillators += (
            self.oscillator_vel * dt
        )  # torch.clamp(self.oscillator_vel * dt, min=0)
        self.oscillators = torch.remainder(self.oscillators, 2 * torch.pi)
        self.oscillator_obs = torch.cat(
            (torch.cos(self.oscillators), torch.sin(self.oscillators)), dim=1
        )

    def _reward_osc_action_sqrdexp(self):
        # put a _sqrdexp so that oscillator_actions are encouraged to stay close to 0
        return self._sqrdexp(self.osc_action_filtered).mean(dim=1)

    def _reward_osc_action_squared(self):
        return -torch.mean(torch.square(self.osc_action_filtered), dim=1)

    def _reward_omega_in_range(self):
        out_of_limit = (self.osc_omega - self.cfg.osc.omega_max).clamp(min=0)
        out_of_limit -= self.osc_omega.clamp(max=0)
        return self._sqrdexp(out_of_limit).mean(dim=1)
