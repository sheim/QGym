import torch

from gym.envs.mini_cheetah.mini_cheetah_osc import MiniCheetahOsc

MINI_CHEETAH_WEIGHT = 8.292 * 9.81  # Weight of mini cheetah in Newtons


class HierarchOsc(MiniCheetahOsc):
    def __init__(self, gym, sim, cfg, sim_params, sim_device, headless):
        super().__init__(gym, sim, cfg, sim_params, sim_device, headless)
        self.process_noise_std = self.cfg.osc.process_noise_std

    def _init_buffers(self):
        super()._init_buffers()
        self.osc_action = torch.zeros(self.num_envs, 4, device=self.device)

    def _step_oscillators(self, dt=None):
        if dt is None:
            dt = self.dt

        local_feedback = self.osc_coupling * (
            torch.cos(self.oscillators) + self.osc_offset
        )
        grf = self._compute_grf()
        # ! only change is here
        self.oscillators_vel = self.osc_omega - grf * local_feedback + self.osc_action
        self.oscillators_vel += (
            torch.randn(self.oscillators_vel.shape, device=self.device)
            * self.cfg.osc.process_noise_std
        )

        self.oscillators_vel *= 2 * torch.pi
        self.oscillators += (
            self.oscillators_vel * dt
        )  # torch.clamp(self.oscillators_vel * dt, min=0)
        self.oscillators = torch.remainder(self.oscillators, 2 * torch.pi)
        self.oscillator_obs = torch.cat(
            (torch.cos(self.oscillators), torch.sin(self.oscillators)), dim=1
        )

    def _reward_osc_action_sqrdexp(self):
        # put a _sqrdexp so that oscillator_actions are encouraged to stay close to 0
        return self._sqrdexp(self.osc_action).mean(dim=1)

    def _reward_osc_action_squared(self):
        return -torch.mean(torch.square(self.osc_action), dim=1)
