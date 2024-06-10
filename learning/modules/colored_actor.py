import torch
import torch.nn as nn
from torch.distributions import Normal
from pink import ColoredNoiseProcess

from .actor import Actor

from gym import LEGGED_GYM_ROOT_DIR


# The following implementation is based on the pinkNoise paper. See code:
# https://github.com/martius-lab/pink-noise-rl/blob/main/pink/sb3.py
class ColoredActor(Actor):
    def __init__(
        self,
        *args,
        num_envs,
        epsilon: float = 1e-6,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if "sample_freq" in kwargs["exploration"]:
            print("sample_freq is not used in ColoredActor")
        self.epsilon = epsilon
        self.log_std_init = kwargs["exploration"]["log_std_init"]
        self.beta = kwargs["exploration"]["beta"]

        self.num_envs = num_envs
        self.gen = [None] * self.num_envs
        horizon = 500  # This is the control frequency times the episode length
        for i in range(self.num_envs):
            self.gen[i] = ColoredNoiseProcess(
                beta=self.beta, size=(self.num_actions, horizon)
            )

        self.log_std = nn.Parameter(
            torch.ones(self.num_actions) * self.log_std_init, requires_grad=True
        )

    def update_distribution(self, observations):
        if self._normalize_obs:
            with torch.no_grad():
                observations = self.obs_rms(observations)
        # Get latent features and compute distribution
        mean_actions = self.NN(observations)
        action_std = torch.ones_like(mean_actions) * torch.exp(self.log_std)
        self.distribution = Normal(mean_actions, action_std)

    def act(self, observations):
        self.update_distribution(observations)
        cn_sample = self.num_envs * [torch.zeros(self.num_actions, 500)]
        for i in range(self.num_envs):
            cn_sample[i] = torch.tensor(self.gen[i].sample()).float()
        # Send cn_sample to the device
        cn_sample = torch.stack(cn_sample).to(self.log_std.device)
        mean = self.distribution.mean
        sample = mean + torch.exp(self.log_std) * cn_sample

        if self.debug:
            path = f"{LEGGED_GYM_ROOT_DIR}/plots/distribution_pink.csv"
            self.log_actions(mean[0][0], sample[0][0], path)
        return sample

    def act_inference(self, observations):
        if self._normalize_obs:
            with torch.no_grad():
                observations = self.obs_rms(observations)
        mean_actions = self.NN(observations)
        return mean_actions
