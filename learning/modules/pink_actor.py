import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from pink import ColoredNoiseProcess

from .actor import Actor

from gym import LEGGED_GYM_ROOT_DIR


# The following implementation is based on the pinkNoise paper. See code:
# https://github.com/martius-lab/pink-noise-rl/blob/main/pink/sb3.py
class PinkActor(Actor):
    def __init__(
        self,
        *args,
        epsilon: float = 1e-6,
        log_std_init: float = 0.0,
        beta=0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.log_std_init = log_std_init
        self.beta = beta

        # TODO[ni]: get control frequency and episode time from config
        self.gen = ColoredNoiseProcess(beta=self.beta, size=(self.num_actions, 500))

        self.log_std = nn.Parameter(
            torch.ones(self.num_actions) * log_std_init, requires_grad=True
        )

        # Debug mode for plotting
        self.debug = True

    def update_distribution(self, observations):
        if self._normalize_obs:
            observations = self.normalize(observations)
        # Get latent features and compute distribution
        mean_actions = self.NN(observations)
        action_std = torch.ones_like(mean_actions) * torch.exp(self.log_std)
        self.distribution = Normal(mean_actions, action_std)

    def act(self, observations):
        self.update_distribution(observations)
        if np.isscalar(self.beta):
            cn_sample = torch.tensor(self.gen.sample()).float()
        else:
            cn_sample = torch.tensor([cnp.sample() for cnp in self.gen]).float()

        mean = self.distribution.mean
        cn_sample = cn_sample.to(self.log_std.device)

        sample = mean + torch.exp(self.log_std) * cn_sample
        if self.debug:
            path = f"{LEGGED_GYM_ROOT_DIR}/plots/distribution_pink.csv"
            self.log_actions(mean[0][0], sample[0][0], path)
        return sample

    def act_inference(self, observations):
        if self._normalize_obs:
            observations = self.normalize(observations)
        mean_actions = self.NN(observations)
        return mean_actions
