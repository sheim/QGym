import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from pink import ColoredNoiseProcess

from .actor import Actor
from .utils import create_MLP

from gym import LEGGED_GYM_ROOT_DIR


# The following implementation is based on the pinkNoise paper. See code:
# https://github.com/martius-lab/pink-noise-rl/blob/main/pink/sb3.py
class PinkActor(Actor):
    _latent_sde: torch.Tensor

    def __init__(
        self,
        *args,
        full_std: bool = True,
        use_exp_ln: bool = False,
        learn_features: bool = True,
        epsilon: float = 1e-6,
        log_std_init: float = -0.5,
        beta=0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.full_std = full_std
        self.use_exp_ln = use_exp_ln
        self.learn_features = learn_features
        self.epsilon = epsilon
        self.log_std_init = log_std_init
        self.beta = beta

        # TODO: is 500 correct?
        self.gen = ColoredNoiseProcess(beta=self.beta, size=(self.num_actions, 1000))

        # Create latent NN and last layer
        self.latent_net = create_MLP(
            self.num_obs,
            self.num_actions,
            self.hidden_dims,
            self.activation,
            latent=True,
        )
        self.latent_dim = self.hidden_dims[-1]

        self.mean_actions_net = nn.Linear(self.latent_dim, self.num_actions)

        self.log_std = nn.Parameter(
            torch.ones(self.num_actions) * log_std_init, requires_grad=True
        )

        self.distribution = None

        # Debug mode for plotting
        self.debug = True

    def update_distribution(self, observations):
        if self._normalize_obs:
            observations = self.normalize(observations)
        # Get latent features and compute distribution
        self._latent_sde = self.latent_net(observations)
        if not self.learn_features:
            self._latent_sde = self._latent_sde.detach()
        mean_actions = self.mean_actions_net(self._latent_sde)
        action_std = torch.ones_like(mean_actions) * torch.exp(self.log_std)
        self.distribution = Normal(mean_actions, action_std)

    # TODO[ni]: Sample actions that do not fit into the distribution at all sometimes
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
            self.log_actions(mean[0][2], sample[0][2], path)
        return torch.tanh(sample)

    def act_inference(self, observations):
        if self._normalize_obs:
            observations = self.normalize(observations)
        latent_sde = self.latent_net(observations)
        mean_actions = self.mean_actions_net(latent_sde)
        return mean_actions

    def get_actions_log_prob(self, actions):
        eps = torch.finfo(actions.dtype).eps
        gaussian_actions = actions.clamp(min=-1.0 + eps, max=1.0 - eps)
        gaussian_actions = 0.5 * (
            gaussian_actions.log1p() - (-gaussian_actions).log1p()
        )
        log_prob = super().get_actions_log_prob(gaussian_actions)
        log_prob -= torch.sum(torch.log(1 - actions**2 + self.epsilon), dim=1)
        return log_prob
