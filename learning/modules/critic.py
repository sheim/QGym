import torch
import torch.nn as nn
from .utils import create_MLP
from .utils import RunningMeanStd


class Critic(nn.Module):
    def __init__(
        self,
        num_obs,
        hidden_dims=None,
        activation="elu",
        normalize_obs=True,
        **kwargs,
    ):
        super().__init__()

        self.NN = create_MLP(num_obs, 1, hidden_dims, activation, **kwargs)
        self._normalize_obs = normalize_obs
        if self._normalize_obs:
            self.obs_rms = RunningMeanStd(num_obs)

    def forward(self, x):
        if self._normalize_obs:
            with torch.no_grad():
                x = self.obs_rms(x)
        return self.NN(x).squeeze()

    def evaluate(self, critic_observations):
        return self.forward(critic_observations)

    def loss_fn(self, obs, target):
        return nn.functional.mse_loss(self.forward(obs), target, reduction="mean")
