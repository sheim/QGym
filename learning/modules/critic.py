import torch
import torch.nn as nn

from .utils import create_MLP
from .utils import RunningMeanStd
from .lqrc import CholeskyPlusConst


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

        self.NN = create_MLP(num_obs, 1, hidden_dims, activation)
        self._normalize_obs = normalize_obs
        if self._normalize_obs:
            self.obs_rms = RunningMeanStd(num_obs)

    def evaluate(self, critic_observations):
        if self._normalize_obs:
            with torch.no_grad():
                critic_observations = self.obs_rms(critic_observations)
        return self.NN(critic_observations).squeeze()

    def loss_fn(self, input, target):
        return nn.functional.mse_loss(input, target, reduction="mean")
