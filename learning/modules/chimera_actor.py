import torch
import torch.nn as nn
from torch.distributions import Normal
from .utils import create_MLP
from .utils import export_network
from .utils import RunningMeanStd


class ChimeraActor(nn.Module):
    def __init__(
        self,
        num_obs,
        num_actions,
        hidden_dims,
        split_idx=1,
        activation="tanh",
        log_std_max: float = 4.0,
        log_std_min: float = -20.0,
        std_init: float = 1.0,
        normalize_obs=True,
        **kwargs,
    ):
        super().__init__()

        self._normalize_obs = normalize_obs
        if self._normalize_obs:
            self.obs_rms = RunningMeanStd(num_obs)

        self.log_std_max = log_std_max
        self.log_std_min = log_std_min
        self.log_std_init = torch.tensor([std_init]).log()  # refactor

        self.num_obs = num_obs
        self.num_actions = num_actions

        assert split_idx < len(hidden_dims)
        self.latent_NN = create_MLP(
            num_obs, hidden_dims[-split_idx], hidden_dims[:-split_idx], activation
        )
        self.mean_NN = create_MLP(
            hidden_dims[-split_idx], num_actions, hidden_dims[split_idx:], activation
        )
        self.std_NN = create_MLP(
            hidden_dims[-split_idx], num_actions, hidden_dims[split_idx:], activation
        )

        # nn.init.orthogonal_()

        # maybe zap
        self.distribution = Normal(torch.zeros(num_actions), torch.ones(num_actions))
        Normal.set_default_validate_args = False

    def forward(self, x, deterministic=True):
        if self._normalize_obs:
            with torch.no_grad():
                x = self.obs_rms(x)
        latent = self.latent_NN(x)
        mean = self.mean_NN(latent)
        if deterministic:
            return mean
        log_std = self.log_std_init + self.std_NN(latent)
        return mean, log_std.clamp(self.log_std_min, self.log_std_max).exp()

    def act(self, x):
        mean, std = self.forward(x, deterministic=False)
        self.distribution = Normal(mean, std)
        return self.distribution.sample()

    def inference_policy(self, x):
        return self.forward(x, deterministic=True)

    def export(self, path):
        export_network(self.inference_policy, "policy", path, self.num_obs)

    def to(self, device):
        super().to(device)
        self.log_std_init = self.log_std_init.to(device)
        return self
