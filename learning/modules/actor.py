import torch
import torch.nn as nn
from torch.distributions import Normal
from .utils import StaticNN, create_MLP, export_network
from .utils import RunningMeanStd


class Actor(nn.Module):
    def __init__(
        self,
        num_obs,
        num_actions,
        hidden_dims,
        activation="elu",
        init_noise_std=1.0,
        normalize_obs=True,
        store_pik=False,
        **kwargs,
    ):
        super().__init__()

        self._normalize_obs = normalize_obs
        if self._normalize_obs:
            self.obs_rms = RunningMeanStd(num_obs)

        self.num_obs = num_obs
        self.num_actions = num_actions
        self.NN = create_MLP(num_obs, num_actions, hidden_dims, activation)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        self.store_pik = store_pik
        if self.store_pik:
            self._NN_pik = StaticNN(
                create_MLP(num_obs, num_actions, hidden_dims, activation)
            )
            self._std_pik = self.std.detach().clone()
            self.update_pik_weights()

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    @property
    def obs_running_mean(self):
        return self.obs_rms.running_mean

    @property
    def obs_running_std(self):
        return self.obs_rms.running_var.sqrt()

    def update_distribution(self, observations):
        mean = self.act_inference(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        if self._normalize_obs:
            with torch.no_grad():
                observations = self.obs_rms(observations)
        return self.NN(observations)

    def forward(self, observations):
        return self.act_inference(observations)

    def export(self, path):
        export_network(self, "policy", path, self.num_obs)

    def update_pik_weights(self):
        with torch.no_grad():
            nn_state_dict = self.NN.state_dict()
            self._NN_pik.model.load_state_dict(nn_state_dict)
            self._std_pik = self.std.detach().clone()

    def get_pik_log_prob(self, observations, actions):
        if self._normalize_obs:
            # TODO: Check if this updates the normalization mean/std
            with torch.no_grad():
                observations = self.obs_rms(observations)
        mean_pik = self._NN_pik(observations)
        std_pik = self._std_pik.to(mean_pik.device)
        distribution = Normal(mean_pik, mean_pik * 0.0 + std_pik)
        return distribution.log_prob(actions).sum(dim=-1)
