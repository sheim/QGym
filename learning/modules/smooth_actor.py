import torch
import torch.nn as nn
from torch.distributions import Normal

from .actor import Actor
from .utils import create_MLP

from gym import LEGGED_GYM_ROOT_DIR


# The following implementation is based on the gSDE paper. See code:
# https://github.com/DLR-RM/stable-baselines3/blob/56f20e40a2206bbb16501a0f600e29ce1b112ef1/stable_baselines3/common/distributions.py#L421C7-L421C38
class SmoothActor(Actor):
    weights_dist: Normal
    latent_sde: torch.Tensor
    exploration_matrices: torch.Tensor
    exploration_scale: float

    def __init__(
        self,
        *args,
        full_std: bool = True,
        use_exp_ln: bool = True,
        learn_features: bool = True,
        epsilon: float = 1e-6,
        log_std_init: float = 0.0,
        exploration_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.full_std = full_std
        self.use_exp_ln = use_exp_ln
        self.learn_features = learn_features
        self.epsilon = epsilon
        self.log_std_init = log_std_init
        self.exploration_scale = exploration_scale  # for finetuning

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
        # Reduce the number of parameters if needed
        if self.full_std:
            log_std = torch.ones(self.latent_dim, self.num_actions)
        else:
            log_std = torch.ones(self.latent_dim, 1)
        self.log_std = nn.Parameter(log_std * self.log_std_init, requires_grad=True)
        # Sample an exploration matrix
        self.sample_weights()
        self.distribution = None

        # Debug mode for plotting
        self.debug = False

    def sample_weights(self, batch_size=1):
        # Sample weights for the noise exploration matrix
        std = self.get_std
        self.weights_dist = Normal(torch.zeros_like(std), std)
        # Pre-compute matrices in case of parallel exploration
        self.exploration_matrices = self.weights_dist.rsample((batch_size,))

    @property
    def get_std(self):
        # TODO[lm]: Check if this is ok, and can use action_std in ActorCritic normally
        if self.use_exp_ln:
            # From gSDE paper, it allows to keep variance
            # above zero and prevent it from growing too fast
            below_threshold = torch.exp(self.log_std) * (self.log_std <= 0)
            # Avoid NaN: zeros values that are below zero
            safe_log_std = self.log_std * (self.log_std > 0) + self.epsilon
            above_threshold = (torch.log1p(safe_log_std) + 1.0) * (self.log_std > 0)
            std = below_threshold + above_threshold
        else:
            # Use normal exponential
            std = torch.exp(self.log_std)

        if self.full_std:
            return std
        assert self.latent_dim is not None
        # Reduce the number of parameters:
        return torch.ones(self.latent_dim, 1).to(self.log_std.device) * std

    def update_distribution(self, observations):
        if self._normalize_obs:
            with torch.no_grad():
                observations = self.obs_rms(observations)
        # Get latent features and compute distribution
        self.latent_sde = self.latent_net(observations)
        std_scaled = self.get_std * self.exploration_scale
        if not self.learn_features:
            self.latent_sde = self.latent_sde.detach()
        if self.latent_sde.dim() == 2:
            variance = torch.mm(self.latent_sde**2, std_scaled**2)
        elif self.latent_sde.dim() == 3:
            variance = torch.einsum("abc,cd->abd", self.latent_sde**2, std_scaled**2)
        else:
            raise ValueError("Invalid latent_sde dimension")
        mean_actions = self.mean_actions_net(self.latent_sde)
        self.distribution = Normal(mean_actions, torch.sqrt(variance + self.epsilon))

    def act(self, observations):
        self.update_distribution(observations)
        mean = self.distribution.mean
        sample = mean + self.get_noise() * self.exploration_scale
        if self.debug:
            path = f"{LEGGED_GYM_ROOT_DIR}/plots/distribution_smooth.csv"
            self.log_actions(mean[0][2], sample[0][2], path)
        return sample

    def act_inference(self, observations):
        if self._normalize_obs:
            with torch.no_grad():
                observations = self.obs_rms(observations)
        latent_sde = self.latent_net(observations)
        mean_actions = self.mean_actions_net(latent_sde)
        return mean_actions

    def get_noise(self):
        latent_sde = self.latent_sde
        if not self.learn_features:
            latent_sde = latent_sde.detach()
        # Use batch matrix multiplication for efficient computation
        # (batch_size, n_features) -> (batch_size, 1, n_features)
        latent_sde = latent_sde.unsqueeze(dim=1)
        # (batch_size, 1, n_actions)
        noise = torch.bmm(latent_sde, self.exploration_matrices.to(latent_sde.device))
        return noise.squeeze(dim=1)
