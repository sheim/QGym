from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.distributions import Normal


# The following implementation was used in the gSDE paper on smooth exploration.
# See code:
# https://github.com/DLR-RM/stable-baselines3/blob/56f20e40a2206bbb16501a0f600e29ce1b112ef1/stable_baselines3/common/distributions.py#L421C7-L421C38
# TODO[lm]: Need to update some of the naming conventions to fit our codebase.
class StateDependentNoiseDistribution:
    """
    Distribution class for using generalized State Dependent Exploration (gSDE).
    It is used to create the noise exploration matrix and compute the log
    probability of an action with that noise.
    """

    latent_sde_dim: Optional[int]
    weights_dist: Normal
    _latent_sde: torch.Tensor
    exploration_mat: torch.Tensor
    exploration_matrices: torch.Tensor

    def __init__(
        self,
        num_actions: int,
        num_actor_obs: int,
        num_critic_obs: int,
        full_std: bool = True,
        use_expln: bool = False,
        learn_features: bool = False,
        epsilon: float = 1e-6,
    ):
        self.num_actions = num_actions
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.full_std = full_std
        self.use_expln = use_expln
        self.learn_features = learn_features
        self.epsilon = epsilon
        self.mean_actions = None
        self.log_std = None

    def get_std(self, log_std: torch.Tensor) -> torch.Tensor:
        """
        Get the standard deviation from the learned parameter
        (log of it by default). This ensures that the std is positive.
        """
        if self.use_expln:
            # From gSDE paper, it allows to keep variance
            # above zero and prevent it from growing too fast
            below_threshold = torch.exp(log_std) * (log_std <= 0)
            # Avoid NaN: zeros values that are below zero
            safe_log_std = log_std * (log_std > 0) + self.epsilon
            above_threshold = (torch.log1p(safe_log_std) + 1.0) * (log_std > 0)
            std = below_threshold + above_threshold
        else:
            # Use normal exponential
            std = torch.exp(log_std)

        if self.full_std:
            return std
        assert self.latent_sde_dim is not None
        # Reduce the number of parameters:
        return torch.ones(self.latent_sde_dim, self.action_dim).to(log_std.device) * std

    def sample_weights(self, log_std: torch.Tensor, batch_size: int = 1) -> None:
        """
        Sample weights for the noise exploration matrix,
        using a centered Gaussian distribution.
        """
        std = self.get_std(log_std)
        self.weights_dist = Normal(torch.zeros_like(std), std)
        # Reparametrization trick to pass gradients
        self.exploration_mat = self.weights_dist.rsample()
        # Pre-compute matrices in case of parallel exploration
        self.exploration_matrices = self.weights_dist.rsample((batch_size,))

    def get_distribution_net(
        self,
        latent_dim: int,
        log_std_init: float = -2.0,
        latent_sde_dim: Optional[int] = None,
    ) -> Tuple[nn.Module, nn.Parameter]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the deterministic action, the other parameter will be the
        standard deviation of the distribution that control the weights of the noise
        matrix.
        """
        # Network for the deterministic action, it represents the mean of the
        # distribution
        mean_actions_net = nn.Linear(latent_dim, self.action_dim)
        # When we learn features for the noise, the feature dimension
        # can be different between the policy and the noise network
        self.latent_sde_dim = latent_dim if latent_sde_dim is None else latent_sde_dim
        # Reduce the number of parameters if needed
        log_std = (
            torch.ones(self.latent_sde_dim, self.action_dim)
            if self.full_std
            else torch.ones(self.latent_sde_dim, 1)
        )
        # Transform it to a parameter so it can be optimized
        log_std = nn.Parameter(log_std * log_std_init, requires_grad=True)
        # Sample an exploration matrix
        self.sample_weights(log_std)
        return mean_actions_net, log_std

    def get_distribution(
        self,
        mean_actions: torch.Tensor,
        log_std: torch.Tensor,
        latent_sde: torch.Tensor,
    ) -> "StateDependentNoiseDistribution":
        """
        Create the distribution given its parameters (mean, std)
        """
        # Stop gradient if we don't want to influence the features
        self._latent_sde = latent_sde if self.learn_features else latent_sde.detach()
        variance = torch.mm(self._latent_sde**2, self.get_std(log_std) ** 2)
        self.distribution = Normal(mean_actions, torch.sqrt(variance + self.epsilon))
        return self

    def log_prob(self, actions: torch.Tensor):
        # TODO[lm]: verify that summed correctly (stable-baselines3 is different)
        return self.distribution.log_prob(actions).sum(-1)

    def entropy(self):
        # TODO[lm]: verify that summed correctly (stable-baselines3 is different)
        return self.distribution.entropy().sum(-1)

    def sample(self) -> torch.Tensor:
        noise = self.get_noise(self._latent_sde)
        actions = self.distribution.mean + noise
        return actions

    def mode(self) -> torch.Tensor:
        return self.distribution.mean

    def get_noise(self, latent_sde: torch.Tensor) -> torch.Tensor:
        latent_sde = latent_sde if self.learn_features else latent_sde.detach()
        # Default case: only one exploration matrix
        if len(latent_sde) == 1 or len(latent_sde) != len(self.exploration_matrices):
            return torch.mm(latent_sde, self.exploration_mat)
        # Use batch matrix multiplication for efficient computation
        # (batch_size, n_features) -> (batch_size, 1, n_features)
        latent_sde = latent_sde.unsqueeze(dim=1)
        # (batch_size, 1, n_actions)
        noise = torch.bmm(latent_sde, self.exploration_matrices)
        return noise.squeeze(dim=1)

    def actions_from_params(
        self,
        mean_actions: torch.Tensor,
        log_std: torch.Tensor,
        latent_sde: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        # Update the distribution
        self.get_distribution(mean_actions, log_std, latent_sde)
        if deterministic:
            return self.mode()
        return self.sample()

    def log_prob_from_params(
        self,
        mean_actions: torch.Tensor,
        log_std: torch.Tensor,
        latent_sde: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions_from_params(mean_actions, log_std, latent_sde)
        log_prob = self.log_prob(actions)
        return actions, log_prob
