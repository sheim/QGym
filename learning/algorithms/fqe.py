"""Fixed-policy fitted Q evaluation for recorded transitions."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F

from learning.modules.utils.neural_net import create_MLP
from learning.offline.nominal_policy import FrozenTrotPolicy


class QNetwork(torch.nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...],
        activation: str,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dims = tuple(hidden_dims)
        self.activation = str(activation)
        self.net = create_MLP(
            obs_dim + action_dim,
            1,
            list(hidden_dims),
            activation,
        )

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        inputs = torch.cat((observations, actions), dim=-1)
        return self.net(inputs).squeeze(-1)


class FQE:
    """Fit Q for a fixed stochastic policy and a fixed reward target."""

    def __init__(
        self,
        q_network: QNetwork,
        policy: FrozenTrotPolicy,
        *,
        gamma: float,
        failure_value: float,
        target_tau: float,
        learning_rate: float,
        device: str | torch.device,
        target_action_samples: int = 1,
    ) -> None:
        if not 0.0 <= gamma < 1.0:
            raise ValueError("gamma must satisfy 0 <= gamma < 1")
        if not math.isfinite(failure_value):
            raise ValueError("failure_value must be finite")
        if not 0.0 <= target_tau <= 1.0:
            raise ValueError("target_tau must be in [0, 1]")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if target_action_samples <= 0:
            raise ValueError("target_action_samples must be positive")
        self.device = torch.device(device)
        self.q = q_network.to(self.device)
        self.target_q = copy.deepcopy(self.q).to(self.device)
        for parameter in self.target_q.parameters():
            parameter.requires_grad_(False)
        self.policy = policy
        self.gamma = gamma
        self.failure_value = failure_value
        self.target_tau = target_tau
        self.target_action_samples = target_action_samples
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=learning_rate)

    def _target_next_q(
        self,
        policy_next_obs: torch.Tensor,
        q_next_obs: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        values = []
        for _ in range(self.target_action_samples):
            actions = self.policy.sample_actions(policy_next_obs, generator)
            values.append(self.target_q(q_next_obs, actions))
        return torch.stack(values, dim=0).mean(dim=0)

    def compute_target(
        self, batch: dict[str, torch.Tensor], generator: torch.Generator
    ) -> torch.Tensor:
        reward = batch["reward"]
        terminated = batch["terminated"].bool()
        next_q = self._target_next_q(
            batch["policy_next_obs"],
            batch["q_next_obs"],
            generator,
        )
        continuing_target = reward + self.gamma * next_q
        terminal_target = reward + self.gamma * self.failure_value
        return torch.where(terminated, terminal_target, continuing_target)

    def update(
        self, batch: dict[str, torch.Tensor], generator: torch.Generator
    ) -> float:
        target = self.compute_target(batch, generator).detach()
        prediction = self.q(batch["q_obs"], batch["action"])
        loss = F.mse_loss(prediction, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._soft_update()
        return float(loss.detach())

    @torch.no_grad()
    def _soft_update(self) -> None:
        for target, source in zip(
            self.target_q.parameters(),
            self.q.parameters(),
            strict=True,
        ):
            target.mul_(1.0 - self.target_tau).add_(source, alpha=self.target_tau)
