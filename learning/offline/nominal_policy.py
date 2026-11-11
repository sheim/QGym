"""Frozen stochastic nominal policy used as the FQE continuation policy."""

from __future__ import annotations

from pathlib import Path

import torch

from learning.modules.actor import Actor


class FrozenTrotPolicy:
    """Load one PPO actor and sample its training-time action distribution."""

    def __init__(
        self,
        actor: Actor,
        *,
        obs_names: tuple[str, ...],
        obs_dims: tuple[int, ...],
        action_names: tuple[str, ...],
        action_dim: int,
        device: str | torch.device,
    ) -> None:
        if len(obs_names) != len(obs_dims):
            raise ValueError("obs_names and obs_dims must have equal length")
        if sum(obs_dims) != actor.num_obs:
            raise ValueError("actor observation width does not match obs_dims")
        if action_dim != actor.num_actions:
            raise ValueError("actor action width does not match action_dim")
        self.actor = actor
        self.obs_names = tuple(obs_names)
        self.obs_dims = tuple(obs_dims)
        self.action_names = tuple(action_names)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)
        self.actor.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        actor_cfg: dict,
        obs_names: tuple[str, ...],
        obs_dims: tuple[int, ...],
        action_names: tuple[str, ...],
        action_dim: int,
        device: str | torch.device,
    ) -> "FrozenTrotPolicy":
        if bool(actor_cfg["smooth_exploration"]):
            raise ValueError("this pipeline supports the standard Actor only")
        num_obs = sum(obs_dims)
        actor = Actor(num_obs, action_dim, **actor_cfg)
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if "actor_state_dict" not in loaded:
            raise ValueError("checkpoint does not contain actor_state_dict")
        actor.load_state_dict(loaded["actor_state_dict"])
        actor.to(device)
        return cls(
            actor,
            obs_names=obs_names,
            obs_dims=obs_dims,
            action_names=action_names,
            action_dim=action_dim,
            device=device,
        )

    def sample_actions(
        self, observations: torch.Tensor, generator: torch.Generator
    ) -> torch.Tensor:
        if observations.device != self.device:
            raise ValueError("observations must be on the policy device")
        if generator.device != self.device:
            raise ValueError("generator must be on the policy device")
        with torch.no_grad():
            mean = self.actor.act_inference(observations)
            std = self.actor.std.detach().to(mean.device).expand_as(mean)
            noise = torch.randn(
                mean.shape,
                generator=generator,
                device=mean.device,
                dtype=mean.dtype,
            )
            return mean + std * noise

    def mean_actions(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.device != self.device:
            raise ValueError("observations must be on the policy device")
        with torch.no_grad():
            return self.actor.act_inference(observations)
