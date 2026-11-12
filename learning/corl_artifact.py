"""Load CORL deterministic MLP artifacts without importing CORL or Gym."""

import json
from pathlib import Path

import torch
from torch import nn


class CORLPolicy(nn.Module):
    def __init__(self, directory):
        super().__init__()
        directory = Path(directory)
        self.metadata = json.loads((directory / "metadata.json").read_text())
        if self.metadata["format"] != "corl-deterministic-mlp-v1":
            raise ValueError("Unsupported CORL artifact format")
        dims = self.metadata["dims"]
        layers = []
        for index, (inputs, outputs) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(inputs, outputs))
            layers.append(nn.Tanh() if index == len(dims) - 2 else nn.ReLU())
        self.model = nn.Sequential(*layers)
        self.model.load_state_dict(
            torch.load(directory / "weights.pt", map_location="cpu", weights_only=True)
        )
        self.register_buffer("mean", torch.tensor(self.metadata["observation_mean"]))
        self.register_buffer("std", torch.tensor(self.metadata["observation_std"]))
        self.max_action = float(self.metadata["max_action"])
        if not torch.all(self.std > 0) or self.max_action <= 0:
            raise ValueError("Artifact requires positive scales")
        self.eval()

    @torch.no_grad()
    def forward(self, observations):
        actions = self.model((observations - self.mean) / self.std)
        return (actions * self.max_action).clamp(-self.max_action, self.max_action)
