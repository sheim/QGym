import json

import torch

from learning.corl_artifact import CORLPolicy


def test_artifact_applies_normalization_action_scale_and_batching(tmp_path):
    """Analytic one-layer policy catches missing input/output transforms."""
    metadata = {
        "format": "corl-deterministic-mlp-v1",
        "dims": [2, 1],
        "observation_mean": [1.0, -2.0],
        "observation_std": [2.0, 4.0],
        "max_action": 0.7,
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    torch.save(
        {"0.weight": torch.tensor([[2.0, -1.0]]), "0.bias": torch.tensor([0.25])},
        tmp_path / "weights.pt",
    )
    policy = CORLPolicy(tmp_path)
    inputs = torch.tensor([[3.0, 2.0], [1.0, -2.0]])
    expected = 0.7 * torch.tanh(torch.tensor([[1.25], [0.25]]))
    torch.testing.assert_close(policy(inputs), expected)
    torch.testing.assert_close(policy(inputs)[0], policy(inputs[:1])[0])
    assert not policy.training
