import torch

from learning.utils import PotentialBasedRewardShaping


class ExampleTask:
    """Minimal task showing the interface expected by PBRS."""

    def __init__(self):
        self.potentials = {"progress": 1.0, "stability": 2.0}

    def compute_reward(self, reward_weights):
        reward = sum(
            weight * self.potentials[name] for name, weight in reward_weights.items()
        )
        return torch.tensor([reward])


def test_potential_based_reward_shaping_usecase():
    task = ExampleTask()
    shaping = PotentialBasedRewardShaping(
        {"progress": 1.0, "stability": 2.0},
        device="cpu",
    )
    shaping.set_discount(horizon=1.0, dt=0.1)

    assert shaping.get_reward_keys() == ["PBRS_progress", "PBRS_stability"]

    shaping.pre_step(task)
    task.potentials.update(progress=2.0, stability=3.0)
    shaped_rewards = shaping.post_step(task, mask=torch.ones(1))

    torch.testing.assert_close(shaped_rewards["PBRS_progress"], torch.tensor([0.8]))
    torch.testing.assert_close(shaped_rewards["PBRS_stability"], torch.tensor([1.4]))
    assert shaping.prestep_counter == shaping.poststep_counter == 1
