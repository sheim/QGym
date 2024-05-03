import pytest
import torch
from tensordict import TensorDict
from learning.utils import compute_MC_returns


class CriticConstant(torch.nn.Module):
    def __init__(self, output):
        super().__init__()
        self.output = output

    def evaluate(self, obs):
        return self.output * torch.ones(obs.shape[0])


@pytest.fixture
def setup_data():
    n_timesteps = 4
    n_envs = 5
    rewards = torch.ones((n_timesteps, n_envs))
    terminated = torch.zeros((n_timesteps, n_envs), dtype=torch.bool)
    timed_out = torch.zeros_like(terminated)
    critic_obs = torch.rand((n_timesteps, n_envs))

    # Terminating conditions setup
    timed_out[-1, [0, 2]] = True
    terminated[-1, [1, 2]] = True
    timed_out[1, 3] = True
    terminated[1, 4] = True

    dones = timed_out | terminated
    data = TensorDict(
        {
            "rewards": rewards,
            "timed_out": timed_out,
            "terminated": terminated,
            "dones": dones,
            "critic_obs": critic_obs,
        }
    )

    return data


def test_critic_always_zero_gamma_zero(setup_data):
    data = setup_data
    critic = CriticConstant(0)
    returns = compute_MC_returns(data, gamma=0.0, critic=critic)
    expected_returns = data["rewards"]
    torch.testing.assert_close(returns, expected_returns)


def test_critic_always_four_gamma_zero(setup_data):
    data = setup_data
    critic = CriticConstant(4)
    returns = compute_MC_returns(data, gamma=0.0, critic=critic)
    expected_returns = data["rewards"]
    torch.testing.assert_close(returns, expected_returns)


def test_critic_always_zero_gamma_one(setup_data):
    data = setup_data
    critic = CriticConstant(0)
    returns = compute_MC_returns(data, gamma=1.0, critic=critic)
    expected_returns = torch.tensor(
        [
            [4.0, 3.0, 2.0, 1.0],
            [4.0, 3.0, 2.0, 1.0],
            [4.0, 3.0, 2.0, 1.0],
            [2.0, 1.0, 2.0, 1.0],
            [2.0, 1.0, 2.0, 1.0],
        ]
    ).T
    torch.testing.assert_close(returns, expected_returns)


def test_critic_always_four_gamma_one(setup_data):
    data = setup_data
    critic = CriticConstant(4)
    returns = compute_MC_returns(data, gamma=1.0, critic=critic)
    expected_returns = torch.tensor(
        [
            [8.0, 7.0, 6.0, 5.0],
            [4.0, 3.0, 2.0, 1.0],
            [4.0, 3.0, 2.0, 1.0],
            [6.0, 5.0, 6.0, 5.0],
            [2.0, 1.0, 6.0, 5.0],
        ]
    ).T
    torch.testing.assert_close(returns, expected_returns)


def test_critic_always_zero_gamma_half(setup_data):
    data = setup_data
    critic = CriticConstant(0)
    returns = compute_MC_returns(data, gamma=0.5, critic=critic)
    expected_returns = torch.tensor(
        [
            [1.875, 1.75, 1.5, 1.0],
            [1.875, 1.75, 1.5, 1.0],
            [1.875, 1.75, 1.5, 1.0],
            [1.5, 1.0, 1.5, 1.0],
            [1.5, 1.0, 1.5, 1.0],
        ]
    ).T
    torch.testing.assert_close(returns, expected_returns)


def test_critic_always_four_gamma_half(setup_data):
    data = setup_data
    critic = CriticConstant(6.0)
    returns = compute_MC_returns(data, gamma=0.5, critic=critic)
    expected_returns = torch.tensor(
        [
            [2.25, 2.5, 3.0, 4.0],
            [1.875, 1.75, 1.5, 1.0],
            [1.875, 1.75, 1.5, 1.0],
            [3.0, 4.0, 3.0, 4.0],
            [1.5, 1.0, 3.0, 4.0],
        ]
    ).T
    torch.testing.assert_close(returns, expected_returns)
