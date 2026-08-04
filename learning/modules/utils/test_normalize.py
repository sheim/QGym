import torch

from learning.modules.utils.normalize import RunningMeanStd


def test_update_moments():
    rms = RunningMeanStd(num_items=3)
    running_mean = 0.95 * torch.ones(3)
    running_var = 0.2 * torch.ones(3)
    batch_mean = 0.35 * torch.ones(3)
    batch_var = 0.1 * torch.ones(3)

    new_mean, new_var, total_count = rms._update_mean_var_from_moments(
        running_mean,
        running_var,
        torch.tensor(10.0),
        batch_mean,
        batch_var,
        batch_count=5,
    )

    torch.testing.assert_close(new_mean, 0.75 * torch.ones(3))
    torch.testing.assert_close(new_var, (3.7 / 14.0) * torch.ones(3))
    assert total_count == 15


def test_forward_updates_statistics_and_normalizes_input():
    rms = RunningMeanStd(num_items=2)
    normalized = rms(torch.ones(2, 2))

    expected = (
        (1.0 / 3.0) / torch.sqrt(torch.tensor(5.0 / 6.0 + rms.epsilon))
    ) * torch.ones_like(normalized)
    torch.testing.assert_close(normalized, expected)
