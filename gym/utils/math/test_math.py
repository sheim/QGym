import numpy as np
import torch

from gym.utils.math.simple_math import (
    exp_avg_filter,
    torch_rand_sqrt_float,
    wrap_to_pi,
)


def test_wrap_to_pi():
    angles = np.array([0, np.pi, 2 * np.pi, 3 * np.pi])
    wrapped_angles = wrap_to_pi(angles)
    assert np.allclose(wrapped_angles, np.array([0, np.pi, 0, np.pi]))


def test_torch_rand_sqrt_float_stays_inside_requested_range():
    torch.manual_seed(1)
    samples = torch_rand_sqrt_float(
        lower=-1.0,
        upper=1.0,
        shape=(10,),
        device=torch.device("cpu"),
    )

    assert samples.shape == (10,)
    assert torch.all(samples >= -1.0)
    assert torch.all(samples <= 1.0)


def test_exp_avg_filter():
    value = torch.tensor(5.0)
    average = torch.tensor(1.0)

    for alpha in (0.0, 0.5, 1.0):
        filtered = exp_avg_filter(value, average, alpha)
        assert filtered == alpha * value + (1 - alpha) * average
