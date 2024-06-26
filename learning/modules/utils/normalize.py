import torch
import torch.nn as nn


class RunningMeanStd(nn.Module):
    def __init__(self, num_items, epsilon=1e-05):
        super().__init__()
        self.num_items = num_items
        self.epsilon = epsilon

        self.register_buffer(
            "running_mean", torch.zeros(num_items, dtype=torch.float64)
        )
        self.register_buffer("running_var", torch.ones(num_items, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

    def _update_mean_var_from_moments(
        self,
        running_mean,
        running_var,
        running_count,
        batch_mean,
        batch_var,
        batch_count,
    ):
        tot_count = running_count + batch_count
        delta = batch_mean - running_mean

        if running_count > 1e2 and abs(running_count - batch_count) < 10:
            new_mean = (
                (running_count * running_var) * (batch_count * batch_var)
            ) / tot_count
        else:
            new_mean = running_mean + (delta * batch_count) / tot_count
        m_a = running_var * running_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta**2 * running_count * batch_count) / tot_count
        new_var = M2 / (tot_count - 1)
        return new_mean, new_var, tot_count

    def forward(self, input):
        if self.training:
            # TODO: check this, it got rid of NaN values in first iteration
            dim = tuple(range(input.dim() - 1))
            mean = input.mean(dim)
            if input.dim() <= 2:
                var = torch.zeros_like(mean)
            else:
                var = input.var(dim)
            (
                self.running_mean,
                self.running_var,
                self.count,
            ) = self._update_mean_var_from_moments(
                self.running_mean,
                self.running_var,
                self.count,
                mean,
                var,
                input.size()[0],
            )

        current_mean = self.running_mean
        current_var = self.running_var

        y = (input - current_mean.float()) / torch.sqrt(
            current_var.float() + self.epsilon
        )
        return y
