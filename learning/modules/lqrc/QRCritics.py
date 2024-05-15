import torch
import torch.nn as nn
import torch.nn.functional as F

from learning.modules.utils import RunningMeanStd, create_MLP


def create_lower_diagonal(x, n, device):
    L = torch.zeros((*x.shape[:-1], n, n), device=device, requires_grad=False)
    tril_indices = torch.tril_indices(row=n, col=n, offset=0)
    rows, cols = tril_indices
    L[..., rows, cols] = x
    return L


def compose_cholesky(L):
    return torch.einsum("...ij, ...jk -> ...ik", L, L.transpose(-2, -1))


def quadratify_xAx(x, A):
    return torch.einsum(
        "...ij, ...jk -> ...ik",
        torch.einsum("...ij, ...jk -> ...ik", x.unsqueeze(-1).transpose(-2, -1), A),
        x.unsqueeze(-1),
    ).squeeze()


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class CholeskyInput(nn.Module):
    def __init__(
        self,
        num_obs,
        hidden_dims,
        latent_dim=None,
        activation="elu",
        dropouts=None,
        normalize_obs=False,
        minimize=False,
        device="cuda",
        **kwargs,
    ):
        super().__init__()
        self.device = device
        if latent_dim is None:
            latent_dim = num_obs
        self.latent_dim = latent_dim
        self.minimize = minimize
        if minimize:
            self.sign = torch.ones(1, device=device)
        else:
            self.sign = -torch.ones(1, device=device)
        self.value_offset = nn.Parameter(torch.zeros(1, device=device))

        self._normalize_obs = normalize_obs
        if self._normalize_obs:
            self.obs_rms = RunningMeanStd(num_obs)

        num_lower_diag_elements = sum(range(latent_dim + 1))
        self.lower_diag_NN = create_MLP(
            num_obs, num_lower_diag_elements, hidden_dims, activation, dropouts
        )
        # self.lower_diag_NN.apply(init_weights)

    def forward(self, x):
        output = self.lower_diag_NN(x)
        L = create_lower_diagonal(output, self.latent_dim, self.device)
        A = compose_cholesky(L)

        value = self.sign * quadratify_xAx(x, A)
        with torch.no_grad():
            # do not affect value offset in this part of the loss
            value += self.value_offset
        return value
        # return self.sign * quadratify_xAx(x, A) + self.value_offset

    def evaluate(self, obs):
        return self.forward(obs)

    def loss_fn(self, obs, target):
        loss_NN = F.mse_loss(self.forward(obs), target, reduction="mean")
        if self.minimize:
            loss_offset = (self.value_offset / target.min() - 1.0).pow(2)
        else:
            loss_offset = (self.value_offset / target.max() - 1.0).pow(2)
        return loss_NN + loss_offset


class CholeskyLatent(CholeskyInput):
    def __init__(
        self,
        num_obs,
        hidden_dims,
        latent_dim,
        latent_hidden_dims=[64],
        latent_activation="tanh",
        latent_dropouts=None,
        activation="elu",
        dropouts=None,
        normalize_obs=False,
        minimize=False,
        device="cuda",
        **kwargs,
    ):
        super().__init__(
            num_obs,
            hidden_dims,
            latent_dim=latent_dim,
            activation=activation,
            dropouts=dropouts,
            normalize_obs=normalize_obs,
            minimize=minimize,
            device="cuda",
            **kwargs,
        )

        self.latent_NN = create_MLP(
            num_obs, latent_dim, latent_hidden_dims, latent_activation
        )

    def forward(self, x):
        z = self.latent_NN(x)
        output = self.lower_diag_NN(x)
        L = create_lower_diagonal(output, self.latent_dim, self.device)
        A = compose_cholesky(L)

        value = self.sign * quadratify_xAx(z, A)
        with torch.no_grad():
            # do not affect value offset in this part of the loss
            value += self.value_offset * 100.0
        return value
        # return self.sign * quadratify_xAx(x, A) + self.value_offset

    def evaluate(self, obs):
        return self.forward(obs)
