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


def create_PD_lower_diagonal(x, n, device):
    tril_indices = torch.tril_indices(row=n, col=n, offset=0)
    diag_indices = (tril_indices[0] == tril_indices[1]).nonzero(as_tuple=True)[0]
    x[..., diag_indices] = F.softplus(x[..., diag_indices])
    L = torch.zeros((*x.shape[:-1], n, n), device=device, requires_grad=False)
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
        # V = x'Ax + b = a* x'Ax + b, and add a regularization for det(A) ~= 1
        self.scaling_quadratic = nn.Parameter(torch.ones(1, device=device))

        self._normalize_obs = normalize_obs
        if self._normalize_obs:
            self.obs_rms = RunningMeanStd(num_obs)

        num_lower_diag_elements = sum(range(latent_dim + 1))
        self.lower_diag_NN = create_MLP(
            num_obs, num_lower_diag_elements, hidden_dims, activation, dropouts
        )
        self.lower_diag_NN.apply(init_weights)

    def forward(self, x, return_all=False):
        output = self.lower_diag_NN(x)
        L = create_lower_diagonal(output, self.latent_dim, self.device)
        A = compose_cholesky(L)
        value = self.sign * quadratify_xAx(x, A)
        with torch.no_grad():
            value += self.value_offset
        value *= self.scaling_quadratic

        if return_all:
            return value, A, L
        else:
            return value
        # return self.sign * quadratify_xAx(x, A) + self.value_offset

    def evaluate(self, obs):
        return self.forward(obs)

    def loss_fn(self, obs, target, **kwargs):
        loss_NN = F.mse_loss(self.forward(obs), target, reduction="mean")

        if self.minimize:
            loss_offset = (self.value_offset / target.min() - 1.0).pow(2)
        else:
            loss_offset = (self.value_offset / target.max() - 1.0).pow(2)

        loss_scaling = (self.scaling_quadratic - target.mean()).pow(2)

        return loss_NN + loss_offset + loss_scaling


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

        self.latent_NN.apply(init_weights)

    def forward(self, x, return_all=False):
        z = self.latent_NN(x)
        output = self.lower_diag_NN(x)
        L = create_lower_diagonal(output, self.latent_dim, self.device)
        A = compose_cholesky(L)

        value = self.sign * quadratify_xAx(z, A)
        with torch.no_grad():
            # do not affect value offset in this part of the loss
            value += self.value_offset
        value *= self.scaling_quadratic

        if return_all:
            return value, A, L, z
        else:
            return value

    def evaluate(self, obs):
        return self.forward(obs)


class PDCholeskyInput(CholeskyInput):
    def forward(self, x, return_all=False):
        output = self.lower_diag_NN(x)
        L = create_PD_lower_diagonal(output, self.latent_dim, self.device)
        A = compose_cholesky(L)
        # assert (torch.linalg.eigvals(A).real > 0).all()
        value = self.sign * quadratify_xAx(x, A)
        with torch.no_grad():
            value += self.value_offset
        value *= self.scaling_quadratic

        if return_all:
            return value, A, L
        else:
            return value
        # return self.sign * quadratify_xAx(x, A) + self.value_offset


class PDCholeskyLatent(CholeskyLatent):
    def forward(self, x, return_all=False):
        z = self.latent_NN(x)
        output = self.lower_diag_NN(x)
        L = create_PD_lower_diagonal(output, self.latent_dim, self.device)
        A = compose_cholesky(L)
        # assert (torch.linalg.eigvals(A).real > 0).all()
        value = self.sign * quadratify_xAx(z, A)
        with torch.no_grad():
            value += self.value_offset
        value *= self.scaling_quadratic

        if return_all:
            return value, A, L
        else:
            return value


class SpectralLatent(nn.Module):
    def __init__(
        self,
        num_obs,
        hidden_dims,
        activation="elu",
        dropouts=None,
        normalize_obs=False,
        relative_dim=None,
        latent_dim=None,
        latent_hidden_dims=[64],
        latent_activation="tanh",
        latent_dropouts=None,
        minimize=False,
        device="cuda",
        **kwargs,
    ):
        super().__init__()
        self.device = device
        if relative_dim is None:
            relative_dim = num_obs
        if latent_dim is None:
            latent_dim = num_obs

        self.num_obs = num_obs
        self.relative_dim = relative_dim
        self.latent_dim = latent_dim
        self.minimize = minimize
        if minimize:
            self.sign = torch.ones(1, device=device)
        else:
            self.sign = -torch.ones(1, device=device)
        self.value_offset = nn.Parameter(torch.zeros(1, device=device))
        self.scaling_quadratic = nn.Parameter(torch.ones(1, device=device))

        self._normalize_obs = normalize_obs
        if self._normalize_obs:
            self.obs_rms = RunningMeanStd(num_obs)
        # TODO up to here, same as Cholesky, can be refactored
        n_outputs = (
            torch.tril_indices(self.relative_dim, self.latent_dim).shape[1]
            + relative_dim
        )
        self.spectral_NN = create_MLP(
            num_obs, n_outputs, hidden_dims, activation, dropouts
        )
        self.latent_NN = create_MLP(
            num_obs, latent_dim, latent_hidden_dims, latent_activation
        )

    def forward(self, x, return_all=False):
        z = self.latent_NN(x)
        y = self.spectral_NN(x)
        A_diag = torch.diag_embed(F.softplus(y[..., : self.relative_dim]))
        # tril_indices = torch.tril_indices(self.latent_dim, self.relative_dim)
        tril_indices = torch.tril_indices(self.relative_dim, self.latent_dim)
        L = torch.zeros(
            (*x.shape[:-1], self.relative_dim, self.latent_dim), device=self.device
        )
        L[..., tril_indices[0], tril_indices[1]] = y[..., self.relative_dim :]
        # Compute (L^T A_diag) L
        A = torch.einsum(
            "...ij,...jk->...ik",
            torch.einsum("...ik,...kj->...ij", L.transpose(-1, -2), A_diag),
            L,
        )
        # assert (torch.linalg.eigvals(A).real >= -1e-6).all()
        # ! This fails, but so far with just -e-10, so really really small...
        value = self.sign * quadratify_xAx(z, A)
        with torch.no_grad():
            value += self.value_offset

        if return_all:
            return value, z, torch.diag(A_diag), L, A
        else:
            return value

    def evaluate(self, obs):
        return self.forward(obs)

    def loss_fn(self, obs, target, **kwargs):
        loss_NN = F.mse_loss(self.forward(obs), target, reduction="mean")

        if self.minimize:
            loss_offset = (self.value_offset / target.min() - 1.0).pow(2)
        else:
            loss_offset = (self.value_offset / target.max() - 1.0).pow(2)

        loss_scaling = (self.scaling_quadratic - target.mean()).pow(2)

        return loss_NN + loss_offset + loss_scaling


class QR(nn.Module):
    def __init__(
        self,
        num_obs,
        hidden_dims,
        action_dim,
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
        self.action_dim = action_dim
        self.minimize = minimize
        if minimize:
            self.sign = torch.ones(1, device=device)
        else:
            self.sign = -torch.ones(1, device=device)

        self._normalize_obs = normalize_obs
        if self._normalize_obs:
            self.obs_rms = RunningMeanStd(num_obs)

        # Make Q estimator
        num_lower_diag_elements_Q = sum(range(self.latent_dim + 1))
        self.lower_diag_NN_Q = create_MLP(
            latent_dim, num_lower_diag_elements_Q, hidden_dims, activation, dropouts
        )
        self.lower_diag_NN_Q.apply(init_weights)

        # Make R estimator
        num_lower_diag_elements_R = sum(range(action_dim + 1))
        self.lower_diag_NN_R = create_MLP(
            latent_dim, num_lower_diag_elements_R, hidden_dims, activation, dropouts
        )
        self.lower_diag_NN_R.apply(init_weights)

    def forward(self, z, u, return_all=False):
        Q_lower_diag = self.lower_diag_NN_Q(z)
        L_Q = create_lower_diagonal(Q_lower_diag, self.latent_dim, self.device)
        Q = compose_cholesky(L_Q)

        R_lower_diag = self.lower_diag_NN_R(z)
        L_R = create_lower_diagonal(R_lower_diag, self.action_dim, self.device)
        R = compose_cholesky(L_R)

        if return_all:
            return Q, L_Q, R, L_R
        else:
            return Q, R

    def evaluate(self, obs):
        return self.forward(obs)

    def loss_fn(self, z, target, actions, **kwargs):
        Q, R = self.forward(z, target)
        Q_value = self.sign * quadratify_xAx(z, Q)
        R_value = self.sign * quadratify_xAx(actions, R)

        riccati_loss = F.mse_loss(Q_value + R_value, target, reduction="mean")

        return riccati_loss


class NN_wRiccati(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # assert (
        #     "critic_name" in kwargs.keys()
        # ), "Name of custom critic not specified in NN with Riccati init"
        # assert "action_dim" in kwargs.keys(), "Dimension of NN actions not specified."
        critic_name = kwargs["critic_name"]
        device = kwargs["device"]
        self.has_latent = False if kwargs["latent_dim"] is None else True
        self.critic = eval(f"{critic_name}(**kwargs).to(device)")
        self.QR_network = QR(**kwargs).to(device)

    def forward(self, x, return_all=False):
        return self.critic.forward(x, return_all)

    def evaluate(self, obs):
        return self.forward(obs)

    def loss_fn(self, obs, target, actions, **kwargs):
        value_loss = self.critic.loss_fn(obs, target, actions=actions)
        z = obs if self.has_latent is None else self.critic.latent_NN(obs)
        riccati_loss = self.QR_network.loss_fn(z, target, actions=actions)
        return value_loss + riccati_loss


# ! torch activaton functions don't like operating on parameters
# class QR_RiccatiNN(SpectralLatent):
#     """
#     Goal is to write this such that we can switch out the NN class it's
#     inheriting from without any further modification
#     """

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         assert (
#             "action_dim" in kwargs.keys()
#         ), "Please indicate your action dimensions as part of NN initializaion kwargs"
#         self.action_dim = kwargs["action_dim"]
#         self.Q_input = nn.Parameter(
#             torch.ones((self.latent_dim, self.latent_dim), device=self.device)
#         )
#         self.R_input = nn.Parameter(
#             torch.ones((self.action_dim, self.action_dim), device=self.device)
#         )

#     def loss_fn(self, obs, target, actions, **kwargs):
#         prediction_loss = super().loss_fn(obs, target)
#         Q = create_PD_lower_diagonal(self.Q_input, self.latent_dim, device=self.device)
#         R = create_PD_lower_diagonal(self.R_input, self.action_dim, device=self.device)
#         riccati_loss = torch.mean(
#             quadratify_xAx(self.latent_NN(obs), Q) + quadratify_xAx(actions, R)
#         )
#         return prediction_loss + riccati_loss
