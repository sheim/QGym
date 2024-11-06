import torch
import torch.nn as nn
import torch.nn.functional as F

from learning.modules.utils import RunningMeanStd, create_MLP
from learning.modules.lqrc.Losses import least_squares_fit, forward_affine


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
    res = torch.einsum(
        "...ij, ...jk -> ...ik",
        torch.einsum("...ij, ...jk -> ...ik", x.unsqueeze(-1).transpose(-2, -1), A),
        x.unsqueeze(-1),
    )
    if res.shape == (1, 1, 1):
        return res.squeeze(0)
    return res.squeeze()


def gradient_xAx(x, A):
    res = 2 * torch.einsum("...ij, ...jk -> ...ik", A, x.unsqueeze(-1)).squeeze()
    return res


def gradient_zAz(x, A, W_latent, b_latent):
    batch_size, feature_dim = x.shape
    z = torch.einsum(
        "...ij, ...jk -> ...ik",
        x.unsqueeze(1),
        W_latent.T.unsqueeze(0).repeat(batch_size, 1, 1),
    ) + b_latent.unsqueeze(0).repeat(batch_size, 1, 1)
    Pz = torch.einsum("...ij, ...jk -> ...ik", A, z.transpose(-2, -1))
    res = (
        2
        * torch.einsum(
            "...ij, ...jk -> ...ik",
            Pz.transpose(-2, -1),
            W_latent.repeat(batch_size, 1, 1),
        ).squeeze()
    )
    return res


def scaling_function(d, dmax):
    scaling = torch.ones_like(d)  # Default to 1.0
    mask = d >= dmax
    scaling[mask] = 0
    mask = (d > 0) & (d < dmax)
    scaling[mask] = 0.5 * (1 + torch.cos(torch.pi * d[mask] / dmax))
    return scaling


def shape_loss(
    loss_fn,
    obs,
    x_offsets,
    A,
    target,
    batch_grad,
    x_setpoint,
    dV_scaling=0.2,
    x_threshold=0.2,
    dmax=0.2,
    W_latent=None,
    b_latent=None,
):
    batch_size, feature_dim = obs.size()

    x_batch_exp = obs.unsqueeze(1).expand(
        batch_size, batch_size, feature_dim
    )  # Shape: [batch_size, batch_size, feature_dim]
    x_offsets_exp = x_offsets.unsqueeze(1).expand(
        batch_size, batch_size, feature_dim
    )  # Shape: [batch_size, batch_size, feature_dim]

    diff_i = x_batch_exp - x_batch_exp.transpose(
        0, 1
    )  # Shape: [batch_size, batch_size, feature_dim]
    scaling = scaling_function(
        torch.norm(diff_i, dim=-1), dmax
    )  # Shape: [batch_size, batch_size]
    # ! TODO: add latent here
    if W_latent is not None and b_latent is not None:
        x_bar = (
            x_batch_exp.transpose(0, 1) - x_offsets_exp
        )  # Shape: [batch_size, batch_size, feature_dim]
        pred_values_grad = torch.vmap(gradient_zAz)(
            x_bar,
            A,
            W_latent.unsqueeze(0).expand(batch_size, -1, -1),
            b_latent.unsqueeze(0).expand(batch_size, -1),
        )  # Shape: [batch_size, batch_size]
        diff_j = torch.einsum(
            "...ij, ...jk -> ...ik",
            x_bar,
            W_latent.T.unsqueeze(0).expand(batch_size, -1, -1),
        ) + b_latent.unsqueeze(0).expand(batch_size, -1)
    else:
        diff_j = (
            x_batch_exp.transpose(0, 1) - x_offsets_exp
        )  # Shape: [batch_size, batch_size, feature_dim]
        pred_values_grad = torch.vmap(gradient_xAx)(
            diff_j, A
        )  # Shape: [batch_size, batch_size]

    pred_values = torch.vmap(quadratify_xAx)(
        diff_j, A
    )  # Shape: [batch_size, batch_size]

    V_losses = loss_fn(
        pred_values, target.unsqueeze(0).expand(batch_size, batch_size)
    )  # Shape: [batch_size, batch_size]
    dV_losses = loss_fn(
        pred_values_grad,
        batch_grad.unsqueeze(0).expand(batch_size, batch_size, feature_dim),
    ).mean(dim=-1)  # Shape: [batch_size, batch_size, feature_dim]
    scaled_losses = scaling * (
        V_losses + dV_scaling * dV_losses
    )  # Element-wise scaling

    nonzero_counts = torch.count_nonzero(scaling, dim=1).clamp(
        min=1
    )  # Shape: [batch_size, 1]
    normalized_losses = (
        scaled_losses.sum(dim=1) / nonzero_counts.squeeze()
    )  # Shape: [batch_size]

    # extra loss penalizing x0offset when close to origin:
    offset_losses = loss_fn(x_offsets, x_setpoint)
    # Compute the distance between x_batch and x_setpoint
    distances = torch.norm(
        obs - x_setpoint, dim=1, keepdim=True
    )  # Shape: [batch_size, 1]
    # Compute the scaling factor
    scaling_factors = scaling_function(distances, x_threshold)  # Shape: [batch_size]
    # Apply scaling to the L1 losses
    scaled_losses = (
        scaling_factors * offset_losses
    )  # Broadcasting applies scaling factor per element

    loss = (
        normalized_losses.mean() + 10 * scaled_losses.mean()
    )  # Average over the batch size
    return loss


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class Critic(nn.Module):
    def __init__(
        self,
        num_obs,
        hidden_dims=None,
        activation="elu",
        normalize_obs=True,
        c_offset=False,
        device="cuda",
        **kwargs,
    ):
        super().__init__()

        self.NN = create_MLP(num_obs, 1, hidden_dims, activation)
        self._normalize_obs = normalize_obs
        if self._normalize_obs:
            self.obs_rms = RunningMeanStd(num_obs)
        self.value_offset = (
            nn.Parameter(torch.zeros(1, device=device))
            if c_offset
            else torch.zeros(1, device=device)
        )

    def forward(self, x, return_all=False):
        if self._normalize_obs:
            with torch.no_grad():
                x = self.obs_rms(x)
        value = self.NN(x).squeeze() + self.value_offset
        if return_all:
            return {"value": value}
        else:
            return value

    def evaluate(self, critic_observations, return_all=False):
        return self.forward(critic_observations, return_all=return_all)

    def loss_fn(self, obs, target, **kwargs):
        return nn.functional.mse_loss(self.forward(obs), target, reduction="mean")


class OuterProduct(nn.Module):
    def __init__(
        self,
        num_obs,
        hidden_dims,
        latent_dim=None,
        activation="elu",
        dropouts=None,
        normalize_obs=False,
        c_offset=False,
        minimize=False,
        offset_hidden_dims=None,
        loss="mse_loss",
        loss_type="standard",
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
        self.value_offset = (
            nn.Parameter(torch.zeros(1, device=device))
            if c_offset
            else torch.zeros(1, device=device)
        )
        self.loss = loss
        self.loss_type = loss_type

        self._normalize_obs = normalize_obs
        if self._normalize_obs:
            self.obs_rms = RunningMeanStd(num_obs)

        self.NN = create_MLP(num_obs, latent_dim, hidden_dims, activation, dropouts)
        self.offset_NN = (
            create_MLP(num_obs, num_obs, offset_hidden_dims, "relu")
            if offset_hidden_dims
            else lambda x: torch.zeros_like(x)
        )

    def forward(self, x, return_all=False):
        z = self.NN(x)
        A = z.unsqueeze(-1) @ z.unsqueeze(-2)
        # A2 = torch.einsum("nmx,nmy->nmxy", z, z)
        x_offsets = self.offset_NN(x)
        value = self.sign * quadratify_xAx(x - x_offsets, A)

        if return_all:
            return {"value": value, "A": A, "x_offsets": x_offsets}
        return value

    def evaluate(self, obs, return_all=False):
        return self.forward(obs, return_all)

    def loss_fn(self, obs, target, **kwargs):
        fn = eval(f"F.{self.loss}")
        output = self.forward(obs, return_all=True)
        value = output["value"]
        if self.loss_type == "standard":
            return fn(value, target, reduction="mean")
        elif self.loss_type == "sobol":
            if "batch_grad" not in kwargs.keys():
                raise ValueError("Sobol loss called with missing kwargs.")
            pred_grad = gradient_xAx(obs - output["x_offsets"], output["A"])
            return fn(value, target) + fn(pred_grad, kwargs["batch_grad"])
        elif self.loss_type == "shape":
            if "batch_grad" not in kwargs.keys():
                raise ValueError("Shape loss called with missing kwargs.")
            return shape_loss(
                fn,
                obs,
                output["x_offsets"],
                output["A"],
                target,
                kwargs["batch_grad"],
                torch.zeros_like(output["x_offsets"]),
            )
        else:
            raise ValueError("Loss type unspecified")


class OuterProductLatent(OuterProduct):
    def __init__(
        self,
        num_obs,
        hidden_dims,
        latent_dim=None,
        activation="elu",
        dropouts=None,
        normalize_obs=False,
        c_offset=False,
        minimize=False,
        latent_hidden_dims=[128, 128],
        latent_activation="tanh",
        offset_hidden_dims=None,
        loss="mse_loss",
        loss_type="standard",
        device="cuda",
        **kwargs,
    ):
        super().__init__(
            num_obs,
            hidden_dims,
            latent_dim,
            activation,
            dropouts,
            normalize_obs,
            c_offset,
            minimize,
            loss,
            loss_type,
            device,
            **kwargs,
        )
        self.latent_NN = create_MLP(
            num_obs,
            latent_dim,
            latent_hidden_dims,
            latent_activation,
            # bias_in_linear_layers=False,
        )

    def forward(self, x, return_all=False):
        x_offsets = self.offset_NN(x)
        z = self.NN(x)
        # outer product. Both of these are equivalent
        A = z.unsqueeze(-1) @ z.unsqueeze(-2)
        # A2 = torch.einsum("nmx,nmy->nmxy", z, z)
        value = self.sign * quadratify_xAx(self.latent_NN(x - x_offsets), A)
        value += self.value_offset

        if return_all:
            # return value, A, L
            return {"value": value, "A": A, "z": z, "x_offsets": x_offsets}
        else:
            return value


class CholeskyInput(nn.Module):
    def __init__(
        self,
        num_obs,
        hidden_dims,
        latent_dim=None,
        activation="elu",
        dropouts=None,
        normalize_obs=False,
        c_offset=False,
        minimize=False,
        offset_hidden_dims=None,
        loss="mse_loss",
        loss_type="standard",
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
        self.value_offset = (
            nn.Parameter(torch.zeros(1, device=device))
            if c_offset
            else torch.zeros(1, device=device)
        )
        # V = x'Ax + b = a* x'Ax + b, and add a regularization for det(A) ~= 1
        self.loss = loss
        self.loss_type = loss_type

        self._normalize_obs = normalize_obs
        if self._normalize_obs:
            self.obs_rms = RunningMeanStd(num_obs)

        num_lower_diag_elements = sum(range(latent_dim + 1))
        self.lower_diag_NN = create_MLP(
            num_obs, num_lower_diag_elements, hidden_dims, activation, dropouts
        )
        self.offset_NN = (
            create_MLP(num_obs, num_obs, offset_hidden_dims, "relu")
            if offset_hidden_dims
            else lambda x: torch.zeros_like(x)
        )
        # self.lower_diag_NN.apply(init_weights)

    def forward(self, x, return_all=False):
        output = self.lower_diag_NN(x)
        L = create_lower_diagonal(output, self.latent_dim, self.device)
        A = compose_cholesky(L)
        x_offsets = self.offset_NN(x)
        value = self.sign * quadratify_xAx(x - x_offsets, A)
        value += self.value_offset
        # value *= self.scaling_quadratic

        if return_all:
            # return value, A, L
            return {"value": value, "A": A, "L": L, "x_offsets": x_offsets}
        else:
            return value
        # return self.sign * quadratify_xAx(x, A) + self.value_offset

    def evaluate(self, obs, return_all=False):
        return self.forward(obs, return_all)

    def loss_fn(self, obs, target, **kwargs):
        fn = eval(f"F.{self.loss}")
        output = self.forward(obs, return_all=True)
        value = output["value"]
        if self.loss_type == "standard":
            return fn(value, target, reduction="mean")
        elif self.loss_type == "sobol":
            if "batch_grad" not in kwargs.keys():
                raise ValueError("Sobol loss called with missing kwargs.")
            pred_grad = gradient_xAx(obs - output["x_offsets"], output["A"])
            return fn(value, target) + fn(pred_grad, kwargs["batch_grad"])
        elif self.loss_type == "shape":
            if "batch_grad" not in kwargs.keys():
                raise ValueError("Shape loss called with missing kwargs.")
            return shape_loss(
                fn,
                obs,
                output["x_offsets"],
                output["A"],
                target,
                kwargs["batch_grad"],
                torch.zeros_like(output["x_offsets"]),
            )
        else:
            raise ValueError("Loss type unspecified")


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
        c_offset=False,
        minimize=False,
        offset_hidden_dims=None,
        loss="mse_loss",
        loss_type="standard",
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
            c_offset=c_offset,
            minimize=minimize,
            loss=loss,
            loss_type=loss_type,
            device="cuda",
            **kwargs,
        )

        self.latent_NN = create_MLP(
            num_obs,
            latent_dim,
            latent_hidden_dims,
            latent_activation,
            # bias_in_linear_layers=False,
        )

        # self.latent_NN.apply(init_weights)

    def forward(self, x, return_all=False):
        x_offsets = self.offset_NN(x)
        z = self.latent_NN(x - x_offsets)
        output = self.lower_diag_NN(x)
        L = create_lower_diagonal(output, self.latent_dim, self.device)
        A = compose_cholesky(L)

        value = self.sign * quadratify_xAx(z, A)
        # do not affect value offset in this part of the loss
        value += self.value_offset
        # value *= self.scaling_quadratic

        if return_all:
            # return value, A, L, z
            return {"value": value, "A": A, "L": L, "z": z, "x_offsets": x_offsets}
        else:
            return value

    def evaluate(self, obs, return_all=False):
        return self.forward(obs, return_all)

    def loss_fn(self, obs, target, **kwargs):
        fn = eval(f"F.{self.loss}")
        output = self.forward(obs, return_all=True)
        value = output["value"]
        if self.loss_type == "standard":
            return fn(value, target, reduction="mean")
        elif self.loss_type == "sobol":
            if not {"batch_grad", "W_latent", "b_latent"}.issubset(kwargs.keys()):
                raise ValueError("Sobol loss called with missing kwargs.")
            pred_grad = gradient_zAz(
                obs - output["x_offsets"],
                output["A"],
                kwargs["W_latent"],
                kwargs["b_latent"],
            )
            return fn(value, target) + fn(pred_grad, kwargs["batch_grad"])
        elif self.loss_type == "shape":
            if not {"batch_grad", "W_latent", "b_latent"}.issubset(set(kwargs.keys())):
                raise ValueError("Shape loss called with missing kwargs.")
            return shape_loss(
                fn,
                obs,
                output["x_offsets"],
                output["A"],
                target,
                kwargs["batch_grad"],
                torch.zeros_like(output["x_offsets"]),
                W_latent=kwargs["W_latent"],
                b_latent=kwargs["b_latent"],
            )
        else:
            raise ValueError("Loss type unspecified")


class PDCholeskyInput(CholeskyInput):
    def forward(self, x, return_all=False):
        x_offsets = self.offset_NN(x)
        output = self.lower_diag_NN(x)
        L = create_PD_lower_diagonal(output, self.latent_dim, self.device)
        A = compose_cholesky(L)
        # assert (torch.linalg.eigvals(A).real > 0).all()
        value = self.sign * quadratify_xAx(x - x_offsets, A)

        if return_all:
            # return value, A, L
            return {"value": value, "A": A, "L": L, "x_offsets": x_offsets}
        else:
            return value
        # return self.sign * quadratify_xAx(x, A) + self.value_offset


class PDCholeskyLatent(CholeskyLatent):
    def forward(self, x, return_all=False):
        x_offsets = self.offset_NN(x)
        z = self.latent_NN(x - x_offsets)
        output = self.lower_diag_NN(x)
        L = create_PD_lower_diagonal(output, self.latent_dim, self.device)
        A = compose_cholesky(L)
        # assert (torch.linalg.eigvals(A).real > 0).all()
        value = self.sign * quadratify_xAx(z, A)
        value += self.value_offset
        # value *= self.scaling_quadratic

        if return_all:
            # return value, A, L
            return {"value": value, "A": A, "L": L, "x_offsets": x_offsets}
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
        c_offset=False,
        minimize=False,
        relative_dim=None,
        latent_dim=None,
        latent_hidden_dims=[64],
        latent_activation="tanh",
        latent_dropouts=None,
        offset_hidden_dims=None,
        loss="mse_loss",
        loss_type="standard",
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
        self.value_offset = (
            nn.Parameter(torch.zeros(1, device=device))
            if c_offset
            else torch.zeros(1, device=device)
        )
        self.loss = loss
        self.loss_type = loss_type

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
            num_obs,
            latent_dim,
            latent_hidden_dims,
            latent_activation,
            # bias_in_linear_layers=False,
        )
        self.offset_NN = (
            create_MLP(num_obs, num_obs, offset_hidden_dims, "relu")
            if offset_hidden_dims
            else lambda x: torch.zeros_like(x)
        )

    def forward(self, x, return_all=False):
        x_offsets = self.offset_NN(x)
        z = self.latent_NN(x - x_offsets)
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
        # with torch.no_grad():
        value += self.value_offset

        if return_all:
            # return value, z, torch.diag(A_diag), L, A
            return {
                "value": value,
                "z": z,
                "L": L,
                "A": A,
                "x_offsets": x_offsets,
            }
        else:
            return value

    def evaluate(self, obs, return_all=False):
        return self.forward(obs, return_all)

    def loss_fn(self, obs, target, **kwargs):
        fn = eval(f"F.{self.loss}")
        output = self.forward(obs, return_all=True)
        value = output["value"]
        if self.loss_type == "standard":
            return fn(value, target, reduction="mean")
        elif self.loss_type == "sobol":
            if not {"batch_grad", "W_latent", "b_latent"}.issubset(kwargs.keys()):
                raise ValueError("Sobol loss called with missing kwargs.")
            pred_grad = gradient_zAz(
                obs - output["x_offsets"],
                output["A"],
                kwargs["W_latent"],
                kwargs["b_latent"],
            )
            return fn(value, target) + fn(pred_grad, kwargs["batch_grad"])
        elif self.loss_type == "shape":
            if "batch_grad" not in kwargs.keys():
                raise ValueError("Shape loss called with missing kwargs.")
            return shape_loss(
                fn,
                obs,
                output["x_offsets"],
                output["A"],
                target,
                kwargs["batch_grad"],
                torch.zeros_like(output["x_offsets"]),
            )
        else:
            raise ValueError("Loss type unspecified")


class DenseSpectralLatent(nn.Module):
    def __init__(
        self,
        num_obs,
        hidden_dims,
        activation="elu",
        dropouts=None,
        normalize_obs=False,
        c_offset=False,
        minimize=False,
        relative_dim=None,
        latent_dim=None,
        latent_hidden_dims=[64],
        latent_activation="tanh",
        latent_dropouts=None,
        offset_hidden_dims=None,
        loss="mse_loss",
        loss_type="standard",
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
        self.value_offset = (
            nn.Parameter(torch.zeros(1, device=device))
            if c_offset
            else torch.zeros(1, device=device)
        )
        self.loss = loss
        self.loss_type = loss_type

        self._normalize_obs = normalize_obs
        if self._normalize_obs:
            self.obs_rms = RunningMeanStd(num_obs)
        # TODO up to here, same as Cholesky, can be refactored
        n_zeros = sum(range(relative_dim))
        n_outputs = relative_dim + relative_dim * latent_dim - n_zeros
        self.rel_fill = n_zeros
        # n_outputs = (
        #     relative_dim * latent_dim
        #     - torch.tril_indices(self.relative_dim, self.latent_dim).shape[1]
        #     + relative_dim
        # )
        self.spectral_NN = create_MLP(
            num_obs, n_outputs, hidden_dims, activation, dropouts
        )
        self.latent_NN = create_MLP(
            num_obs,
            latent_dim,
            latent_hidden_dims,
            latent_activation,
            # bias_in_linear_layers=False,
        )
        self.offset_NN = (
            create_MLP(num_obs, num_obs, offset_hidden_dims, "relu")
            if offset_hidden_dims
            else lambda x: torch.zeros_like(x)
        )

    def forward(self, x, return_all=False):
        x_offsets = self.offset_NN(x)
        z = self.latent_NN(x - x_offsets)
        y = self.spectral_NN(x)
        A_diag = torch.diag_embed(F.softplus(y[..., : self.relative_dim]))
        # tril_indices = torch.tril_indices(self.latent_dim, self.relative_dim)
        tril_indices = torch.tril_indices(self.relative_dim, self.latent_dim)
        L = torch.zeros(
            (*x.shape[:-1], self.relative_dim, self.latent_dim), device=self.device
        )
        L[..., tril_indices[0], tril_indices[1]] = y[
            ..., self.relative_dim : self.relative_dim + tril_indices.shape[1]
        ]
        L[..., :, self.relative_dim :] = y[
            ..., self.relative_dim + tril_indices.shape[1] :
        ].view(L[..., :, self.relative_dim :].shape)

        # Compute (L^T A_diag) L
        A = torch.einsum(
            "...ij,...jk->...ik",
            torch.einsum("...ik,...kj->...ij", L.transpose(-1, -2), A_diag),
            L,
        )
        # A = torch.einsum(
        #     "...ij,...jk->...ik",
        #     torch.einsum("...ik,...kj->...ij", L, A_diag),
        #     L.transpose(-1, -2),
        # )
        # assert (torch.linalg.eigvals(A).real >= -1e-6).all()
        # ! This fails, but so far with just -e-10, so really really small...
        value = self.sign * quadratify_xAx(z, A)
        # with torch.no_grad():
        value += self.value_offset

        if return_all:
            # return value, z, torch.diag(A_diag), L, A
            return {
                "value": value,
                "z": z,
                # "A_diag": torch.diag(A_diag), #! throws shape error
                "L": L,
                "A": A,
                "x_offsets": x_offsets,
            }
        else:
            return value

    def evaluate(self, obs, return_all=False):
        return self.forward(obs, return_all)

    def loss_fn(self, obs, target, **kwargs):
        fn = eval(f"F.{self.loss}")
        output = self.forward(obs, return_all=True)
        value = output["value"]
        if self.loss_type == "standard":
            return fn(value, target, reduction="mean")
        elif self.loss_type == "sobol":
            if not {"batch_grad", "W_latent", "b_latent"}.issubset(kwargs.keys()):
                raise ValueError("Sobol loss called with missing kwargs.")
            pred_grad = gradient_zAz(
                obs - output["x_offsets"],
                output["A"],
                kwargs["W_latent"],
                kwargs["b_latent"],
            )
            return fn(value, target) + fn(pred_grad, kwargs["batch_grad"])
        elif self.loss_type == "shape":
            if "batch_grad" not in kwargs.keys():
                raise ValueError("Shape loss called with missing kwargs.")
            return shape_loss(
                fn,
                obs,
                output["x_offsets"],
                output["A"],
                target,
                kwargs["batch_grad"],
                torch.zeros_like(output["x_offsets"]),
                W_latent=kwargs["W_latent"],
                b_latent=kwargs["b_latent"],
            )
        else:
            raise ValueError("Loss type unspecified")


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
        # self.lower_diag_NN_Q.apply(init_weights)

        # Make R estimator
        num_lower_diag_elements_R = sum(range(action_dim + 1))
        self.lower_diag_NN_R = create_MLP(
            latent_dim, num_lower_diag_elements_R, hidden_dims, activation, dropouts
        )
        # self.lower_diag_NN_R.apply(init_weights)

    def forward(self, z, u, return_all=False):
        Q_lower_diag = self.lower_diag_NN_Q(z)
        L_Q = create_lower_diagonal(Q_lower_diag, self.latent_dim, self.device)
        Q = compose_cholesky(L_Q)

        R_lower_diag = self.lower_diag_NN_R(z)
        L_R = create_lower_diagonal(R_lower_diag, self.action_dim, self.device)
        R = compose_cholesky(L_R)

        if return_all:
            # return Q, L_Q, R, L_R
            return {"Q": Q, "L_Q": L_Q, "R": R, "L_R": L_R}
        else:
            return Q, R

    def evaluate(self, obs, return_all=False):
        return self.forward(obs, return_all)

    def loss_fn(self, z, target, actions, **kwargs):
        Q, R = self.forward(z, target)
        Q_value = self.sign * quadratify_xAx(z, Q)
        R_value = self.sign * quadratify_xAx(actions, R)
        gae_loss = F.mse_loss(Q_value + R_value, target, reduction="mean")
        return gae_loss

    def riccati_loss_fn(self, z, target, P, **kwargs):
        Q, R = self.forward(z, target)
        A, B = self.linearize_pendulum_dynamics()
        A = A.to(self.device)
        B = B.to(self.device)
        AT_P_A = torch.einsum(
            "...ij, ...ik -> ...ik",
            torch.einsum("...ij, ...jk -> ...ik", A.transpose(-1, -2), P),
            A,
        )
        AT_P_B = torch.einsum(
            "...ij, ...ik -> ...ik",
            torch.einsum("...ij, ...jk -> ...ik", A.transpose(-1, -2), P),
            B,
        )
        R_BT_B = R + torch.einsum("...ij, ...jk -> ...ik", B.transpose(-1, -2), B)
        BT_P_A = torch.einsum(
            "...ij, ...ik -> ...ik",
            torch.einsum("...ij, ...jk -> ...ik", B.transpose(-1, -2), P),
            A,
        )
        final_term = torch.einsum(
            "...ij, ...jk -> ...ik",
            torch.einsum("...ij, ...jk -> ...ik", AT_P_B, torch.linalg.inv(R_BT_B)),
            BT_P_A,
        )
        riccati_loss = Q + AT_P_A - P - final_term
        return F.mse_loss(riccati_loss, target=0.0, reduction="mean")

    def linearize_pendulum_dynamics(self, x_desired=torch.tensor([0.0, 0.0])):
        m = 1.0
        b = 0.1
        length = 1.0
        g = 9.81
        ml2 = m * length**2

        A = torch.tensor(
            [[0.0, 1.0], [g / length * torch.cos(x_desired[0]), -b / ml2]],
            device=self.device,
        )
        B = torch.tensor([[0.0], [(1.0 / ml2)]], device=self.device)
        return A, B


class QPNet(nn.Module):
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

        self.num_lower_diag_elements = sum(range(latent_dim + 1))
        self.lower_diag_NN = create_MLP(
            num_obs,
            self.num_lower_diag_elements + num_obs,
            hidden_dims,
            activation,
            dropouts,
        )  # ! num_obs added to output to create c for c.T@x
        # self.lower_diag_NN.apply(init_weights)

    def forward(self, x, return_all=False):
        res = self.lower_diag_NN(x)
        output = res[..., : self.num_lower_diag_elements]
        c = res[..., self.num_lower_diag_elements :]
        L = create_lower_diagonal(output, self.latent_dim, self.device)
        A = compose_cholesky(nn.ReLU()(L))
        value = self.sign * quadratify_xAx(x, A) + torch.einsum(
            "...ij, ...ij -> ...i", c, x
        )
        value += self.value_offset
        # value *= self.scaling_quadratic

        if return_all:
            # return value, A, L
            return {"value": value, "A": A, "L": L, "c": c}
        else:
            return value

    def evaluate(self, obs, return_all=False):
        return self.forward(obs, return_all)

    def loss_fn(self, obs, target, **kwargs):
        loss_NN = F.mse_loss(self.forward(obs), target, reduction="mean")

        # loss_scaling = (self.scaling_quadratic - target.mean()).pow(2)

        return loss_NN  #  + loss_offset + loss_scaling


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

    def evaluate(self, obs, return_all=False):
        return self.forward(obs, return_all)

    def loss_fn(self, obs, target, actions, **kwargs):
        return self.value_loss(obs, target, actions) + self.reg_loss(
            obs, target, actions
        )

    def value_loss(self, obs, target, actions, **kwargs):
        return self.critic.loss_fn(obs, target, actions=actions)

    def reg_loss(self, obs, target, actions, **kwargs):
        with torch.no_grad():
            P = self.critic(obs, return_all=True)["A"]
        z = obs if self.has_latent is False else self.critic.latent_NN(obs)
        return self.QR_network.riccati_loss_fn(z, target, P=P)


class NN_wQR(nn.Module):
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

    def evaluate(self, obs, return_all=False):
        return self.forward(obs, return_all)

    def loss_fn(self, obs, target, actions, **kwargs):
        return self.value_loss(obs, target, actions) + self.reg_loss(
            obs, target, actions
        )

    def value_loss(self, obs, target, actions, **kwargs):
        return self.critic.loss_fn(obs, target, actions=actions)

    def reg_loss(self, obs, target, actions, **kwargs):
        z = obs if self.has_latent is False else self.critic.latent_NN(obs)
        return self.QR_network.loss_fn(z, target, actions=actions)


class NN_wLinearLatent(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        critic_name = kwargs["critic_name"]
        device = kwargs["device"]  # noqa
        self.critic = eval(f"{critic_name}(**kwargs).to(device)")

    @property
    def value_offset(self):
        return self.critic.value_offset

    def forward(self, x, return_all=False):
        return self.critic.forward(x, return_all)

    def evaluate(self, obs, return_all=False):
        return self.forward(obs, return_all)

    def loss_fn(self, obs, target, actions, **kwargs):
        return self.value_loss(obs, target, actions) + self.reg_loss(
            obs, target, actions
        )

    def value_loss(self, obs, target, actions, **kwargs):
        return self.critic.loss_fn(obs, target, actions=actions)

    def reg_loss(self, obs, target, actions, **kwargs):
        z = self.critic.latent_NN(obs)
        u = actions.detach().clone()
        u = u.view(-1, u.shape[-1])
        K, z_offset = least_squares_fit(z, u)
        return F.mse_loss(forward_affine(z, K, z_offset), target=u, reduction="mean")
