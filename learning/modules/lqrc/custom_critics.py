import torch
import torch.nn as nn
import torch.nn.functional as F

from learning.modules.critic import Critic
from learning.modules.utils import RunningMeanStd
from learning.modules.utils.neural_net import get_activation


class CustomCriticBaseline(Critic):
    def __init__(
        self,
        num_obs,
        hidden_dims=None,
        activation="elu",
        normalize_obs=True,
        output_size=1,
        device="cuda",
        **kwargs,
    ):
        super(Critic, self).__init__()
        hidden_dims = [32, 128] if hidden_dims is None else hidden_dims
        assert len(hidden_dims) == 2, "Too many hidden dims passed to Custom Critic"
        activation = get_activation(activation)
        self.input_size = num_obs
        self.output_size = output_size
        self.device = device
        self.intermediates = {}

        self._normalize_obs = normalize_obs
        if self._normalize_obs:
            self.obs_rms = RunningMeanStd(num_obs)

        self.connection_1 = nn.Linear(self.input_size, hidden_dims[0])
        self.activation_1 = activation
        self.connection_2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.activation_2 = activation
        self.connection_3 = nn.Linear(hidden_dims[1], output_size)
        self.activation_3 = nn.Softplus()

        self.NN = self.forward

    def forward(self, x):
        output = self.connection_1(x)
        output = self.activation_1(output)
        output = self.connection_2(output)
        output = self.activation_2(output)
        output = self.connection_3(output)
        output = self.activation_3(output)
        return output


class Cholesky(CustomCriticBaseline):
    def __init__(
        self,
        num_obs,
        hidden_dims=None,
        activation="elu",
        normalize_obs=True,
        output_size=1,
        device="cuda",
        **kwargs,
    ):
        super().__init__(
            num_obs,
            hidden_dims=hidden_dims,
            activation=activation,
            normalize_obs=normalize_obs,
            output_size=sum(range(num_obs + 1)),
            device=device,
        )
        self.activation_3.register_forward_hook(self.save_intermediate())

    def forward(self, x):
        output = self.connection_1(x)
        output = self.activation_1(output)
        output = self.connection_2(output)
        output = self.activation_2(output)
        output = self.connection_3(output)
        output = self.activation_3(output)
        C = self.create_cholesky(output)
        A = torch.einsum("...ij, ...jk -> ...ik", C, C.transpose(-2, -1))
        y_pred = torch.einsum(
            "...ij, ...jk -> ...ik",
            torch.einsum("...ij, ...jk -> ...ik", x.unsqueeze(-1).transpose(-2, -1), A),
            x.unsqueeze(-1),
        ).squeeze(-1)
        return y_pred

    def create_cholesky(self, x):
        n = self.input_size
        L = torch.zeros((*x.shape[:-1], n, n), device=self.device, requires_grad=False)
        tril_indices = torch.tril_indices(row=n, col=n, offset=0)
        rows, cols = tril_indices
        L[..., rows, cols] = x
        return L

    def save_intermediate(self):
        def hook(module, input, output):
            C = self.create_cholesky(output)
            A = torch.einsum("...ij, ...jk -> ...ik", C, C.transpose(-2, -1))
            self.intermediates["A"] = A

        return hook


class CholeskyPlusConst(Cholesky):
    def __init__(
        self,
        num_obs,
        hidden_dims=None,
        activation="elu",
        normalize_obs=True,
        output_size=1,
        device="cuda",
        **kwargs,
    ):
        # additional 1 to output_size for +c
        super(Cholesky, self).__init__(
            num_obs=num_obs,
            hidden_dims=hidden_dims,
            activation=activation,
            normalize_obs=normalize_obs,
            output_size=sum(range(num_obs + 1)) + 1,
            device=device,
        )
        self.activation_3.register_forward_hook(self.save_intermediate())
        # 0.0 if kwargs.get("const_penalty") is None else kwargs.get("const_penalty")
        self.const_penalty = 0.1

    def forward(self, x):
        output = self.connection_1(x)
        output = self.activation_1(output)
        output = self.connection_2(output)
        output = self.activation_2(output)
        output = self.connection_3(output)
        output = self.activation_3(output)
        C = self.create_cholesky(output[..., :-1])
        A = torch.einsum("...ij, ...jk -> ...ik", C, C.transpose(-2, -1))
        c = output[..., -1]
        y_pred = torch.einsum(
            "...ij, ...jk -> ...ik",
            torch.einsum("...ij, ...jk -> ...ik", x.unsqueeze(-1).transpose(-2, -1), A),
            x.unsqueeze(-1),
        ) + c.unsqueeze(-1).unsqueeze(-1)
        return y_pred.squeeze(-1)

    def save_intermediate(self):
        def hook(module, input, output):
            C = self.create_cholesky(output[..., :-1])
            A = torch.einsum("...ij, ...jk -> ...ik", C, C.transpose(-2, -1))
            c = output[..., -1]
            self.intermediates["A"] = A
            self.intermediates["c"] = c

        return hook

    def loss_fn(self, input, target):
        loss = F.mse_loss(input, target, reduction="mean")
        const_loss = self.const_penalty * torch.mean(self.intermediates["c"])
        return loss + const_loss


class CholeskyOffset1(Cholesky):
    def __init__(
        self,
        num_obs,
        hidden_dims=None,
        activation="elu",
        normalize_obs=True,
        output_size=1,
        device="cuda",
        **kwargs,
    ):
        # + input_size for offset
        super(Cholesky, self).__init__(
            num_obs=num_obs,
            hidden_dims=hidden_dims,
            activation=activation,
            normalize_obs=normalize_obs,
            output_size=sum(range(num_obs + 1)) + num_obs,
            device=device,
        )
        self.L_indices = sum(range(num_obs + 1))
        self.activation_3.register_forward_hook(self.save_intermediate())

    def forward(self, x):
        output = self.connection_1(x)
        output = self.activation_1(output)
        output = self.connection_2(output)
        output = self.activation_2(output)
        output = self.connection_3(output)
        output = self.activation_3(output)
        C = self.create_cholesky(output[..., : self.L_indices])
        A = torch.einsum("...ij, ...jk -> ...ik", C, C.transpose(-2, -1))
        offset = output[..., self.L_indices :]
        x_bar = x - offset
        y_pred = torch.einsum(
            "...ij, ...jk -> ...ik",
            torch.einsum(
                "...ij, ...jk -> ...ik", x_bar.unsqueeze(-1).transpose(-2, -1), A
            ),
            x_bar.unsqueeze(-1),
        ).squeeze(-1)
        return y_pred

    def save_intermediate(self):
        def hook(module, input, output):
            C = self.create_cholesky(output[..., : self.L_indices])
            A = torch.einsum("...ij, ...jk -> ...ik", C, C.transpose(-2, -1))
            offset = output[..., self.L_indices :]
            self.intermediates["A"] = A
            self.intermediates["offset"] = offset

        return hook


class CholeskyOffset2(Cholesky):
    def __init__(
        self,
        num_obs,
        hidden_dims=None,
        activation="elu",
        normalize_obs=True,
        output_size=1,
        device="cuda",
        **kwargs,
    ):
        super().__init__(num_obs)
        self.xhat_layer = nn.Linear(num_obs, num_obs)
        self.xhat_layer.register_forward_hook(self.save_xhat())

    def forward(self, x):
        xhat = self.xhat_layer(x)
        output = self.connection_1(xhat)
        output = self.activation_1(output)
        output = self.connection_2(output)
        output = self.activation_2(output)
        output = self.connection_3(output)
        output = self.activation_3(output)
        C = self.create_cholesky(output)
        A = torch.einsum("...ij, ...jk -> ...ik", C, C.transpose(-2, -1))
        y_pred = torch.einsum(
            "...ij, ...jk -> ...ik",
            torch.einsum(
                "...ij, ...jk -> ...ik", xhat.unsqueeze(-1).transpose(-2, -1), A
            ),
            xhat.unsqueeze(-1),
        ).squeeze(-1)
        return y_pred

    def save_intermediate(self):
        def hook(module, input, output):
            C = self.create_cholesky(output)
            A = torch.einsum("...ij, ...jk -> ...ik", C, C.transpose(-2, -1))
            self.intermediates["A"] = A

        return hook

    def save_xhat(self):
        def hook(module, input, output):
            self.intermediates["xhat"] = output

        return hook
