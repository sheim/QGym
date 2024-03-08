from math import floor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class BaselineMLP(nn.Module):
    def __init__(self, input_size, output_size=1, hidden_dims=128, device="cuda"):
        super(BaselineMLP, self).__init__()
        assert hidden_dims % 4 == 0, "Critic hidden dims must be divisible by 4!"
        self.connection_1 = nn.Linear(input_size, hidden_dims)
        self.activation_1 = nn.ELU()
        self.connection_2 = nn.Linear(
            hidden_dims, hidden_dims // 4
        )  # nn.Linear(hidden_dims, hidden_dims//2)
        self.activation_2 = nn.ELU()
        self.connection_3 = nn.Linear(
            hidden_dims // 4, output_size
        )  # nn.Linear(hidden_dims//2, hidden_dims//4)
        self.activation_3 = nn.Softplus()  # nn.ELU()
        # self.connection_4 = nn.Linear(hidden_dims//4, output_size)
        # self.activation_4 = nn.Softplus()
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.intermediates = {}

    def forward(self, x):
        output = self.connection_1(x)
        output = self.activation_1(output)
        output = self.connection_2(output)
        output = self.activation_2(output)
        output = self.connection_3(output)
        output = self.activation_3(output)
        # output = self.connection_4(output)
        # output = self.activation_4(output)
        return output


class QuadraticNetCholesky(BaselineMLP):
    def __init__(self, input_size, hidden_dims=128, device="cuda"):
        super().__init__(
            input_size,
            sum(range(input_size + 1)),
            hidden_dims=hidden_dims,
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
        # output = self.connection_4(output)
        # output = self.activation_4(output)
        C = self.create_cholesky(output)
        A = C.bmm(C.transpose(1, 2))
        y_pred = (x.unsqueeze(2).transpose(1, 2).bmm(A).bmm(x.unsqueeze(2))).squeeze(2)
        return y_pred

    def create_cholesky(self, x):
        batch_size = x.shape[0]
        n = self.input_size
        L = torch.zeros((batch_size, n, n), device=self.device, requires_grad=False)
        tril_indices = torch.tril_indices(row=n, col=n, offset=0)
        rows, cols = tril_indices
        L[:, rows, cols] = x
        return L

    def save_intermediate(self):
        """
        Forward hook to save A
        """

        def hook(module, input, output):
            C = self.create_cholesky(output)
            A = C.bmm(C.transpose(1, 2))
            self.intermediates["A"] = A

        return hook


class CustomCholeskyLoss(nn.Module):
    def __init__(self, diag_L2_loss=0.0, diag_nuclear_loss=0.0):
        super().__init__()
        self.diag_L2_loss = diag_L2_loss
        self.diag_nuclear_loss = diag_nuclear_loss

    def forward(self, y_pred, y, intermediates=None):
        loss = F.mse_loss(y_pred, y, reduction="mean")
        if intermediates:
            pass  # reconstruct Cholesky and do special matrix losses
        return loss


class CholeskyPlusConst(QuadraticNetCholesky):
    def __init__(self, input_size, hidden_dims=128, device="cuda"):
        # additional 1 to output_size for +c
        super(QuadraticNetCholesky, self).__init__(
            input_size,
            sum(range(input_size + 1)) + 1,
            hidden_dims=hidden_dims,
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
        # output = self.connection_4(output)
        # output = self.activation_4(output)
        C = self.create_cholesky(output[:, :-1])
        A = C.bmm(C.transpose(1, 2))
        c = output[:, -1]
        y_pred = (x.unsqueeze(2).transpose(1, 2).bmm(A).bmm(x.unsqueeze(2))).squeeze(
            2
        ) + c.unsqueeze(1)
        return y_pred

    def save_intermediate(self):
        """
        Forward hook to save A and c
        """

        def hook(module, input, output):
            C = self.create_cholesky(output[:, :-1])
            A = C.bmm(C.transpose(1, 2))
            c = output[:, -1]
            self.intermediates["A"] = A
            self.intermediates["c"] = c

        return hook


class CustomCholeskyPlusConstLoss(nn.Module):
    def __init__(self, const_penalty=0.0):
        super().__init__()
        self.const_penalty = const_penalty

    def forward(self, y_pred, y, intermediates=None):
        loss = F.mse_loss(y_pred, y, reduction="mean")
        const_loss = self.const_penalty * torch.mean(intermediates["c"])
        return loss + const_loss


class CholeskyOffset1(QuadraticNetCholesky):
    def __init__(self, input_size, hidden_dims=128, device="cuda"):
        # + input_size for offset
        super(QuadraticNetCholesky, self).__init__(
            input_size,
            sum(range(input_size + 1)) + input_size,
            hidden_dims=hidden_dims,
            device=device,
        )
        self.L_indices = sum(range(input_size + 1))
        self.activation_3.register_forward_hook(self.save_intermediate())

    def forward(self, x):
        output = self.connection_1(x)
        output = self.activation_1(output)
        output = self.connection_2(output)
        output = self.activation_2(output)
        output = self.connection_3(output)
        output = self.activation_3(output)
        C = self.create_cholesky(output[:, :self.L_indices])
        A = C.bmm(C.transpose(1, 2))
        offset = output[:, self.L_indices:]
        x_bar = x - offset
        y_pred = (x_bar.unsqueeze(2).transpose(1, 2).bmm(A).bmm(x_bar.unsqueeze(2))).squeeze(2)
        return y_pred

    def save_intermediate(self):
        """
        Forward hook to save A and offset
        """
        def hook(module, input, output):
            C = self.create_cholesky(output[:, :self.L_indices])
            A = C.bmm(C.transpose(1, 2))
            offset = output[:, self.L_indices:]
            self.intermediates["A"] = A
            self.intermediates["offset"] = offset

        return hook


class CholeskyOffset2(QuadraticNetCholesky):
    def __init__(self, input_size, hidden_dims=128, device="cuda"):
        super().__init__(input_size)
        self.xhat_layer = nn.Linear(input_size, input_size)
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
        A = C.bmm(C.transpose(1, 2))
        y_pred = (xhat.unsqueeze(2).transpose(1, 2).bmm(A).bmm(xhat.unsqueeze(2))).squeeze(2)
        return y_pred

    def save_intermediate(self):
        """
        Forward hook to save A
        """
        def hook(module, input, output):
            C = self.create_cholesky(output)
            A = C.bmm(C.transpose(1, 2))
            self.intermediates["A"] = A

        return hook

    def save_xhat(self):
        """
        Forward hook to save xhat for debugging
        """
        def hook(module, input, output):
            self.intermediates["xhat"] = output

        return hook


class LQRCDataset(Dataset):
    def __init__(self, X, y):
        """
        X is data_size by num_variables
        y is data_size by 1
        """
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ix):
        return self.X[ix, ...], self.y[ix]

    def get_train_test_split_len(self, train_frac, test_frac):
        return [floor(self.__len__() * train_frac), floor(self.__len__() * test_frac)]


# symmetry based network with the little heuristic adjustment from lyapunov net paper
# class QuadraticMatrices(nn.Module):
#     def __init__(self, input_size, output_size, alpha=0.1, device='cuda'):
#         super(QuadraticMatrices, self).__init__()
#         self.alpha = alpha

#     def forward(self, x):
#         A = "some matrix"
#         return x.T @ A @ x
