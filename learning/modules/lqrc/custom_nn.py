from math import floor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class BaselineMLP(nn.Module):
    def __init__(self, input_size, output_size=1, device="cuda"):
        super(BaselineMLP, self).__init__()
        layer_width = 32
        self.connection_1 = nn.Linear(input_size, 4 * layer_width)
        self.activation_1 = nn.ELU()
        self.connection_2 = nn.Linear(4 * layer_width, layer_width)
        self.activation_2 = nn.ELU()
        self.connection_3 = nn.Linear(layer_width, output_size)
        self.activation_3 = nn.Softplus()
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
        return output


class QuadraticNetCholesky(BaselineMLP):
    def __init__(self, input_size, device="cuda"):
        super(QuadraticNetCholesky, self).__init__(
            input_size, sum(range(input_size + 1)), device=device
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
        A = C.bmm(C.transpose(1, 2))
        y_pred = (x.unsqueeze(2).transpose(1, 2).bmm(A).bmm(x.unsqueeze(2))).squeeze(2)
        return y_pred

    def create_cholesky(self, x):
        batch_size = x.shape[0]
        n = self.input_size
        L = torch.zeros((batch_size, n, n), device=self.device)
        tril_indices = torch.tril_indices(row=n, col=n, offset=0)
        rows, cols = tril_indices
        L[:, rows, cols] = x

        ####
        # tril_indices = torch.tril_indices(row=n, col=n, offset=0)
        # rows, cols = tril_indices
        # for i in range(batch_size):
        #     L[i, rows, cols] = x[i]
        ####
        # idx = 0
        # for i in range(self.input_size):
        #     for j in range(self.input_size):
        #         if i >= j:
        #             L2[:, i, j] = x[:, idx]
        #             idx += 1
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
    def __init__(self, input_size, device="cuda"):
        # additional 1 to output_size for +c
        super(QuadraticNetCholesky, self).__init__(
            input_size, sum(range(input_size + 1)) + 1, device=device
        )
        # self.activation_3.register_forward_hook(self.save_intermediate())

    def forward(self, x):
        output = self.connection_1(x)
        output = self.activation_1(output)
        output = self.connection_2(output)
        output = self.activation_2(output)
        output = self.connection_3(output)
        output = self.activation_3(output)
        C = self.create_cholesky(output[:, :-1])
        A = C.bmm(C.transpose(1, 2))
        c = output[:, -1]
        y_pred = (x.unsqueeze(2).transpose(1, 2).bmm(A).bmm(x.unsqueeze(2))).squeeze(
            2
        ) + c.unsqueeze(1)
        return y_pred

    # def save_intermediate(self):
    #     """
    #     Forward hook to save A and c
    #     """

    #     def hook(module, input, output):
    #         C = self.create_cholesky(output[:, :-1])
    #         A = C.bmm(C.transpose(1, 2))
    #         c = output[:, -1]
    #         self.intermediates["A"] = A
    #         self.intermediates["c"] = c

    #     return hook


class CustomCholeskyPlusConstLoss(nn.Module):
    def __init__(self, const_penalty=0.0):
        super().__init__()
        self.const_penalty = const_penalty

    def forward(self, y_pred, y, intermediates=None):
        loss = F.mse_loss(y_pred, y, reduction="mean")
        const_loss = self.const_penalty * torch.mean(intermediates["c"])
        return loss + const_loss


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
