from math import floor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def create_cholesky(x, output_size, device):
    batch_size = x.shape[0]
    L = torch.zeros(batch_size, output_size, output_size, device=device)
    idx = 0
    for i in range(output_size):
        for j in range(output_size):
            if i >= j:
                L[:, i, j] = x[:, idx]
                idx += 1
    return L


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

    def save_intermediate(self, name):
        """
        Forward hook to save a given intermediate value
        to a class attribute dictionary with the given name
        """

        def hook(module, input, output):
            self.intermediates[name] = output

        return hook


class QuadraticNetCholesky(BaselineMLP):
    def __init__(self, input_size, device="cuda"):
        super(QuadraticNetCholesky, self).__init__(
            input_size, sum(range(input_size + 1)), device="cuda"
        )

    def forward(self, x):
        output = self.connection_1(x)
        output = self.activation_1(output)
        output = self.connection_2(output)
        output = self.activation_2(output)
        output = self.connection_3(output)
        output = self.activation_3(output)
        M = create_cholesky(output, self.output_size, self.device)
        # * create symmetric matrix A out of predicted
        # * Cholesky decomposition
        return M.bmm(M.transpose(1, 2))


class CustomCholeskyLoss(nn.Module):
    def __init__(self, diag_L2_loss=0.0, diag_nuclear_loss=0.0):
        super().__init__()
        self.diag_L2_loss = diag_L2_loss
        self.diag_nuclear_loss = diag_nuclear_loss

    def forward(self, M, x, y):
        # M is the symmetric matrix created using the predicted
        # lower-triangular Cholesky decomposition
        # x is the input data, batch_size x n
        # y is the target data, batch_size x 1
        quadratic_form = x.unsqueeze(2).transpose(1, 2).bmm(M).bmm(x.unsqueeze(2))
        loss = F.mse_loss(quadratic_form.squeeze(2), y, reduction="mean")

        # come back to these loss variations
        # if self.diag_L2_loss > 0:
        #     diags = torch.diagonal(M, dim1=-2, dim2=-1)
        #     diagonal_loss = -(
        #         self.diag_L2_loss * F.mse_loss(diags, torch.zeros_like(diags))
        #     )
        #     loss += diagonal_loss
        # if self.diag_nuclear_loss > 0:
        #     nuclear = torch.linalg.matrix_norm(M, ord="nuc", dim=(-2, -1)).mean()

        return loss


class CholeskyPlusConst(nn.Module):
    def __init__(self, input_size, device="cuda"):
        # additional 1 to output_size for +c
        super(CholeskyPlusConst, self).__init__(
            input_size, sum(range(input_size + 1)) + 1, device="cuda"
        )

    def forward(self, x):
        output = self.connection_1(x)
        output = self.activation_1(output)
        output = self.connection_2(output)
        output = self.activation_2(output)
        output = self.connection_3(output)
        output = self.activation_3(output)
        M = create_cholesky(output[:, :-1], self.output_size - 1, self.device)
        c = output[:, -1]
        # * create symmetric matrix A out of predicted
        # * Cholesky decomposition
        return M.bmm(M.transpose(1, 2)), c


class CustomCholeskyPlusConstLoss(nn.Module):
    def __init__(self, const_penalty=0.0):
        super().__init__()
        self.const_penalty = const_penalty

    def forward(self, M, c, x, y):
        # M is the symmetric matrix created using the predicted
        # lower-triangular Cholesky decomposition
        # x is the input data, batch_size x n
        # y is the target data, batch_size x 1
        quadratic_form = x.unsqueeze(2).transpose(1, 2).bmm(M).bmm(
            x.unsqueeze(2)
        ).squeeze(2) + c.unsqueeze(1)
        loss = F.mse_loss(quadratic_form, y, reduction="mean")
        const_loss = self.const_penalty * torch.mean(c)
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
