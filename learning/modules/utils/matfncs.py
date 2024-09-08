import torch


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


def least_squares_fit(x, y):
    x_flat = x.view(-1, x.shape[-1])
    y_flat = y.view(-1, y.shape[-1])
    ones = torch.ones(x_flat.shape[0], 1, device=x.device)
    x_aug = torch.cat([ones, x_flat], dim=1)
    # Use torch.linalg.lstsq to find the least-squares solution
    result = torch.linalg.lstsq(x_aug, y_flat)
    coefficients = result.solution
    bias = coefficients[0]
    weights = coefficients[1:]

    return weights, bias


def forward_affine(x, weights, bias):
    x_flat = x.view(-1, x.shape[-1])
    x_aug = torch.cat(
        [torch.ones(x_flat.shape[0], 1, device=x_flat.device), x_flat], dim=1
    )
    return x_aug.matmul(torch.cat([bias.unsqueeze(0), weights], dim=0))
