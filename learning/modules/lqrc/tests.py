import torch
from train_benchmark import generate_nD_quadratic

DEVICE = "cuda"


def check_nD_matching(n):
    A = torch.zeros(n, n, device=DEVICE)
    vals = torch.rand(int(n * (n + 1) / 2), device=DEVICE) * 10.0
    i, j = torch.triu_indices(n, n)
    A[i, j] = vals
    A.T[i, j] = vals
    out_X, out_y = generate_nD_quadratic(n, -5, 5, 25, A=A)
    for i in range(out_X.shape[0]):
        expected = out_X[i].unsqueeze(1).T @ A @ out_X[i].unsqueeze(1)
        assert abs(out_y[i].item() - expected.item()) < 1e-3


def test_nD_quadratic_generator_for_1d(n=1):
    check_nD_matching(n)


def test_nD_quadratic_generator_for_2d(n=2):
    check_nD_matching(n)


def test_nD_quadratic_generator_for_3d(n=3):
    check_nD_matching(n)


def test_nD_quadratic_generator_for_4d(n=4):
    check_nD_matching(n)


def test_nD_quadratic_generator_for_5d(n=5):
    check_nD_matching(n)
