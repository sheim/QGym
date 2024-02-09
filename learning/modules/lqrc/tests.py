import torch
from train_benchmark import generate_nD_quadratic, generate_rosenbrock_grad

DEVICE = "cuda"


def make_rand_symmetric_matrix(n):
    A = torch.zeros(n, n, device=DEVICE)
    vals = torch.rand(int(n * (n + 1) / 2), device=DEVICE) * 10.0
    i, j = torch.triu_indices(n, n)
    A[i, j] = vals
    A.T[i, j] = vals
    return A


def check_nD_matching(n):
    A = make_rand_symmetric_matrix(n)
    out_X, out_y = generate_nD_quadratic(n, -5, 5, 25, A=A)
    for i in range(out_X.shape[0]):
        expected = out_X[i].unsqueeze(1).T @ A @ out_X[i].unsqueeze(1)
        assert abs(out_y[i].item() - expected.item()) < 1e-3


def test_nD_quadratic_generator_for_1D(n=1):
    check_nD_matching(n)


def test_nD_quadratic_generator_for_2D(n=2):
    check_nD_matching(n)


def test_nD_quadratic_generator_for_3D(n=3):
    check_nD_matching(n)


def test_nD_quadratic_generator_for_4D(n=4):
    check_nD_matching(n)


def test_nD_quadratic_generator_for_5D(n=5):
    check_nD_matching(n)


def test_rosenbrock_grad_2D():
    X = torch.tensor([[3, 7]]).repeat(5, 1)
    expected_grad = torch.tensor([[2404, -400]]).repeat(5, 1)
    result_grad = generate_rosenbrock_grad(X)
    assert torch.equal(expected_grad, result_grad)


def test_rosenbrock_grad_3D():
    X = torch.tensor([[3, 4, 5]]).repeat(5, 1)
    expected_grad = torch.tensor([[6004, 16606, -2200]]).repeat(5, 1)
    result_grad = generate_rosenbrock_grad(X)
    assert torch.equal(expected_grad, result_grad)
