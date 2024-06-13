import torch


DEVICE = "cuda:0"


def generate_rosenbrock(n, lb, ub, steps):
    """
    Generates data based on Rosenbrock function
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """
    assert n > 1, "n must be > 1 for Rosenbrock"
    all_linspaces = [torch.linspace(lb, ub, steps, device=DEVICE) for i in range(n)]
    X = torch.cartesian_prod(*all_linspaces)
    term_1 = 100 * torch.square(X[:, 1:] - torch.square(X[:, :-1]))
    term_2 = torch.square(1 - X[:, :-1])
    y = torch.sum(term_1 + term_2, axis=1)
    return X, y.unsqueeze(1)


def generate_bounded_rosenbrock(n, lb, ub, steps):
    """
    Generates data based on Rosenbrock function
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """

    def bound_function_smoothly(y):
        a = 50.0  # Threshold
        c = 60.0  # Constant to transition to
        k = 0.1  # Sharpness of transition
        return y * (1 - 1 / (1 + torch.exp(-k * (y - a)))) + c * (
            1 / (1 + torch.exp(-k * (y - a)))
        )

    X, y = generate_rosenbrock(n, lb, ub, steps)
    y = bound_function_smoothly(y)
    return X, y


def generate_rosenbrock_g_data_dict(names, learning_rates):
    graphing_data = {
        lr: {
            data_name: {name: {} for name in names}
            for data_name in [
                "critic_obs",
                "values",
                "returns",
                "error",
            ]
        }
        for lr in learning_rates
    }
    return graphing_data
