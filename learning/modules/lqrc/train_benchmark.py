import os
import time
import torch
from torchviz import make_dot
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from custom_nn import (
    BaselineMLP,
    LQRCDataset,
    QuadraticNetCholesky,
    CustomCholeskyLoss,
    CholeskyPlusConst,
    CustomCholeskyPlusConstLoss,
)
from learning import LEGGED_GYM_LQRC_DIR
from utils import benchmark_args
from plotting import (
    plot_loss,
    plot_predictions_and_gradients,
)

DEVICE = "cuda"


def generate_nD_quadratic(n, lb, ub, steps, rand_scaling=10.0, A=None, noise=None):
    """
    Generate nD quadratic in the form of x.T @ A @ x
    """
    if A is None:
        A = torch.zeros(n, n, device=DEVICE)
        vals = torch.rand(int(n * (n + 1) / 2), device=DEVICE) * rand_scaling
        i, j = torch.triu_indices(n, n)
        A[i, j] = vals
        A.T[i, j] = vals

    all_linspaces = [torch.linspace(lb, ub, steps, device=DEVICE) for i in range(n)]
    X = (
        torch.cartesian_prod(*all_linspaces).unsqueeze(2)
        if n > 1
        else torch.cartesian_prod(*all_linspaces).unsqueeze(1).unsqueeze(2)
    )
    batch_A = A.repeat(X.shape[0], 1, 1).to(DEVICE)
    y = X.transpose(1, 2).bmm(batch_A).bmm(X)
    return X.squeeze(2), y.squeeze(2)


def generate_cos(n, lb, ub, steps):
    # make X a n-dimensional lin-space
    X = torch.rand((steps, n), device=DEVICE) * (ub - lb) + lb
    freqs = torch.rand(n, device=DEVICE) * 2.0 + 0.5
    offsets = torch.rand(n, device=DEVICE) * 2.0 * torch.pi
    y = (torch.cos(freqs * X + offsets) + 1.0).sum(axis=1).unsqueeze(1)
    return X, torch.tensor(y, device=DEVICE)


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


def generate_rosenbrock_grad(X):
    g = torch.zeros_like(X)
    term_1 = -400.0 * X[:, 1:-1] * (X[:, 2:] - torch.square(X[:, 1:-1])) - 2.0 * (
        1 - X[:, 1:-1]
    )
    term_2 = 200 * (X[:, 1:-1] - torch.square(X[:, :-2]))
    g[:, 0] = -400.0 * X[:, 0] * (X[:, 1] - torch.square(X[:, 0])) - 2.0 * (1 - X[:, 0])
    g[:, 1:-1] = term_1 + term_2
    g[:, -1] = 200 * (X[:, -1] - torch.square(X[:, -2]))
    return g


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


def model_switch(input_dim, model_name=None):
    if model_name == "QuadraticNetCholesky":
        return QuadraticNetCholesky(input_dim, device=DEVICE).to(
            DEVICE
        ), CustomCholeskyLoss()
    elif model_name == "CholeskyPlusConst":
        return CholeskyPlusConst(input_dim, device=DEVICE).to(
            DEVICE
        ), CustomCholeskyPlusConstLoss(const_penalty=0.1)
    else:
        return BaselineMLP(input_dim, device=DEVICE).to(DEVICE), torch.nn.MSELoss(
            reduction="mean"
        )


def test_case_switch(case_name=None):
    if case_name == "rosenbrock":
        return generate_bounded_rosenbrock
    elif case_name == "cos":
        return generate_cos
    elif case_name == "quadratic":
        return generate_nD_quadratic
    else:
        assert case_name is not None, "Please specify a valid test case when running."


if __name__ == "__main__":
    args = vars(benchmark_args())
    save_model = args["save_model"]
    num_epochs = args["epochs"]
    input_dim = args["input_dim"]
    test_case = args["test_case"]
    model_type = args["model_type"]
    save_str = f"{input_dim}D_{test_case}" + model_type

    generate_data = test_case_switch(test_case)
    X, y = generate_data(input_dim, -3.0, 3.0, steps=500)
    data = LQRCDataset(X, y)
    if args["split_chunk"]:
        training_data = torch.utils.data.Subset(
            data, range(data.get_train_test_split_len(0.6, 0.4)[0])
        )
        testing_data = torch.utils.data.Subset(
            data, range(data.get_train_test_split_len(0.6, 0.4)[1])
        )
    else:
        training_data, testing_data = random_split(
            data, data.get_train_test_split_len(0.64, 0.36)
        )
    train_dataloader = DataLoader(training_data, batch_size=100, shuffle=True)
    test_dataloader = DataLoader(
        testing_data,
        batch_size=1,
        shuffle=False if args["colormap_diff"] or args["colormap_values"] else True,
    )

    model, loss_fn = model_switch(input_dim, model_type)
    print(model)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=9.869448957882323e-06,
        betas=(0.75, 0.9),
        weight_decay=2.074821225483474e-07,
    )

    # train
    model.train()
    training_losses = []
    for epoch in range(num_epochs):
        loss_per_batch = []
        for X_batch, y_batch in train_dataloader:
            X_batch.requires_grad_()
            if model_type == "QuadraticNetCholesky":
                A_pred = model(X_batch)
                loss = loss_fn(A_pred, X_batch, y_batch)
            elif model_type == "CholeskyPlusConst":
                A_pred, const_pred = model(X_batch)
                loss = loss_fn(A_pred, const_pred, X_batch, y_batch)
            else:
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
            loss_per_batch.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            # print(X_batch.grad)
            optimizer.step()
        training_losses.append(torch.mean(torch.tensor(loss_per_batch)).item())
        if epoch % 250 == 0:
            print(f"Finished epoch {epoch}, latest loss {training_losses[-1]}")
    print(f"Finished training at epoch {epoch}, latest loss {training_losses[-1]}")

    # test
    # model.eval()  # removed to enable access to the gradients for graphing
    all_inputs = []
    all_predictions = []
    all_targets = []
    all_gradients = []
    # with torch.no_grad():
    loss_per_batch = []
    # * redefine the test_dataloader to contain both test and train to suit
    # * matplotlib colormap requirements
    if args["test_case"] == "rosenbrock":
        graphing_data = LQRCDataset(*generate_data(input_dim, -3.0, 3.0, steps=100))
        test_dataloader = DataLoader(
            torch.utils.data.Subset(graphing_data, range(len(graphing_data))),
            batch_size=1,
            shuffle=False,
        )

    for X_batch, y_batch in test_dataloader:
        # give X_batch a grad_fn
        X_batch.requires_grad_()
        X_batch = X_batch + 0
        if model_type == "QuadraticNetCholesky":
            A_pred = model(X_batch)
            loss = loss_fn(A_pred, X_batch, y_batch)
            # turning symmetric matrix into quadratic form for graphing
            y_pred = (
                X_batch.unsqueeze(2)
                .transpose(1, 2)
                .bmm(A_pred)
                .bmm(X_batch.unsqueeze(2))
            ).squeeze(2)
        elif model_type == "CholeskyPlusConst":
            A_pred, const_pred = model(X_batch)
            loss = loss_fn(A_pred, const_pred, X_batch, y_batch)
            y_pred = (
                X_batch.unsqueeze(2)
                .transpose(1, 2)
                .bmm(A_pred)
                .bmm(X_batch.unsqueeze(2))
            ).squeeze(2) + const_pred
        else:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

        if model_type == "QuadraticNetCholesky":
            all_gradients.append(
                (
                    X_batch.view(-1),
                    y_pred.view(-1),
                    (2.0 * A_pred.bmm(X_batch.unsqueeze(2))).view(-1),
                )
            )
        else:
            all_gradients.append(
                (
                    X_batch.view(-1),
                    y_pred.view(-1),
                    torch.autograd.grad(y_pred, X_batch)[0].view(-1),
                )
            )
        loss_per_batch.append(loss.item())
        all_inputs.append(X_batch)
        all_predictions.append(y_pred)
        all_targets.append(y_batch)
    print("Loss on test set", torch.mean(torch.tensor(loss_per_batch)).item())

    time_str = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(LEGGED_GYM_LQRC_DIR, "logs", save_str)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if save_model:
        torch.save(
            model.state_dict(),
            save_path + f"/model_{time_str}" + ".pt",
        )
        make_dot(
            y_pred,
            params=dict(model.named_parameters()),
            show_attrs=True,
            show_saved=True,
        ).render(f"{LEGGED_GYM_LQRC_DIR}/logs/model_viz", format="png")
        print("Saving to", save_path)

    plot_predictions_and_gradients(
        input_dim + 1,
        torch.vstack(all_inputs),
        torch.vstack(all_predictions),
        torch.vstack(all_targets),
        all_gradients,
        f"{save_path}/{time_str}_grad_graph",
        colormap_diff=args["colormap_diff"],
        colormap_values=args["colormap_values"],
        actual_grad=generate_rosenbrock_grad(torch.vstack(all_inputs))
        if test_case == "rosenbrock"
        else None,
    )

    plot_loss(training_losses, f"{save_path}/{time_str}_loss")
