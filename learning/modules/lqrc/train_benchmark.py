import os
import sys
import time
import torch
import random
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from custom_nn import BaselineMLP, LQRCDataset, QuadraticNetCholesky, CustomCholeskyLoss
from learning import LEGGED_GYM_LQRC_DIR
from plotting import plot_loss, plot_pointwise_predictions

DEVICE = "cuda"


def generate_1D_quadratic(lb, ub, rand_lb=0.0, rand_ub=10.0, steps=100, noise=False):
    """
    Generate 1D quadratic of the form ax**2 + bx + c
    """
    a = random.uniform(rand_lb, rand_ub)
    b = 0  # random.uniform(rand_lb, rand_ub)
    c = random.uniform(rand_lb, rand_ub)
    print("a", a, "b", b, "c", c)

    def baseline_func(x, noise=noise):
        if noise:
            return a * x**2 + b * x + c + random.gauss() * 0.1
        return a * x**2 + b * x + c

    X = torch.linspace(lb, ub, steps).to(DEVICE).reshape(-1, 1)
    y = (
        torch.tensor([baseline_func(X[i]) for i in range(X.shape[0])])
        .to(DEVICE)
        .reshape(-1, 1)
    )
    return baseline_func, X, y


def generate_2D_quadratic():
    """
    Generate 2D quadratic
    """
    pass


def model_switch():
    model_name = sys.argv[1]
    if model_name == "QuadraticNetCholesky":
        return QuadraticNetCholesky(1, X.shape[-1], device=DEVICE).to(
            DEVICE
        ), CustomCholeskyLoss()
    else:
        return BaselineMLP(1, X.shape[-1], device=DEVICE).to(DEVICE), torch.nn.MSELoss(
            reduction="mean"
        )


if __name__ == "__main__":
    save_model = False
    num_epochs = 5000

    baseline_func, X, y = generate_1D_quadratic(-10.0, 10.0, steps=1000)
    data = LQRCDataset(X, y)
    training_data, testing_data = random_split(
        data, data.get_train_test_split_len(0.6, 0.4)
    )
    train_dataloader = DataLoader(training_data, batch_size=100, shuffle=True)
    test_dataloader = DataLoader(testing_data, batch_size=100, shuffle=True)

    model, loss_fn = model_switch()
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # train
    model.train()
    training_losses = []
    for epoch in range(num_epochs):
        loss_per_batch = []
        for X_batch, y_batch in train_dataloader:
            X_batch.requires_grad_()
            if isinstance(model, QuadraticNetCholesky):
                A_pred = model(
                    X_batch
                )  # A is the symmetric matrix from the predicted Cholesky decomposition
                loss = loss_fn(A_pred, X_batch, y_batch)
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
    model.eval()
    all_inputs = []
    all_predictions = []
    all_targets = []
    # with torch.no_grad():
    loss_per_batch = []
    for X_batch, y_batch in test_dataloader:
        if isinstance(model, QuadraticNetCholesky):
            A_pred = model(X_batch)
            loss = loss_fn(A_pred, X_batch, y_batch)
            # turning symmetric matrix into quadratic form for graphing
            y_pred = (
                X_batch.unsqueeze(2)
                .transpose(1, 2)
                .bmm(A_pred)
                .bmm(X_batch.unsqueeze(2))
            ).squeeze(2)
        else:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
        loss_per_batch.append(loss.item())
        all_inputs.append(X_batch)
        all_predictions.append(y_pred)
        all_targets.append(y_batch)
    print("Loss on test set", torch.mean(torch.tensor(loss_per_batch)).item)

    time_str = time.strftime("%Y%m%d_%H%M%S")
    save_str = "1D_quadratic"
    save_path = os.path.join(LEGGED_GYM_LQRC_DIR, "logs", save_str)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if save_model:
        torch.save(
            model.state_dict(),
            save_path + f"/model_{time_str}" + ".pt",
        )
    plot_pointwise_predictions(
        torch.vstack(all_inputs),
        torch.vstack(all_predictions),
        torch.vstack(all_targets),
        f"{save_path}/{time_str}_graph",
    )
    plot_loss(training_losses, f"{save_path}/{time_str}_loss")
