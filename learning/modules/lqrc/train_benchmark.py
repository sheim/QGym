import os
import sys
import time
import torch
import random
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from custom_nn import BaselineMLP, LQRCDataset, QuadraticNetCholesky, CustomCholeskyLoss
from learning import LEGGED_GYM_LQRC_DIR
from plotting import (
    plot_loss,
    plot_predictions_and_gradients,
)

DEVICE = "cuda"


def generate_nD_quadratic(n, lb, ub, steps, rand_scaling=100.0, A=None, noise=False):
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
    y = (
        X.transpose(1, 2).bmm(batch_A).bmm(X)
        + (random.gauss(mu=0, sigma=1) * rand_scaling / 2.0)
        if noise
        else X.transpose(1, 2).bmm(batch_A).bmm(X)
    )
    return X.squeeze(2), y.squeeze(2)


def model_switch(input_dim):
    model_name = sys.argv[1]
    if model_name == "QuadraticNetCholesky":
        return QuadraticNetCholesky(input_dim, X.shape[-1], device=DEVICE).to(
            DEVICE
        ), CustomCholeskyLoss()
    else:
        return BaselineMLP(input_dim, X.shape[-1], device=DEVICE).to(
            DEVICE
        ), torch.nn.MSELoss(reduction="mean")


if __name__ == "__main__":
    save_model = False
    num_epochs = 2000
    input_dim = 1

    X, y = generate_nD_quadratic(input_dim, -10.0, 10.0, steps=100, noise=False)
    data = LQRCDataset(X, y)
    training_data, testing_data = random_split(
        data, data.get_train_test_split_len(0.6, 0.4)
    )
    train_dataloader = DataLoader(training_data, batch_size=100, shuffle=True)
    test_dataloader = DataLoader(testing_data, batch_size=1, shuffle=True)

    model, loss_fn = model_switch(input_dim)
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
    # model.eval()  # removed to enable access to the gradients for graphing
    all_inputs = []
    all_predictions = []
    all_targets = []
    all_gradients = []
    # with torch.no_grad():
    loss_per_batch = []
    for X_batch, y_batch in test_dataloader:
        # give X_batch a grad_fn
        X_batch.requires_grad_()
        X_batch = X_batch + 0
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
    save_str = "1D_quadratic"
    save_path = os.path.join(LEGGED_GYM_LQRC_DIR, "logs", save_str)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if save_model:
        torch.save(
            model.state_dict(),
            save_path + f"/model_{time_str}" + ".pt",
        )

    plot_predictions_and_gradients(
        input_dim + 1,
        torch.vstack(all_inputs),
        torch.vstack(all_predictions),
        torch.vstack(all_targets),
        all_gradients,
        f"{save_path}/{time_str}_grad_graph",
    )

    plot_loss(training_losses, f"{save_path}/{time_str}_loss")
