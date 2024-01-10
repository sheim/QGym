import time
import torch
import random
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from custom_nn import LQRCDataset, QuadraticNetCholesky, CustomCholeskyLoss
from plotting import plot_pointwise_predictions


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

    X = torch.linspace(lb, ub, steps).to("cuda").reshape(-1, 1)
    y = (
        torch.tensor([baseline_func(X[i]) for i in range(X.shape[0])])
        .to("cuda")
        .reshape(-1, 1)
    )
    return baseline_func, X, y


def generate_2D_quadratic():
    """
    Generate 2D quadratic
    """
    pass


if __name__ == "__main__":
    save_model = False
    baseline_func, X, y = generate_1D_quadratic(-10.0, 10.0, steps=1000)
    data = LQRCDataset(X, y)
    training_data, testing_data = random_split(
        data, data.get_train_test_split_len(0.6, 0.4)
    )
    train_dataloader = DataLoader(training_data, batch_size=100, shuffle=True)
    test_dataloader = DataLoader(testing_data, batch_size=100, shuffle=True)

    model = QuadraticNetCholesky(1, X.shape[-1]).to("cuda")
    print(model)
    loss_fn = CustomCholeskyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # train
    model.train()
    losses = []
    for epoch in range(5000):
        loss_per_batch = []
        for X_batch, y_batch in train_dataloader:
            # X_batch.require_grad = True
            A_pred = model(
                X_batch
            )  # where A is the symmetric matrix resulting from the predicted Cholesky decomposition
            loss = loss_fn(A_pred, X_batch, y_batch)
            loss_per_batch.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            # print(X_batch.grad)
            optimizer.step()
        losses.append(torch.mean(torch.tensor(loss_per_batch)))
        if epoch % 50 == 0:
            print(f"Finished epoch {epoch}, latest loss {losses[-1]}")
    print(f"Finished training at epoch {epoch}, latest loss {losses[-1]}")

    # test
    model.eval()
    all_inputs = []
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        loss_per_batch = []
        for X_batch, y_batch in test_dataloader:
            A_pred = model(X_batch)
            loss = loss_fn(A_pred, X_batch, y_batch)
            loss_per_batch.append(loss.item())
            # turning symmetric matrix into quadratic form for graphing
            y_pred = (
                X_batch.unsqueeze(2)
                .transpose(1, 2)
                .bmm(A_pred)
                .bmm(X_batch.unsqueeze(2))
            )
            all_inputs.append(X_batch)
            all_predictions.append(y_pred)
            all_targets.append(y_batch)
        print("Loss on test set", torch.mean(torch.tensor(loss_per_batch)))

    time_str = time.strftime("%Y%m%d_%H%M%S")
    save_str = "1D_quadratic"
    if save_model:
        torch.save(
            model.state_dict(),
            "learning/modules/lqrc/" + save_str + "_" + time_str + ".pt",
        )
    plot_pointwise_predictions(
        torch.vstack(all_inputs),
        torch.vstack(all_predictions).squeeze(2),
        torch.vstack(all_targets),
        f"{save_str}_{time_str}",
    )
