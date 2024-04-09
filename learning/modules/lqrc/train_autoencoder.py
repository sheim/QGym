import os
import time
import torch
import torch.nn as nn
from autoencoder import Autoencoder
from gym import LEGGED_GYM_ROOT_DIR
from utils import autoencoder_args
from plotting import plot_loss, plot_autoencoder


def generate_data(lb, ub, n):
    return (ub - lb).expand(n, -1)*torch.rand(n, ub.shape[-1]) + lb


if __name__ == "__main__":
    args = vars(autoencoder_args())
    num_epochs = args["epochs"]
    num_batches = args["batches"]
    input_dim = args["input_dim"]
    latent_dim = args["latent_dim"]
    n = args["n"]
    lb = torch.tensor([-torch.pi, -8.0])
    ub = torch.tensor([torch.pi, 8.0])

    model = Autoencoder(input_dim, latent_dim, [16, 8, 4, 2])

    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    training_loss = []
    for epoch in range(num_epochs):
        batch_loss = []
        for batch in range(num_batches):
            target = generate_data(lb, ub, n)
            prediction = model(target)
            loss = nn.MSELoss(reduction="mean")(target, prediction)
            batch_loss.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        training_loss.append(torch.mean(torch.tensor(batch_loss)).item())
        if epoch % 250 == 0:
            print(f"Finished epoch {epoch}, latest loss {training_loss[-1]}")
    print(f"Training finished with final loss of {training_loss[-1]}")

    test_loss = []
    targets = generate_data(lb, ub, 100)
    predictions = torch.zeros_like(targets)
    for ix, target in enumerate(targets):
            prediction = model(target)
            predictions[ix, :] = prediction
            loss = nn.MSELoss(reduction="mean")(target, prediction)
            test_loss.append(loss)

    print(f"Average loss on test set {torch.mean(torch.tensor(test_loss))}")

    save_str = "autoencoder"
    time_str = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "autoencoders", save_str)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plot_loss(training_loss, f"{save_path}/{save_str}_loss")
    plot_autoencoder(targets, predictions,  f"{save_path}/{save_str}_predictions")

    if args["save_model"]:
        torch.save(
            model.state_dict(),
            save_path + f"/model_{time_str}" + ".pt",
        )
