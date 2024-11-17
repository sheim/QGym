import pickle
import matplotlib
import fire
import tqdm
from gym import LEGGED_GYM_ROOT_DIR

from learning.modules.ampc.simple_unicycle.casadi.utils import (
    plot_costs_histogram,
    plot_3d_costs,
)
from learning.modules.lqrc.QRCritics import (
    create_PD_lower_diagonal,
    compose_cholesky,
    quadratify_xAx,
    gradient_xAx,
)

import torch

# from torch. import Dataset, DataLoader, random_split
# import torch.nn as nn
# import torch.optim as optim
import numpy as np
import random


def set_deterministic():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(42)
    np.random.seed(42)


class AmpcValueDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, v, dv = self.data[idx]
        return x, v, dv


class NestedAmpcValueDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Each item is an inner list; shuffle this inner list before returning
        inner_list = self.data[idx]
        random.shuffle(inner_list)
        x_values, v_values = zip(*inner_list)
        return torch.stack(x_values), torch.stack(v_values)


class Diagonal(torch.nn.Module):
    def __init__(
        self,
        input_dim,
    ):
        super(Diagonal, self).__init__()
        self.input_dim = input_dim
        self.NN = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, input_dim),
        )

        self.cone_center_offset_NN = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, input_dim),
        )

    def forward(self, x):
        z = self.NN(x)
        A = torch.vmap(torch.diag)(torch.atleast_2d(z)).squeeze()
        x_offsets = self.cone_center_offset_NN(x)
        value = quadratify_xAx(x - x_offsets, A)

        return value, A, x_offsets


class PDCholesky(torch.nn.Module):
    def __init__(
        self,
        input_dim,
    ):
        super(PDCholesky, self).__init__()
        self.input_dim = input_dim
        num_lower_diag_elements = sum(range(input_dim + 1))
        self.lower_diag_NN = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, num_lower_diag_elements),
        )

        self.cone_center_offset_NN = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, input_dim),
        )

    def forward(self, x):
        output = self.lower_diag_NN(x)
        L = create_PD_lower_diagonal(output, self.input_dim, "cpu")
        A = compose_cholesky(L)
        x_off = self.cone_center_offset_NN(x)
        value = quadratify_xAx((x - x_off), A)
        return value, A, x_off


# cholesky latent
# spectral latent


def scaling_function(d, dmax):
    scaling = torch.ones_like(d)  # Default to 1.0
    mask = d >= dmax
    scaling[mask] = 0
    mask = (d > 0) & (d < dmax)
    scaling[mask] = 0.5 * (1 + torch.cos(np.pi * d[mask] / dmax))
    return scaling


class LocalShapeLoss(torch.nn.Module):
    def __init__(
        self,
        V_loss=torch.nn.L1Loss(reduction="none"),
        dV_loss=torch.nn.L1Loss(reduction="none"),
        dmax=0.2,
        dV_scaling=0.5,
    ):
        super(LocalShapeLoss, self).__init__()
        self.dmax = dmax
        # self.l1_loss = torch.nn.MSELoss(reduction='none')
        self.V_loss = V_loss
        self.dV_loss = dV_loss
        self.dV_scaling = dV_scaling

        self.offset_loss = torch.nn.L1Loss(reduction="none")
        self.x_setpoint = torch.zeros(4)
        self.x_threshold = 0.2

    def forward(self, x_batch, psd_matrices, x_offsets, v_batch, dV_batch):
        batch_size, feature_dim = x_batch.size()

        x_batch_exp = x_batch.unsqueeze(1).expand(
            batch_size, batch_size, feature_dim
        )  # Shape: [batch_size, batch_size, feature_dim]
        x_offsets_exp = x_offsets.unsqueeze(1).expand(
            batch_size, batch_size, feature_dim
        )  # Shape: [batch_size, batch_size, feature_dim]

        diff_i = x_batch_exp - x_batch_exp.transpose(
            0, 1
        )  # Shape: [batch_size, batch_size, feature_dim]
        scaling = scaling_function(
            torch.norm(diff_i, dim=-1), self.dmax
        )  # Shape: [batch_size, batch_size]

        diff_j = (
            x_batch_exp.transpose(0, 1) - x_offsets_exp
        )  # Shape: [batch_size, batch_size, feature_dim]

        pred_values = torch.vmap(quadratify_xAx)(
            diff_j, psd_matrices
        )  # Shape: [batch_size, batch_size]
        pred_values_grad = torch.vmap(gradient_xAx)(
            diff_j, psd_matrices
        )  # Shape: [batch_size, batch_size]

        V_losses = self.V_loss(
            pred_values, v_batch.unsqueeze(0).expand(batch_size, batch_size)
        )  # Shape: [batch_size, batch_size]
        dV_losses = self.dV_loss(
            pred_values_grad,
            dV_batch.unsqueeze(0).expand(batch_size, batch_size, feature_dim),
        ).mean(dim=-1)  # Shape: [batch_size, batch_size, feature_dim]
        scaled_l1_losses = scaling * (
            V_losses + self.dV_scaling * dV_losses
        )  # Element-wise scaling

        nonzero_counts = torch.count_nonzero(scaling, dim=1).clamp(
            min=1
        )  # Shape: [batch_size, 1]
        normalized_losses = (
            scaled_l1_losses.sum(dim=1) / nonzero_counts.squeeze()
        )  # Shape: [batch_size]

        # extra loss penalizing x0offset when close to origin:
        offset_losses = self.offset_loss(x_offsets, self.x_setpoint)
        # Compute the distance between x_batch and x_setpoint
        distances = torch.norm(
            x_batch - self.x_setpoint, dim=1, keepdim=True
        )  # Shape: [batch_size, 1]
        # Compute the scaling factor
        # scaling_factors = torch.clamp(1 - distances / self.x_threshold, min=0.0)  # Shape: [batch_size, 1]
        scaling_factors = scaling_function(
            distances, self.x_threshold
        )  # Shape: [batch_size]
        # Apply scaling to the L1 losses
        scaled_losses = (
            scaling_factors * offset_losses
        )  # Broadcasting applies scaling factor per element

        loss = (
            normalized_losses.mean() + 10 * scaled_losses.mean()
        )  # Average over the batch size
        return loss


def process_data(filename, load_norm=False, plt_show=False):
    # Load data from pickle file
    with open(f"{LEGGED_GYM_ROOT_DIR}/data/{filename}.pkl", "rb") as file:
        data = pickle.load(file)
        # ! hot fix for key mismatch
        data["cost_gradient"] = data["gradients"]
        del data["gradients"]

    X, V, dVdx = data["x0"], data["cost"], data["cost_gradient"]

    # Filter and prepare data for plotting
    if type(X[0]) is not list:
        X_plot = [x[0, :] for x in X if abs(x[0, 3]) <= 0.2]
        V_plot = [v[0] for x, v in zip(X, V) if abs(x[0, 3]) <= 0.2]
        dVdx_plot = [dv[0, :] for x, dv in zip(X, dVdx) if abs(x[0, 3]) <= 0.2]

        early_stopping = 0.8
        X_earlystop = [x[: int(early_stopping * x.shape[0]), :] for x in X]
        V_earlystop = [v[: int(early_stopping * v.shape[0])] for v in V]
        dVdx_earlystop = [dv[: int(early_stopping * dv.shape[0]), :] for dv in dVdx]

        X = np.concatenate(X_earlystop, axis=0)
        V = np.concatenate(V_earlystop, axis=0)
        dVdx = np.concatenate(dVdx_earlystop, axis=0)
    else:
        X_plot, V_plot, dVdx_plot = X, V, dVdx

    # Histogram plotting
    plot_costs_histogram(
        np.array(V),
        plt_show=plt_show,
        plt_name=f"{LEGGED_GYM_ROOT_DIR}/plots/{filename}_cost_histogram.png",
    )
    plot_costs_histogram(
        np.array(dVdx),
        plt_show=plt_show,
        plt_name=f"{LEGGED_GYM_ROOT_DIR}/plots/{filename}_cost_gradient_histogram.png",
    )

    # Filter high costs
    def remove_high_cost(X, V, dVdx, threshold=20):
        filtered_X = [x for x, v in zip(X, V) if v <= threshold]
        filtered_V = [v for v in V if v <= threshold]
        filtered_dVdx = [dv for dv, v in zip(dVdx, V) if v <= threshold]
        return filtered_X, filtered_V, filtered_dVdx

    X_clip, V_clip, dVdx_clip = remove_high_cost(X, V, dVdx)
    X_clip_plot, V_clip_plot, dVdx_clip_plot = remove_high_cost(
        X_plot, V_plot, dVdx_plot
    )

    plot_costs_histogram(
        np.array(V_clip),
        plt_show=plt_show,
        plt_name=f"{LEGGED_GYM_ROOT_DIR}/plots/{filename}_cost_clip_histogram.png",
    )
    plot_costs_histogram(
        np.array(dVdx_clip),
        plt_show=plt_show,
        plt_name=f"{LEGGED_GYM_ROOT_DIR}/plots/{filename}_cost_gradient_clip_histogram.png",
    )

    # Normalization
    if load_norm:
        with open(
            f"{LEGGED_GYM_ROOT_DIR}/models/{filename}_normalization.pkl", "rb"
        ) as f:
            normalization_data = pickle.load(f)
            X_min, X_max = normalization_data["X_min"], normalization_data["X_max"]
            V_min, V_max = normalization_data["V_min"], normalization_data["V_max"]
    else:
        X_min = np.array(X_clip).min(axis=0)
        X_max = np.array(X_clip).max(axis=0)
        V_min = np.array(V_clip).min()
        V_max = np.array(V_clip).max()

        # Save normalization values if not provided
        with open(
            f"{LEGGED_GYM_ROOT_DIR}/models/{filename}_normalization.pkl", "wb"
        ) as f:
            pickle.dump(
                {"X_min": X_min, "X_max": X_max, "V_min": V_min, "V_max": V_max}, f
            )

    # Min-Max Scaling for X and V
    X_normalized = 2 * (np.array(X_clip) - X_min) / (X_max - X_min) - 1
    X_normalized_plot = 2 * (np.array(X_clip_plot) - X_min) / (X_max - X_min) - 1
    V_scaled = (np.array(V_clip) - V_min) / (V_max - V_min)
    V_scaled_plot = (np.array(V_clip_plot) - V_min) / (V_max - V_min)

    # Scale dVdx
    dVdx_scaled = (X_max - X_min) / 2 * dVdx_clip / (V_max - V_min)
    dVdx_scaled_plot = (X_max - X_min) / 2 * dVdx_clip_plot / (V_max - V_min)
    plot_costs_histogram(
        np.array(dVdx_scaled),
        plt_show=plt_show,
        plt_name=f"{LEGGED_GYM_ROOT_DIR}/plots/{filename}_cost_gradient_scaled_histogram.png",
    )

    print(f"Normalization X: min={X_min}, max={X_max}")
    print(f"Normalization V: min={V_min:.3f}, max={V_max:.3f}")

    return (
        X_normalized,
        V_scaled,
        dVdx_scaled,
        X_min,
        X_max,
        V_min,
        V_max,
        X_normalized_plot,
        V_scaled_plot,
        dVdx_scaled_plot,
    )


def train(model, filename, plt_show=False):
    (
        X_normalized,
        V_scaled,
        dVdx_scaled,
        X_min,
        X_max,
        V_min,
        V_max,
        X_normalized_plot,
        V_scaled_plot,
        dVdx_scaled_plot,
    ) = process_data(filename, plt_show=plt_show)

    # Prepare data for training and validation
    train_ratio = 0.8
    X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
    V_tensor = torch.tensor(V_scaled, dtype=torch.float32)
    dVdx_tensor = torch.tensor(dVdx_scaled, dtype=torch.float32)

    dataset_size = len(X_tensor)
    train_size = int(dataset_size * train_ratio)
    val_size = dataset_size - train_size

    train_data, val_data = torch.utils.data.random_split(
        list(zip(X_tensor, V_tensor, dVdx_tensor)), [train_size, val_size]
    )

    # Create DataLoaders
    batch_size = 1024
    train_loader = torch.utils.data.DataLoader(
        AmpcValueDataset(train_data), batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        AmpcValueDataset(val_data), batch_size=batch_size, shuffle=False
    )

    # model = PDCholesky(input_dim=len(X_min))
    # model = Diagonal(input_dim=len(X_min))
    dmax = 0.2
    dV_scaling = 0.2
    criterion = LocalShapeLoss(
        torch.nn.L1Loss(reduction="none"),
        torch.nn.L1Loss(reduction="none"),
        dmax=dmax,
        dV_scaling=dV_scaling,
    )
    # criterion = SobolLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, verbose=True
    )

    epochs = 2  # 250
    for epoch in tqdm.tqdm(range(epochs)):
        model.train()
        train_loss = 0.0
        for x_batch, v_batch, dV_batch in train_loader:
            optimizer.zero_grad()

            # if type(model) in (PsdChol, PsdCholLatentLin):
            #     predictions, _ = model(x_batch)
            #     loss = criterion(predictions.squeeze(), v_batch)
            if type(model) is PDCholesky or type(model) is Diagonal:
                predictions, psd_matrices, x_offset = model(x_batch)
                if type(criterion) is LocalShapeLoss:
                    loss = criterion(x_batch, psd_matrices, x_offset, v_batch, dV_batch)
                else:
                    loss = criterion(predictions.squeeze(), v_batch) + 10 * criterion(
                        x_offset.squeeze(), torch.zeros_like(x_offset.squeeze())
                    )

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, v_batch, dV_batch in val_loader:
                # if type(model) in (PsdChol, PsdCholLatentLin):
                #     predictions, psd_matrices = model(x_batch)
                #     loss = criterion(predictions.squeeze(), v_batch)
                if type(model) is PDCholesky:
                    predictions, psd_matrices, x_offsets = model(x_batch)
                    if type(criterion) is LocalShapeLoss:
                        loss = criterion(
                            x_batch, psd_matrices, x_offsets, v_batch, dV_batch
                        )
                    else:
                        loss = criterion(
                            predictions.squeeze(), v_batch
                        ) + 10 * criterion(
                            x_offset.squeeze(), torch.zeros_like(x_offset.squeeze())
                        )

                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}"
        )

    torch.save(
        model.state_dict(),
        f"{LEGGED_GYM_ROOT_DIR}/models/{filename}_{type(model).__name__}.pth",
    )
    with open(
        f"{LEGGED_GYM_ROOT_DIR}/models/{filename}_{type(model).__name__}.pkl", "wb"
    ) as f:
        pickle.dump({"V_max": V_max, "V_min": V_min, "X_max": X_max, "X_min": X_min}, f)

    # model.eval()
    # with torch.no_grad():
    #     predictions, psd_matrices, x_offsets = model(torch.tensor(X_normalized_plot,dtype=torch.float32))

    # if type(model) is PDCholesky:
    #     for xy, v_true, v_predict, P_predict, x_off in zip(X_normalized_plot, V_scaled_plot, predictions.detach().cpu().numpy(), psd_matrices.detach().cpu().numpy(), x_offsets.detach().cpu().numpy()):
    #         plot_3d_costs(X_normalized_plot, V_scaled_plot, xy, v_true, v_predict, P_predict, x_off, zlim=1,dmax=dmax)


def eval_model(model, filename):
    (
        _,
        _,
        _,
        X_min,
        X_max,
        V_min,
        V_max,
        X_normalized_plot,
        V_scaled_plot,
        dVdx_scaled_plot,
    ) = process_data(filename, load_norm=True)

    # model = PDCholesky(
    #     np.shape(X_min)[0]
    # )  # Make sure to define or initialize your model appropriately
    model.load_state_dict(
        torch.load(
            f"{LEGGED_GYM_ROOT_DIR}/models/{filename}_{type(model).__name__}.pth"
        )
    )
    model.eval()

    with torch.no_grad():
        # Model predictions
        predictions, psd_matrices, x_offsets = model(
            torch.tensor(X_normalized_plot, dtype=torch.float32)
        )

    # Convert to NumPy for plotting
    predictions = predictions.detach().cpu().numpy()
    psd_matrices = psd_matrices.detach().cpu().numpy()
    x_offsets = x_offsets.detach().cpu().numpy()

    # Only take every 10th entry for plotting
    for xy, v_true, v_predict, P_predict, x_off in zip(
        X_normalized_plot[::10],
        V_scaled_plot[::10],
        predictions[::10],
        psd_matrices[::10],
        x_offsets[::10],
    ):
        plot_3d_costs(
            X_normalized_plot,
            V_scaled_plot,
            xy,
            v_true,
            v_predict,
            P_predict,
            x_off,
            zlim=1,
            dmax=0.3,
        )


if __name__ == "__main__":
    # fire.Fire({
    #     "train": train
    # })
    filename = "4d_data_9261"
    # model = PDCholesky(input_dim=len(X_min))
    # model = Diagonal(input_dim=len(X_min))
    model_names = [
        "Diagonal",
        # "OuterProduct",
        # # "OuterProductLatent",
        "PDCholesky",
        # "CholeskyInput",
        # "CholeskyLatent",
        # "DenseSpectralLatent",
    ]

    input_dim = 4

    for model in model_names:
        model = eval(f"{model}(input_dim=input_dim)")
        train(model, filename=filename)
    # for model in model_names:
    #     model = eval(f"{model}(input_dim=input_dim)")
    #     eval_model(model, filename=filename)
