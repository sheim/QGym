import pickle
import matplotlib
import fire

# from psdnets import PDCholeskyInput
from utils import plot_costs_histogram, plot_3d_costs
from psdnets import create_PD_lower_diagonal, compose_cholesky, quadratify_xAx, gradient_xAx

import torch
# from torch. import Dataset, DataLoader, random_split
# import torch.nn as nn
# import torch.optim as optim
import numpy as np
import random

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
    

class MLP(torch.nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)

class PsdChol(torch.nn.Module):
    def __init__(
        self,
        input_dim,
    ):
        super(PsdChol, self).__init__()
        self.input_dim = input_dim
        num_lower_diag_elements = sum(range(input_dim + 1))
        self.lower_diag_NN = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, num_lower_diag_elements)
        )

    def forward(self, x):
        output = self.lower_diag_NN(x)
        L = create_PD_lower_diagonal(output, self.input_dim, "cpu")
        A = compose_cholesky(L)
        value = quadratify_xAx(x, A)
        return value, A
    
class PsdCholOff(torch.nn.Module):
    def __init__(
        self,
        input_dim,
    ):
        super(PsdCholOff, self).__init__()
        self.input_dim = input_dim
        num_lower_diag_elements = sum(range(input_dim + 1))
        self.lower_diag_NN = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_lower_diag_elements)
        )
        
        self.cone_center_offset_NN = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, input_dim)
        )

    def forward(self, x):
        output = self.lower_diag_NN(x)
        L = create_PD_lower_diagonal(output, self.input_dim, "cpu")
        A = compose_cholesky(L)
        x_off = self.cone_center_offset_NN(x)
        value = quadratify_xAx((x-x_off), A)
        return value, A, x_off


class PsdCholLatentLin(torch.nn.Module):
    def __init__(
        self,
        input_dim,
    ):
        super(PsdCholLatentLin, self).__init__()
        self.latent_dim = input_dim*4
        num_lower_diag_elements = sum(range(self.latent_dim  + 1))
        self.lower_diag_NN = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, num_lower_diag_elements)
        )
        
        self.latent_linear = torch.nn.Linear(input_dim, self.latent_dim)
    
    def get_W_b(self):
        W = self.latent_linear.weight.detach().cpu().numpy()
        b = self.latent_linear.bias.detach().cpu().numpy()
        return W,b
    
    def forward(self, x):
        output = self.lower_diag_NN(x)
        L = create_PD_lower_diagonal(output, self.latent_dim, "cpu")
        A = compose_cholesky(L)
        z = self.latent_linear(x)
        value = quadratify_xAx(z, A)
        return value, A
    
    
def scaling_function(d, dmax):
    scaling = torch.ones_like(d)  # Default to 1.0
    mask = d >= dmax
    scaling[mask]=0
    mask = (d > 0) & (d < dmax)
    scaling[mask] = 0.5 * (1 + torch.cos(np.pi * d[mask] / dmax))
    return scaling

class LocalShapeLoss(torch.nn.Module):
    def __init__(self, V_loss = torch.nn.L1Loss(reduction='none'), dV_loss = torch.nn.L1Loss(reduction='none'), dmax=0.2, dV_scaling=0.5):
        super(LocalShapeLoss, self).__init__()
        self.dmax = dmax
        # self.l1_loss = torch.nn.MSELoss(reduction='none')
        self.V_loss = V_loss
        self.dV_loss = dV_loss
        self.dV_scaling = dV_scaling

    def forward(self, x_batch, psd_matrices, x_offsets, v_batch, dV_batch):
        batch_size,feature_dim = x_batch.size()
        
        x_batch_exp = x_batch.unsqueeze(1).expand(batch_size, batch_size, feature_dim)  # Shape: [batch_size, batch_size, feature_dim]
        x_offsets_exp = x_offsets.unsqueeze(1).expand(batch_size, batch_size, feature_dim)  # Shape: [batch_size, batch_size, feature_dim]
        
        diff_i = x_batch_exp - x_batch_exp.transpose(0, 1)  # Shape: [batch_size, batch_size, feature_dim]
        scaling = scaling_function(torch.norm(diff_i, dim=-1), self.dmax)  # Shape: [batch_size, batch_size]
        
        diff_j = x_batch_exp.transpose(0, 1) - x_offsets_exp  # Shape: [batch_size, batch_size, feature_dim]
    
        pred_values = torch.vmap(quadratify_xAx)(diff_j, psd_matrices)  # Shape: [batch_size, batch_size]
        pred_values_grad = torch.vmap(gradient_xAx)(diff_j, psd_matrices)  # Shape: [batch_size, batch_size]

        V_losses = self.V_loss(pred_values, v_batch.unsqueeze(0).expand(batch_size, batch_size))  # Shape: [batch_size, batch_size]
        dV_losses = self.dV_loss(pred_values_grad, dV_batch.unsqueeze(0).expand(batch_size, batch_size, feature_dim)).mean(dim=-1)  # Shape: [batch_size, batch_size, feature_dim]
        scaled_l1_losses = scaling * (V_losses+self.dV_scaling*dV_losses)  # Element-wise scaling
        
        nonzero_counts = torch.count_nonzero(scaling, dim=1).clamp(min=1)  # Shape: [batch_size, 1]
        normalized_losses = scaled_l1_losses.sum(dim=1) / nonzero_counts.squeeze()  # Shape: [batch_size]
        loss = normalized_losses.mean()  # Average over the batch size
        return loss

class SobolLoss(torch.nn.Module):
    def __init__(self, V_loss=torch.nn.L1Loss(), dV_loss=torch.nn.L1Loss(), scaling=1):
        super(SobolLoss, self).__init__()
        self.V_loss = V_loss
        self.dV_loss = dV_loss
        self.scaling = scaling

    def forward(self, x_batch, psd_matrices, x_offsets, v_batch, dV_batch):
        V_pred = quadratify_xAx(x_batch-x_offsets, psd_matrices)
        dV_pred = gradient_xAx(x_batch-x_offsets, psd_matrices)

        # Calculate Sobol loss as the difference between predicted and true gradients
        sobol_loss = self.V_loss(V_pred, v_batch) + self.scaling * self.dV_loss(dV_pred, dV_batch)
        return sobol_loss

def train(filename, plt_show=False):
    with open(f"data/{filename}.pkl", "rb") as file:
        data = pickle.load(file)
    
    if "nested_x0" in data.keys():
        print("Using nested dataloader!")
        
        # Load nested data
        X_nested = data["nested_x0"]
        V_nested = data["nested_cost"]

        # Flatten costs for histogram and min-max normalization
        V_flattened = [v for inner_list in V_nested for v in inner_list]
        X_flattened = np.array([x for inner_list in X_nested for x in inner_list])

        # Plot histogram of flattened costs
        plot_costs_histogram(np.array(V_flattened), plt_show=plt_show, plt_name=f"plots/{filename}_cost_histogram.png")

        # Compute min-max normalization values from flattened data
        X_min = X_flattened.min(axis=0)
        X_max = X_flattened.max(axis=0)
        V_min = min(V_flattened)
        V_max = max(V_flattened)

        print(f"Min-Max Scaling X: min={X_min}, max={X_max}")
        print(f"Normalization V: min={V_min:.3f}, max={V_max:.3f}")

        # Normalize each nested list using the global min and max values
        X_normalized_nested = [
            [(torch.tensor((x - X_min) / (X_max - X_min), dtype=torch.float32), torch.tensor((v - V_min) / (V_max - V_min), dtype=torch.float32))
            for x, v in zip(inner_X, inner_V)]
            for inner_X, inner_V in zip(X_nested, V_nested)
        ]

        # Train-test split on the outer lists, maintaining inner lists intact
        train_ratio = 0.8
        dataset_size = len(X_normalized_nested)
        train_size = int(dataset_size * train_ratio)
        val_size = dataset_size - train_size

        # Randomly split the dataset
        train_data, val_data = torch.utils.data.random_split(X_normalized_nested, [train_size, val_size])

        def collate_fn(batch):
            x_values = torch.cat([x_values for x_values, _ in batch], dim=0)
            v_values = torch.cat([v_values for _, v_values in batch], dim=0)
            return x_values, v_values

        # Create data loaders where each batch is a randomly chosen inner list
        batch_size   = 4  # Each batch is one inner list
        train_loader = torch.utils.data.DataLoader(NestedAmpcValueDataset(train_data), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader   = torch.utils.data.DataLoader(NestedAmpcValueDataset(val_data), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        
        # for plotting
        X_normalized = (np.array(X_flattened) - X_min) / (X_max - X_min)
        V_scaled = (np.array(V_flattened) - V_min) / (V_max - V_min)

    else:
        X = data["x0"]
        V = data["cost"]
        dVdx = data["cost_gradient"]
        plot_costs_histogram(np.array(V), plt_show=plt_show, plt_name=f"plots/{filename}_cost_histogram.png")
        plot_costs_histogram(np.array(dVdx), plt_show=plt_show, plt_name=f"plots/{filename}_cost_gradient_histogram.png")
        
        def remove_high_cost(X, V, dVdx, threshold=30):
            filtered_X = [x for x, v in zip(X, V) if v <= threshold]
            filtered_V = [v for v in V if v <= threshold]
            filtered_dVdx = [dv for dv, v in zip(dVdx, V) if v <= threshold]
            return filtered_X, filtered_V, filtered_dVdx
        
        X_clip, V_clip, dVdx_clip = remove_high_cost(X, V, dVdx)
        plot_costs_histogram(np.array(V_clip), plt_show=plt_show, plt_name=f"plots/{filename}_cost_clip_histogram.png")
        plot_costs_histogram(np.array(dVdx_clip), plt_show=plt_show, plt_name=f"plots/{filename}_cost_gradient_clip_histogram.png")

        X_clip = np.array(X_clip)
        # X_mean = X_clip.mean(axis=0)
        # X_std  = X_clip.std(axis=0)
        # X_normalized = (X_clip - X_mean) / X_std
        # print(f"Normalization X: mean={X_mean}, std={X_std}")
        # Perform min-max scaling
        X_min = X_clip.min(axis=0)
        X_max = X_clip.max(axis=0)
        X_normalized = 2*(X_clip - X_min) / (X_max - X_min)-1
        print(f"Min-Max Scaling X: min={X_min}, max={X_max}")
        
        V_clip = np.array(V_clip)
        V_min = V_clip.min()
        V_max = V_clip.max()
        V_scaled = (V_clip - V_min) / (V_max - V_min)
        dVdx_scaled = (X_max - X_min) / 2 * dVdx_clip / (V_max-V_min)
        plot_costs_histogram(np.array(dVdx_scaled), plt_show=plt_show, plt_name=f"plots/{filename}_cost_gradient_scaled_histogram.png")
        print(f"Normalization V: min={V_min:.3f}, max={V_max:.3f}")
        
        # verify that scaled gradients are correct!
        # plot_3d_costs(X_normalized, V_scaled, cost_gradient=dVdx_scaled, zlim=1)

        train_ratio = 0.8
        X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
        V_tensor = torch.tensor(V_scaled, dtype=torch.float32)
        dVdx_tensor = torch.tensor(dVdx_scaled, dtype=torch.float32)
        dataset_size = len(X_tensor)
        train_size = int(dataset_size * train_ratio)
        val_size = dataset_size - train_size
        train_data, val_data = torch.utils.data.random_split(list(zip(X_tensor, V_tensor, dVdx_tensor)), [train_size, val_size])
        
        batch_size = 128
        train_loader = torch.utils.data.DataLoader(AmpcValueDataset(train_data), batch_size=batch_size, shuffle=True)
        val_loader   = torch.utils.data.DataLoader(AmpcValueDataset(val_data), batch_size=batch_size, shuffle=False)

    # model = MLP(input_dim=len(X_clip[0]))
    # model = PsdChol(input_dim=len(X_clip[0]))
    model = PsdCholOff(input_dim=len(X_min))
    # model = PsdCholLatentLin(input_dim=len(X_clip[0]))
    # criterion = torch.nn.MSELoss()
    # criterion = torch.nn.L1Loss()
    dmax = 0.2
    criterion = LocalShapeLoss(dmax=dmax)
    # criterion = SobolLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
    
    epochs = 300
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x_batch, v_batch, dV_batch in train_loader:
            optimizer.zero_grad()
            
            if type(model) in (PsdChol,PsdCholLatentLin):
                predictions, _ = model(x_batch)
                loss = criterion(predictions.squeeze(), v_batch)
            if type(model) is PsdCholOff:
                predictions, psd_matrices, x_offset = model(x_batch)
                if type(criterion) in (LocalShapeLoss, SobolLoss):
                    loss = criterion(x_batch, psd_matrices, x_offset, v_batch, dV_batch)
                else:
                    loss = criterion(predictions.squeeze(), v_batch)+10*criterion(x_offset.squeeze(), torch.zeros_like(x_offset.squeeze()))
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, v_batch, dV_batch in val_loader:
                if type(model) in (PsdChol,PsdCholLatentLin):
                    predictions, psd_matrices = model(x_batch)
                    loss = criterion(predictions.squeeze(), v_batch)
                if type(model) is PsdCholOff:
                    predictions, psd_matrices, x_offsets = model(x_batch)
                    if type(criterion) in (LocalShapeLoss, SobolLoss):
                        loss = criterion(x_batch, psd_matrices, x_offsets, v_batch, dV_batch)
                    else:
                        loss = criterion(predictions.squeeze(), v_batch)+10*criterion(x_offset.squeeze(), torch.zeros_like(x_offset.squeeze()))
            
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
    
    if type(model) in (PsdChol,PsdCholLatentLin):
        for xy, v_true, v_predict, P_predict in zip(x_batch.detach().cpu().numpy(), v_batch.detach().cpu().numpy(), predictions.detach().cpu().numpy(), psd_matrices.detach().cpu().numpy()):
            if type(model) is PsdCholLatentLin:
                W, b = model.get_W_b()
                plot_3d_costs(X_normalized, V_scaled, xy, v_true, v_predict, P_predict, [0,0], linLatW=W, linLatb=b, zlim=1)
            else:
                plot_3d_costs(X_normalized, V_scaled, xy, v_true, v_predict, P_predict, [0,0], zlim=1)
    if type(model) is PsdCholOff:
        for xy, v_true, v_predict, P_predict, x_off in zip(x_batch.detach().cpu().numpy(), v_batch.detach().cpu().numpy(), predictions.detach().cpu().numpy(), psd_matrices.detach().cpu().numpy(), x_offsets.detach().cpu().numpy()):
            plot_3d_costs(X_normalized, V_scaled, xy, v_true, v_predict, P_predict, x_off, zlim=1,dmax=dmax)
    
if __name__=="__main__":
    # fire.Fire({
    #     "train": train
    # })
    
    train(filename="unicycle_2D_900")