import pickle
import os
import time
from gym import LEGGED_GYM_ROOT_DIR
import tqdm
import torch
from torch import nn  # noqa F401
import numpy as np
import matplotlib.pyplot as plt

from learning.modules.critic import Critic  # noqa F401
from learning.modules.lqrc import *  # noqa F401

# from utils import (
#     DEVICE,
# )
from critic_params_ampc import critic_params
from tensordict import TensorDict

from learning.modules.lqrc.plotting import (
    plot_critic_3d_interactive,
    plot_learning_progress,
    plot_variable_lr,
    plot_grad_histogram,
)

from learning.modules.lqrc.utils import get_latent_matrix
import random

DEVICE = "cpu"


def set_deterministic():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(42)
    np.random.seed(42)


set_deterministic()


class AmpcValueDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data["critic_obs"][idx]
        v = self.data["cost"][idx]
        dv = self.data["grad"][idx]
        # x, v, dv = self.data[idx]
        return x, v, dv


SAVE_LOCALLY = True
LOAD_NORM = False
# choose critics to include in comparison
critic_names = [
    # "Diagonal",
    # "OuterProduct",
    # # "OuterProductLatent",
    "PDCholeskyInput",
    # "CholeskyInput",
    # "CholeskyLatent",
    # "DenseSpectralLatent",
]

# make dir for saving this run's results
model_path = os.path.join(LEGGED_GYM_ROOT_DIR, "models")
if not os.path.exists(model_path):
    os.makedirs(model_path)

latent_weight = None
latent_bias = None
# set up critics
for critic_param in critic_params.values():
    critic_param["device"] = DEVICE

# 2 DIM data
# with open(
#     f"{LEGGED_GYM_ROOT_DIR}/learning/modules/ampc/simple_unicycle/casadi/50_unicycle_soft_constraints_grad.pkl",
#     "rb",
# ) as f:
#     data = pickle.load(f)
# 4 DIM data

early_stopping = 0.8
with open(
    f"{LEGGED_GYM_ROOT_DIR}/learning/modules/ampc/simple_unicycle/casadi/4d_data_9261.pkl",
    "rb",
) as f:
    data = pickle.load(f)
    # ! hot fix to avoid regenerating data
    # data["gradients"] = data["cost_gradient"]
    # data.pop("cost_gradient")

x0 = (
    np.array(data["x0"])
    if not isinstance(data["x0"], list)
    else np.vstack([x[: int(early_stopping * x.shape[0])] for x in data["x0"]])
)
cost = (
    np.array(data["cost"])
    if not isinstance(data["cost"], list)
    else np.vstack(
        [c[: int(early_stopping * c.shape[0])].reshape(-1, 1) for c in data["cost"]]
    )
)
grad = (
    np.array(data["gradients"])
    if not isinstance(data["gradients"], list)
    else np.vstack([g[: int(early_stopping * g.shape[0])] for g in data["gradients"]])
)

n_dim = data["x0"][0].shape[-1]
if n_dim == 4:
    x0_plot = np.vstack([x[0] for x in data["x0"] if abs(x[0, 3]) <= 0.2])
    cost_plot = np.vstack(
        [c[0] for x, c in zip(data["x0"], data["cost"]) if abs(x[0, 3]) <= 0.2]
    )
    # dVdx_plot = [dv[0, :] for x, dv in zip(X, dVdx) if abs(x[0, 3]) <= 0.2]

# plot_grad_histogram(grad, "Gradient Histogram - Raw Data")

print(
    f"Raw data mean {x0.mean(axis=0)} \n median {np.median(x0, axis=0)} \n max {x0.max(axis=0)} \n min {x0.min(axis=0)}"
)

# remove states, gradients, and values where value > 20
top_indices = (cost > 20).squeeze()
mask = np.ones(len(cost), dtype=bool)
mask[top_indices] = False
x0 = x0[mask]
cost = cost[mask]
grad = grad[mask]
# repeat for plotting points
top_indices_plot = (cost_plot > 20).squeeze()
mask_plot = np.ones(len(cost_plot), dtype=bool)
mask_plot[top_indices_plot] = False
x0_plot = x0_plot[mask_plot]
cost_plot = cost_plot[mask_plot]

# hack to see data dist
# plt.hist(cost, bins=100)
# plt.show()
# plt.savefig(os.path.join("NNdata_dist.png"), dpi=300)

# plot_grad_histogram(grad, "Gradient Histogram - Clipped Data")

# Normalization
if LOAD_NORM and os.path.exists(f"{model_path}/normalization.pkl"):
    with open(f"{model_path}/normalization.pkl", "rb") as f:
        normalization_data = pickle.load(f)
        x0_min, x0_max = normalization_data["x0_min"], normalization_data["x0_max"]
        cost_min, cost_max = (
            normalization_data["cost_min"],
            normalization_data["cost_max"],
        )
else:
    # min max normalization to put state on [-1, 1] and cost on [0, 1]
    x0_min = x0.min(axis=0)
    x0_max = x0.max(axis=0)
    cost_min = cost.min()
    cost_max = cost.max()
    # Save normalization values if not provided
    with open(f"{model_path}/normalization.pkl", "wb") as f:
        pickle.dump(
            {
                "x0_min": x0_min,
                "x0_max": x0_max,
                "cost_min": cost_min,
                "cost_max": cost_max,
            },
            f,
        )

x0 = 2 * (x0 - x0_min) / (x0_max - x0_min) - 1
x0_plot = 2 * (x0_plot - x0_min) / (x0_max - x0_min) - 1
cost = (cost - cost_min) / (cost_max - cost_min)
cost_plot = (cost_plot - cost_min) / (cost_max - cost_min)
grad = ((x0_max - x0_min) / 2) * (grad / (cost_max - cost_min))
print(
    f"Normalized data mean {x0.mean(axis=0)} \n median {np.median(x0, axis=0)} \n max {x0.max(axis=0)} \n min {x0.min(axis=0)}"
)
# plot_grad_histogram(grad, "Gradient Histogram - Clipped + Normalized Data")
# ! make save fig show swap easy

# turn numpy arrays to torch before training
x0 = torch.from_numpy(x0).float().to(DEVICE)
cost = torch.from_numpy(cost).float().to(DEVICE)
grad = torch.from_numpy(grad).float().to(DEVICE)
# turn numpy arrays to torch to keep standard
x0_plot = torch.from_numpy(x0_plot).float().to(DEVICE)
cost_plot = torch.from_numpy(cost_plot).float().to(DEVICE)
print(
    "cost mean", cost.mean(), "cost median", cost.median(), "cost std dev", cost.std()
)


# set up constants
total_data = x0.shape[0]
graphing_data = {
    data_name: {name: {} for name in critic_names}
    for data_name in [
        "xy_eval",
        "cost_eval",
        "A",
        "x_offsets",
        "W_latent",
        "b_latent",
        "cost_true",
    ]
}
test_error = {name: [] for name in critic_names}
lr_history = {name: [] for name in critic_names}
loss_history = {
    name: []
    for name in [f"{critic}_train_loss" for critic in critic_names]
    + [f"{critic}_test_loss" for critic in critic_names]
}


# set up training
max_gradient_steps = 250  # 500
batch_size = 1024
n_training_data = int(0.8 * total_data)
n_validation_data = total_data - n_training_data
print(f"training data: {n_training_data}, validation data: {n_validation_data}")
# rand_perm = torch.randperm(total_data)
# train_idx = rand_perm[0:n_training_data]
# test_idx = rand_perm[n_training_data:]

# data = TensorDict(
#     {
#         "critic_obs": x0.unsqueeze(dim=0),
#         "cost": cost.unsqueeze(dim=0),
#         "grad": grad.unsqueeze(dim=0),
#     },
#     batch_size=(1, total_data),
#     device=DEVICE,
# )

train_data, val_data = torch.utils.data.random_split(
    list(zip(x0, cost, grad)), [n_training_data, n_validation_data]
)
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=False,
)

trained_critics = {}

standard_offset = 0
for ix, name in enumerate(critic_names):
    params = critic_params[name]
    if "critic_name" in params.keys():
        params.update(critic_params[params["critic_name"]])
    params["num_obs"] = n_dim

    critic_class = globals()[name]
    critic = critic_class(**params).to(DEVICE)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-2)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        critic_optimizer, mode="min", factor=0.5, patience=20
    )

    # train new critic
    for epoch in tqdm.tqdm(range(max_gradient_steps)):
        critic.train()
        train_loss = 0.0
        for obs_batch, cost_batch, grad_batch in train_loader:
            critic_optimizer.zero_grad()
            if "Latent" in name and epoch == 0:
                latent_weight, latent_bias = get_latent_matrix(
                    obs_batch.shape, critic.latent_NN, device=DEVICE
                )

            # calculate loss and optimize
            value_loss = critic.loss_fn(
                obs_batch.squeeze(),
                cost_batch.squeeze(),
                batch_grad=grad_batch.squeeze(),
                W_latent=latent_weight,
                b_latent=latent_bias,
            )
            value_loss.backward()
            critic_optimizer.step()
            train_loss += value_loss.item()

        # pointwise prediction test error
        critic.eval()
        test_loss = 0.0
        with torch.no_grad():
            # actual_error = (
            #     (
            #         data["cost"][0, test_idx].squeeze()
            #         - critic.evaluate(data["critic_obs"][0, test_idx])
            #     ).pow(2)
            # ).to("cpu")
            for obs_batch, cost_batch, grad_batch in test_loader:
                value_loss = critic.loss_fn(
                    obs_batch.squeeze(),
                    cost_batch.squeeze(),
                    batch_grad=grad_batch.squeeze(),
                    W_latent=latent_weight,
                    b_latent=latent_bias,
                )
                test_loss += value_loss.item()
        # scale by dataset size
        train_loss /= len(train_loader)
        test_loss /= len(test_loader)
        current_lr = critic_optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{max_gradient_steps}, Train Loss: {train_loss:.4f}, Val Loss: {test_loss:.4f}, LR: {current_lr:.6f}"
        )
        # update graphing trakcers
        # test_error[name].append(actual_error.detach().mean().numpy())
        lr_history[name].append(lr_scheduler.get_last_lr()[0])
        loss_history[f"{name}_train_loss"].append(train_loss)
        loss_history[f"{name}_test_loss"].append(test_loss)
        lr_scheduler.step(test_loss)

    # print(f"{name} average error: ", actual_error.mean().item())
    # print(f"{name} max error: ", actual_error.max().item())
    trained_critics[name] = critic
    torch.save(critic.state_dict(), f"{model_path}/{type(critic).__name__}.pth")

if SAVE_LOCALLY:
    time_str = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", "offline_critics_graph", time_str
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # plot_learning_progress(
    #     test_error,
    #     "Pointwise Error on Test Set \n (Comparison of Normed Vals Used in Supervised Training)",
    #     fn=f"{save_path}/test_error",
    #     log_scale=False,
    # )
    plot_variable_lr(lr_history, f"{save_path}/lr_history")
    plot_learning_progress(
        loss_history,
        "Loss History over Training Epochs",
        fn=f"{save_path}/loss_history",
        y_label1="Average Value Loss",
        y_label2="Change in Average Value Loss",
        log_scale=False,
    )

for eval_ix in [150, 210, 221, 238, 329, 84, 103, 165, 298]:
    for name in critic_names:
        critic = trained_critics[name]
        with torch.no_grad():
            eval_pt = data[0, eval_ix]["critic_obs"] if n_dim == 2 else x0_plot[eval_ix]
            graphing_data["xy_eval"][name] = eval_pt
            prediction = critic.evaluate(eval_pt, return_all=True)
            graphing_data["cost_eval"][name] = prediction.get("value")
            graphing_data["A"][name] = prediction["A"]
            graphing_data["x_offsets"][name] = prediction["x_offsets"]
            graphing_data["W_latent"][name] = (
                latent_weight if "Latent" in name else None
            )
            graphing_data["b_latent"][name] = latent_bias if "Latent" in name else None
            graphing_data["cost_true"][name] = (
                data[0, eval_ix]["cost"] if n_dim == 2 else cost_plot[eval_ix]
            )

    if n_dim == 2:
        plot_critic_3d_interactive(
            x0,
            cost,
            graphing_data["A"],
            graphing_data["xy_eval"],
            graphing_data["cost_true"],
            graphing_data["cost_eval"],
            graphing_data["x_offsets"],
            display_names={
                "Diagonal": "Diagonal",
                "CholeskyInput": "Cholesky",
                "PDCholeskyInput": "PD Cholesky",
                "OuterProduct": "Outer Product",
                "CholeskyLatent": "Cholesky Latent",
                "DenseSpectralLatent": "Spectral Latent",
                "Critic": "Critic",
            },
            W_latent=graphing_data["W_latent"],
            b_latent=graphing_data["b_latent"],
            dmax=0.2,
        )
    elif n_dim == 4:
        plot_critic_3d_interactive(
            x0_plot[:, :2],
            cost_plot,
            graphing_data["A"],
            graphing_data["xy_eval"],
            graphing_data["cost_true"],
            graphing_data["cost_eval"],
            graphing_data["x_offsets"],
            display_names={
                "Diagonal": "Diagonal",
                "CholeskyInput": "Cholesky",
                "PDCholeskyInput": "PD Cholesky",
                "OuterProduct": "Outer Product",
                "CholeskyLatent": "Cholesky Latent",
                "DenseSpectralLatent": "Spectral Latent",
                "Critic": "Critic",
            },
            W_latent=graphing_data["W_latent"],
            b_latent=graphing_data["b_latent"],
            dmax=0.2,
        )
