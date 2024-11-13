import pickle
import math
import random
import os
import time
from gym import LEGGED_GYM_ROOT_DIR

import torch
from torch import nn  # noqa F401
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

from learning.modules.critic import Critic  # noqa F401
from learning.modules.lqrc import *  # noqa F401
from learning.utils import (
    create_uniform_generator,
)
from utils import (
    DEVICE,
)
from critic_params_ampc import critic_params
from tensordict import TensorDict

from learning.modules.lqrc.plotting import (
    plot_critic_3d_interactive,
    plot_learning_progress,
    plot_variable_lr,
)

from learning.modules.lqrc.utils import get_latent_matrix

SAVE_LOCALLY = True
LOAD_NORM = False
# choose critics to include in comparison
critic_names = [
    "Diagonal",
    "OuterProduct",
    # # "OuterProductLatent",
    # # "PDCholeskyInput",
    "CholeskyInput",
    "CholeskyLatent",
    "DenseSpectralLatent",
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

n_dim = data["x0"][0].shape[-1]
if n_dim == 4:
    x0_plot = np.vstack([x[0] for x in data["x0"] if abs(x[0, 3]) <= 0.2])
    cost_plot = np.vstack(
        [c[0] for x, c in zip(data["x0"], data["cost"]) if abs(x[0, 3]) <= 0.2]
    )
    # dVdx_plot = [dv[0, :] for x, dv in zip(X, dVdx) if abs(x[0, 3]) <= 0.2]

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

# hack to see data dist
plt.hist(cost, bins=100)
# plt.show()
# plt.savefig(os.path.join("NNdata_dist.png"), dpi=300)

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

# ! make save fig show swap easy

# turn numpy arrays to torch before training
x0 = torch.from_numpy(x0).float().to(DEVICE)
cost = torch.from_numpy(cost).float().to(DEVICE)
grad = torch.from_numpy(grad).float().to(DEVICE)
# turn numpy arrays to torch to keep standard
x0_plot = torch.from_numpy(x0_plot).float().to(DEVICE)
cost_plot = torch.from_numpy(cost_plot).float().to(DEVICE)
eval_ix = len(x0) // 4 + 50 if n_dim == 2 else len(x0_plot) // 4 + 40
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
loss_history = {name: [] for name in critic_names}

# set up training
max_gradient_steps = 500
batch_size = 512
n_training_data = int(0.6 * total_data)
n_validation_data = total_data - n_training_data
print(f"training data: {n_training_data}, validation data: {n_validation_data}")
rand_perm = torch.randperm(total_data)
train_idx = rand_perm[0:n_training_data]
test_idx = rand_perm[n_training_data:]

data = TensorDict(
    {
        "critic_obs": x0.unsqueeze(dim=0),
        "cost": cost.unsqueeze(dim=0),
        "grad": grad.unsqueeze(dim=0),
    },
    batch_size=(1, total_data),
    device=DEVICE,
)

standard_offset = 0
for ix, name in enumerate(critic_names):
    params = critic_params[name]
    if "critic_name" in params.keys():
        params.update(critic_params[params["critic_name"]])
    params["num_obs"] = n_dim

    critic_class = globals()[name]
    critic = critic_class(**params).to(DEVICE)
    # critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3, weight_decay=0.01)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     critic_optimizer, mode="min", factor=0.5, patience=100, threshold=1e-5
    # )
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-2)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        critic_optimizer, mode="min", factor=0.5, patience=20
    )

    # train new critic
    mean_value_loss = 0
    counter = 0

    generator = create_uniform_generator(
        data[:1, train_idx],
        batch_size,
        max_gradient_steps=max_gradient_steps,
    )

    for batch in generator:
        # print offset to check it's working as intended
        if counter == 0:
            if ix == 0:
                standard_offset = batch["cost"].mean()
            print(f"{name} value offset before mean assigning", critic.value_offset)
            if params["c_offset"]:
                with torch.no_grad():
                    critic.value_offset.copy_(standard_offset)
                print(f"{name} value offset after mean assigning", critic.value_offset)

        if "Latent" in name:
            latent_weight, latent_bias = get_latent_matrix(
                batch["critic_obs"].shape, critic.latent_NN, device=DEVICE
            )

        # calculate loss and optimize
        value_loss = critic.loss_fn(
            batch["critic_obs"].squeeze(),
            batch["cost"].squeeze(),
            batch_grad=batch["grad"].squeeze(),
            W_latent=latent_weight,
            b_latent=latent_bias,
        )
        critic_optimizer.zero_grad()
        value_loss.backward()
        critic_optimizer.step()
        lr_scheduler.step(value_loss)
        counter += 1
        # pointwise prediction test error
        with torch.no_grad():
            actual_error = (
                (
                    data["cost"][0, test_idx].squeeze()
                    - critic.evaluate(data["critic_obs"][0, test_idx])
                ).pow(2)
            ).to("cpu")
        test_error[name].append(actual_error.detach().mean().numpy())
        lr_history[name].append(lr_scheduler.get_last_lr()[0])
        loss_history[name].append(value_loss.cpu().detach().mean().numpy())

    print(f"{name} average error: ", actual_error.mean().item())
    print(f"{name} max error: ", actual_error.max().item())

    with torch.no_grad():
        eval_pt = data[0, eval_ix]["critic_obs"] if n_dim == 2 else x0_plot[eval_ix]
        graphing_data["xy_eval"][name] = eval_pt
        prediction = critic.evaluate(eval_pt, return_all=True)
        graphing_data["cost_eval"][name] = prediction.get("value")
        graphing_data["A"][name] = prediction["A"]
        graphing_data["x_offsets"][name] = prediction["x_offsets"]
        graphing_data["W_latent"][name] = latent_weight if "Latent" in name else None
        graphing_data["b_latent"][name] = latent_bias if "Latent" in name else None
        graphing_data["cost_true"][name] = (
            data[0, eval_ix]["cost"] if n_dim == 2 else cost_plot[eval_ix]
        )

    torch.save(critic.state_dict(), f"{model_path}/{type(critic).__name__}.pth")

if SAVE_LOCALLY:
    time_str = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", "offline_critics_graph", time_str
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plot_learning_progress(
        test_error,
        "Pointwise Error on Test Set \n (Comparison of Normed Vals Used in Supervised Training)",
        fn=f"{save_path}/test_error",
    )
    plot_variable_lr(lr_history, f"{save_path}/lr_history")
    plot_learning_progress(
        loss_history,
        "Loss History over Training Epochs",
        fn=f"{save_path}/loss_history",
        y_label1="Average Value Loss",
        y_label2="Change in Average Value Loss",
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
    for name, val in graphing_data["A"].items():
        print(f"{name}: min A", val.min(), "max A", val.max(), "mean A", val.mean())
    for name, val in graphing_data["W_latent"].items():
        if val is not None:
            print(
                f"{name}: min W_latent",
                val.min(),
                "max W_latent",
                val.max(),
                "mean W_latent",
                val.mean(),
            )
    for name, val in graphing_data["b_latent"].items():
        if val is not None:
            print(
                f"{name}: min b_latent",
                val.min(),
                "max b_latent",
                val.max(),
                "mean b_latent",
                val.mean(),
            )

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
            "OuterProduct": "Outer Product",
            "CholeskyLatent": "Cholesky Latent",
            "DenseSpectralLatent": "Spectral Latent",
            "Critic": "Critic",
        },
        W_latent=graphing_data["W_latent"],
        b_latent=graphing_data["b_latent"],
        dmax=0.2,
    )
