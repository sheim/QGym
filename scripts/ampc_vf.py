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
)

from learning.modules.lqrc.utils import get_latent_matrix

SAVE_LOCALLY = False

# choose critics to include in comparison
critic_names = [
    "OuterProduct",
    # "OuterProductLatent",
    # "PDCholeskyInput",
    "CholeskyInput",
    "CholeskyLatent",
    "DenseSpectralLatent",
]

# make dir for saving this run's results
time_str = time.strftime("%Y%m%d_%H%M%S")
save_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "offline_critics_graph", time_str)
if SAVE_LOCALLY and not os.path.exists(save_path):
    os.makedirs(save_path)

latent_weight = None
latent_bias = None
# set up critics
for critic_param in critic_params.values():
    critic_param["device"] = DEVICE


print("Loading data")
with open(
    f"{LEGGED_GYM_ROOT_DIR}/learning/modules/ampc/simple_unicycle/casadi/50_unicycle_soft_constraints_grad.pkl",
    "rb",
) as f:
    data = pickle.load(f)
x0 = np.array(data["x0"])
cost = np.array(data["cost"])
grad = np.array(data["gradients"])
eval_ix = len(x0) // 4

print(
    f"Raw data mean {x0.mean(axis=0)} \n median {np.median(x0, axis=0)} \n max {x0.max(axis=0)} \n min {x0.min(axis=0)}"
)

# remove top 10% of cost values and corresponding states
num_to_remove = math.ceil(0.1 * len(cost))
top_indices = np.argsort(cost)[-num_to_remove:]
mask = np.ones(len(cost), dtype=bool)
mask[top_indices] = False
x0 = x0[mask]
cost = cost[mask]
grad = grad[mask]
# optimal_u = optimal_u[mask]
n_samples = x0.shape[0]

# min max normalization to put state and cost on [0, 1]
x0_min = x0.min(axis=0)
x0_max = x0.max(axis=0)
x0 = 2 * (x0 - x0_min) / (x0_max - x0_min) - 1
cost_min = cost.min()
cost_max = cost.max()
cost = (cost - cost_min) / (cost_max - cost_min)
grad = (x0_max - x0_min) / 2 * grad / (cost_max - cost_min)

print(
    f"Normalized data mean {x0.mean(axis=0)} \n median {np.median(x0, axis=0)} \n max {x0.max(axis=0)} \n min {x0.min(axis=0)}"
)

# turn numpy arrays to torch before training
x0 = torch.from_numpy(x0).float().to(DEVICE)
cost = torch.from_numpy(cost).float().to(DEVICE)
grad = torch.from_numpy(grad).float().to(DEVICE)

print(
    "cost mean", cost.mean(), "cost median", cost.median(), "cost std dev", cost.std()
)

# set up constants
total_data = x0.shape[0]
n_dims = x0.shape[1]
graphing_data = {
    data_name: {name: {} for name in critic_names}
    for data_name in [
        "xy_eval",
        "cost_eval",
        "A",
        "W_latent",
        "b_latent",
        "cost_true",
    ]
}
test_error = {name: [] for name in critic_names}
lr_history = {name: [] for name in critic_names}

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
    params["num_obs"] = n_dims

    critic_class = globals()[name]
    critic = critic_class(**params).to(DEVICE)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        critic_optimizer, mode="min", factor=0.5, patience=100, threshold=1e-5
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
            with torch.no_grad():
                critic.value_offset.copy_(standard_offset)
            print(f"{name} value offset after mean assigning", critic.value_offset)

        if "Latent" in name:
            latent_weight, latent_bias = get_latent_matrix(
                batch["critic_obs"].shape, critic.latent_NN, device=DEVICE
            )
            latent_weight = latent_weight.cpu().detach().numpy()
            latent_bias = latent_bias.cpu().detach().numpy()

        # calculate loss and optimize
        value_loss = critic.loss_fn(
            batch["critic_obs"].squeeze(),
            batch["cost"].squeeze(),
            batch_grad=batch["grad"].squeeze(),
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

    print(f"{name} average error: ", actual_error.mean().item())
    print(f"{name} max error: ", actual_error.max().item())

    with torch.no_grad():
        graphing_data["xy_eval"][name] = data[0, eval_ix]["critic_obs"]
        prediction = critic.evaluate(data[0, eval_ix]["critic_obs"], return_all=True)
        graphing_data["cost_eval"][name] = prediction.get("value")
        graphing_data["A"][name] = prediction.get("A")
        graphing_data["W_latent"][name] = latent_weight if "Latent" in name else None
        graphing_data["b_latent"][name] = latent_bias if "Latent" in name else None
        graphing_data["cost_true"][name] = data[0, eval_ix]["cost"]

plot_critic_3d_interactive(
    x0,
    cost,
    graphing_data["A"],
    graphing_data["xy_eval"],
    graphing_data["cost_true"],
    graphing_data["cost_eval"],
    display_names={
        "CholeskyInput": "Cholesky",
        "OuterProduct": "Outer Product",
        "CholeskyLatent": "Cholesky Latent",
        "DenseSpectralLatent": "Spectral Latent",
        "Critic": "Critic",
    },
    W_latent=graphing_data["W_latent"],
    b_latent=graphing_data["b_latent"],
)
