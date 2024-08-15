import pickle
import math
import optuna
import csv
import time
from gym import LEGGED_GYM_ROOT_DIR

import torch
from torch import nn  # noqa F401
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial import KDTree

from tqdm import tqdm

from learning.modules.critic import Critic  # noqa F401
from learning.modules.lqrc import *  # noqa F401
from learning.utils import (
    create_uniform_generator,
)
from utils import DEVICE
from critic_params_ampc import critic_params
from tensordict import TensorDict

from learning.modules.lqrc.plotting import (
    plot_learning_progress,
    plot_binned_errors_ampc
)
from learning.modules.utils.neural_net import export_network

# make dir for saving this run's results
time_str = time.strftime("%Y%m%d_%H%M%S")
save_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "offline_critics_graph", time_str)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# set up critics
for critic_param in critic_params.values():
    critic_param["device"] = DEVICE

critic_names = [
    "Critic",
    # "OuterProduct",
    # "PDCholeskyInput",
    # "CholeskyLatent",
    "DenseSpectralLatent",
]
print("Loading data")
# load data
with open(f"{LEGGED_GYM_ROOT_DIR}/learning/modules/lqrc/dataset.pkl", "rb") as f:
    data = pickle.load(f)
x0 = np.array(data["x0"]) # (3478114, 10)
cost = np.array(data["cost"]) # (3478114,)

# remove top 1% of cost values and corresponding states
num_to_remove = int(0.01 * len(cost))
top_indices = np.argsort(cost)[-num_to_remove:]
mask = np.ones(len(cost), dtype=bool)
mask[top_indices] = False
x0 = x0[mask]
cost = cost[mask]

# min max normalization to put state and cost on [0, 1]
x0_min = x0.min(axis=0)
x0_max = x0.max(axis=0)
x0 = (x0 - x0_min) / (x0_max - x0_min)
cost_min = cost.min()
cost_max = cost.max()
cost = (cost - cost_min) / (cost_max - cost_min)

d_max = 0

print("Building KDTree")
# Build the KDTree to get pairwise point distances
start = time.time()
tree = KDTree(x0)

random_x0 = np.random.uniform(low=-0.2, high=1.2, size=(10000, x0.shape[1]))

def process_batch(batch):
    indices = tree.query_ball_point(batch, d_max)
    mask = np.array([len(idx) == 0 for idx in indices])
    return batch[mask]

print("Creating non feasible state data points")
# filter random_x0 in batches
chunk_size = 10000
x0_non_fs_list = []
for i in range(0, len(random_x0), chunk_size):
    batch = random_x0[i:i+chunk_size]
    Y_filtered_batch = process_batch(batch)
    if len(Y_filtered_batch) > 0:
        x0_non_fs_list.append(Y_filtered_batch)
print("")

# concatenate all filtered batches and create matching cost array
x0_non_fs = np.concatenate(x0_non_fs_list) if x0_non_fs_list else None
cost_non_fs = np.ones((x0_non_fs.shape[0],))
# union non feasible and feasible states
x0 = np.concatenate((x0, x0_non_fs))
cost = np.concatenate((cost, cost_non_fs))

#hack to see data dist
import matplotlib.pyplot as plt
plt.hist(cost, bins=100)
plt.savefig(os.path.join(save_path, "data_dist.png"), dpi=300)

# turn numpy arrays to torch before training
x0 = torch.from_numpy(x0).float().to(DEVICE)
cost = torch.from_numpy(cost).float().to(DEVICE)

# set up constants
total_data = x0.shape[0]
n_dims = x0.shape[1]
graphing_data = {data_name: {name: {} for name in critic_names}
            for data_name in [
                "critic_obs",
                "values",
                "cost",
                "error",
            ]}
# set up training
# max_gradient_steps = 1000
# batch_size = 128
n_training_data = int(0.6 * total_data)
n_validation_data = total_data - n_training_data
print(f"training data: {n_training_data}, validation data: {n_validation_data}")
rand_perm = torch.randperm(total_data)
train_idx = rand_perm[0:n_training_data]
test_idx = rand_perm[n_training_data:]

data = TensorDict(
    {"critic_obs": x0.unsqueeze(dim=0), "cost": cost.unsqueeze(dim=0)},
    batch_size=(1, total_data),
    device=DEVICE,
)

def objective(trial, name):
    max_gradient_steps = trial.suggest_int("max_gradient_steps", low=100, high=5000, step=100)
    batch_size = trial.suggest_categorical("batch_size", [2**x for x in range(7, 10)])
    lr = trial.suggest_float("lr_critic", 1.0e-7, 1.0e-2, log=True)
    torch.cuda.empty_cache()
    print("")
    params = critic_params[name]
    if "critic_name" in params.keys():
        params.update(critic_params[params["critic_name"]])
    params["num_obs"] = n_dims

    critic_class = globals()[name]
    critic = critic_class(**params).to(DEVICE)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr)

    # train new critic
    mean_value_loss = 0
    counter = 0

    generator = create_uniform_generator(
        data[:1, train_idx],
        batch_size,
        max_gradient_steps=max_gradient_steps,
    )
    for batch in generator:
        value_loss = critic.loss_fn(
            batch["critic_obs"].squeeze(), batch["cost"].squeeze()
        )
        if counter == 0:
            with torch.no_grad():
                critic.value_offset.copy_(6476.1982)
        critic_optimizer.zero_grad()
        value_loss.backward()
        critic_optimizer.step()
        counter += 1
        with torch.no_grad():
            actual_error = (
                (
                    data["cost"][0, test_idx].squeeze()
                    - critic.evaluate(data["critic_obs"][0, test_idx])
                ).pow(2)
            ).to("cpu")
        # test_error[name].append(actual_error.detach().mean().numpy())
    print(f"{name} average error: ", actual_error.mean().item())
    print(f"{name} max error: ", actual_error.max().item())

    with torch.no_grad():
        graphing_data["error"][name] = actual_error
        graphing_data["critic_obs"][name] = data[0, :]["critic_obs"]
        graphing_data["values"][name] = critic.evaluate(
            data[0, :]["critic_obs"]
        )
        graphing_data["cost"][name] = data[0, :]["cost"]
    
    return actual_error.mean().item(), actual_error.max().item()

for name in critic_names:
    study = optuna.create_study(
        storage=f"sqlite:///{save_path}/{name}_db.sqlite3",
        directions=["minimize", "minimize"],
    )
    n_trials=100
    study.optimize(lambda trial: objective(trial, name), n_trials=n_trials)
    study.trials_dataframe().to_csv(save_path + f"/{name}.csv")
    best_trials = [trial.params for trial in study.best_trials]
    with open(save_path + f"/{name}_best_trials.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=best_trials[0].keys())
        writer.writeheader()
        writer.writerows(best_trials)

# plot_learning_progress(test_error, "AMPC VF Test Error", fn=f"{save_path}/test_error")


# plot_binned_errors_ampc(
#     graphing_data,
#     save_path + "/ampc",
#     title_add_on=f"Value Function for AMPC",
# )
