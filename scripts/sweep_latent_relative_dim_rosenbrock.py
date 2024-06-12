import time
import matplotlib.pyplot as plt  # noqa F401
import numpy as np  # noqa F401
from learning.modules.critic import Critic  # noqa F401
from learning.modules.lqrc import *  # noqa F401
from learning.modules.lqrc import *  # noqa F401
from learning.utils import (
    compute_generalized_advantages,
    compute_MC_returns,
    create_uniform_generator,
)
from learning.modules.lqrc.plotting import (
    plot_dim_sweep_mean_std,
    plot_learning_progress,
)
from gym import LEGGED_GYM_ROOT_DIR
import os
import shutil

import torch
from torch import nn  # noqa F401

# from critic_params import critic_params
from critic_params_rosenbrock import critic_params
from optimizer_params import optimizer_params
from utils import *
from tensordict import TensorDict


for critic_param in critic_params.values():
    critic_param["device"] = DEVICE

# generate data
n_dims = 3
grid_resolution = 50
total_data = grid_resolution**n_dims
input, target = generate_bounded_rosenbrock(n_dims, lb=0.0, ub=2.0, steps=grid_resolution)

# handle some bookkeeping
time_str = time.strftime("%Y%m%d_%H%M%S")

save_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "offline_critics_graph", time_str)
if not os.path.exists(save_path):
    os.makedirs(save_path)


gamma = 0.95
lam = 1.0
tot_iter = 100
iter_offset = 0
iter_step = 2
max_gradient_steps = 1000
# max_grad_norm = 1.0
batch_size = 128
n_training_data = int(0.6 * total_data)
n_validation_data = total_data - n_training_data
print(f"training data: {n_training_data}, validation data: {n_validation_data}")
rand_perm = torch.randperm(total_data)
train_idx = rand_perm[0:n_training_data]
test_idx = rand_perm[n_training_data:]


num_trials = 3
size = 6
step = 1
x = np.arange(1, size + 2, step)
y = np.arange(1, size + 2, step)
xx, yy = np.meshgrid(x, y, indexing="xy")
graphing_data = np.zeros((num_trials, size // step, size // step, 2)) - float("inf")


data = TensorDict(
    {"critic_obs": input.unsqueeze(dim=0), "returns": target.unsqueeze(dim=0)},
    batch_size=(1, total_data),
    device=DEVICE,
)

for trial in range(num_trials):
    for i in range(0, size // step):
        for j in range(0, size // step):
            test_error = {"DenseSpectralLatent": []}
            # np.savez(
            #     save_path + "/graphing_data.npz",
            #     relative_dim=xx,
            #     latent_dim=yy,
            #     mean=graphing_data[..., 0],
            #     max=graphing_data[..., 1],
            # )
            torch.cuda.empty_cache()
            name = "DenseSpectralLatent"
            params = critic_params[name]
            rel_dim = int(xx[i, j].item())
            lat_dim = int(yy[i, j].item())
            # print("relative dim", rel_dim, "latent dim", lat_dim)
            if lat_dim < rel_dim:
                continue

            params["relative_dim"] = rel_dim
            params["latent_dim"] = lat_dim
            # set up critic
            if "critic_name" in params.keys():
                params.update(critic_params[params["critic_name"]])
            critic_class = globals()[name]
            critic = critic_class(**params).to(DEVICE)
            critic_optimizer = torch.optim.Adam(
                critic.parameters(), lr=optimizer_params[name]["lr"]
            )

            for iteration in range(iter_offset, tot_iter, iter_step):
                # base_data = torch.load(
                #     os.path.join(log_dir, "data_{}.pt".format(500))
                # ).to(DEVICE)
                # compute ground-truth
                # episode_rollouts = compute_MC_returns(base_data, gamma)
                # print(f"Initializing value offset to: {episode_rollouts.mean().item()}")

                print("")
                # if hasattr(test_critics[name], "value_offset"):
                # with torch.no_grad():
                #     critic.value_offset.copy_(episode_rollouts.mean())

                # data = base_data.detach().clone()
                # # train new critic
                # data["values"] = critic.evaluate(data["critic_obs"])
                # try:
                #     data["advantages"] = compute_generalized_advantages(
                #         data, gamma, lam, critic
                #     )
                # except:
                #     print("relative dim", rel_dim, "latent_dim", lat_dim)
                # data["returns"] = data["advantages"] + data["values"]

                mean_value_loss = 0
                counter = 0

                generator = create_uniform_generator(
                    data[:1, train_idx],
                    batch_size,
                    max_gradient_steps=max_gradient_steps,
                )
                for batch in generator:
                    # print("batch['critic_obs'].squeeze()",  batch["critic_obs"].squeeze().shape)
                    # print("batch['returns'].squeeze()", batch["returns"].shape)
                    # exit()
                    value_loss = critic.loss_fn(
                        batch["critic_obs"].squeeze(), batch["returns"].squeeze()
                    )
                    
                    critic_optimizer.zero_grad()
                    value_loss.backward()
                    critic_optimizer.step()
                    counter += 1
                    with torch.no_grad():
                        actual_error = (
                            (
                                data["returns"][0, test_idx].squeeze()
                                - critic.evaluate(data["critic_obs"][0, test_idx])
                            ).pow(2)
                        ).to("cpu")
                    test_error[name].append(actual_error.detach().mean().numpy())
                print(f"{name} average error: ", actual_error.mean().item())
                print(f"{name} max error: ", actual_error.max().item())
                mean_value_loss /= counter
                # episode_rollouts = compute_MC_returns(data, gamma)
                # with torch.no_grad():
                #     actual_error = (
                #         critic.evaluate(data["critic_obs"][0]) - episode_rollouts[0]
                #     ).pow(2)
            mean_actual_error = actual_error.mean().item()
            max_actual_error = actual_error.max().item()
            graphing_data[trial, i, j, 0] = mean_actual_error
            graphing_data[trial, i, j, 1] = max_actual_error
            plot_learning_progress(
                test_error,
                fn=save_path + f"/latent_{lat_dim}_relative_{rel_dim}",
                smoothing_window=50,
            )

mask = np.isfinite(graphing_data)
avg_graphing_data = np.where(mask, graphing_data, np.nan).mean(axis=0)
std_graphing_data = np.where(mask, graphing_data, np.nan).std(axis=0)


plot_dim_sweep_mean_std(
    xx,
    yy,
    avg_graphing_data[..., 0],
    avg_graphing_data[..., 1],
    std_graphing_data[..., 0],
    std_graphing_data[..., 1],
    fn=save_path + "/latent_relative_dim_sweep",
    trial_num=num_trials,
    title=f"Prediction Error on {n_dims + 1}D Rosenbrock Function vs Rank(A) and Dim(A)"
)


this_file = os.path.join(LEGGED_GYM_ROOT_DIR, "scripts", "multiple_offline_critics.py")
params_file = os.path.join(LEGGED_GYM_ROOT_DIR, "scripts", "critic_params.py")
shutil.copy(this_file, os.path.join(save_path, os.path.basename(this_file)))
shutil.copy(params_file, os.path.join(save_path, os.path.basename(params_file)))
