import time
import matplotlib.pyplot as plt  # noqa F401
import numpy as np  # noqa F401
from learning.modules.critic import Critic  # noqa F401
from learning.modules.lqrc import *  # noqa F401
from learning.utils import (
    create_uniform_generator,
)
from learning.modules.lqrc.plotting import (
    plot_rosenbrock_multiple_critics_w_data,
    plot_learning_progress,
    plot_binned_errors,
)
from gym import LEGGED_GYM_ROOT_DIR
import os
import shutil

import torch
from torch import nn  # noqa F401

# from critic_params import critic_params
from critic_params_rosenbrock import critic_params
from utils import * # noqa F403
from tensordict import TensorDict


for critic_param in critic_params.values():
    critic_param["device"] = DEVICE


time_str = time.strftime("%Y%m%d_%H%M%S")


critic_names = [
    "Critic",
    # "CholeskyInput",
    # "CholeskyLatent",
    "OuterProduct",
    "OuterProductLatent",
    # # "PDCholeskyInput",
    # # "PDCholeskyLatent",
    # "QPNet",
    # # "SpectralLatent",
    "DenseSpectralLatent",
    # # ]
    # "Cholesky",
    # "CholeskyPlusConst",
    # "CholeskyOffset1",
    # "CholeskyOffset2",
    # "NN_wQR",
    # "NN_wLinearLatent",
]

# generate data
n_dims = 2
grid_resolution = 50
total_data = grid_resolution**n_dims
x, target = generate_bounded_rosenbrock(n_dims, lb=0.0, ub=2.0, steps=grid_resolution)

# set up training
tot_iter = 1
iter_offset = 0
iter_step = 2
max_gradient_steps = 200
# max_grad_norm = 1.0
batch_size = 128
n_training_data = int(0.6 * total_data)
n_validation_data = total_data - n_training_data
print(f"training data: {n_training_data}, validation data: {n_validation_data}")
rand_perm = torch.randperm(total_data)
train_idx = rand_perm[0:n_training_data]
test_idx = rand_perm[n_training_data:]


all_graphing_names = ["Rosenbrock"] + critic_names.copy()
learning_rates = [1e-3, 1e-4, 1e-5]
graphing_data = generate_rosenbrock_g_data_dict(all_graphing_names, learning_rates)

test_error = {lr: {name: [] for name in critic_names} for lr in learning_rates}


for g_data in graphing_data.values():
    g_data["critic_obs"]["Rosenbrock"] = x
    g_data["values"]["Rosenbrock"] = target
    g_data["returns"]["Rosenbrock"] = target
    g_data["error"]["Rosenbrock"] = torch.zeros_like(target)


data = TensorDict(
    {"critic_obs": x.unsqueeze(dim=0), "returns": target.unsqueeze(dim=0)},
    batch_size=(1, total_data),
    device=DEVICE,
)

for lr in learning_rates:
    for iteration in range(iter_offset, tot_iter, iter_step):
        torch.cuda.empty_cache()

        for name in critic_names:
            print("")
            params = critic_params[name]
            if "critic_name" in params.keys():
                params.update(critic_params[params["critic_name"]])
            params["num_obs"] = n_dims

            critic_class = globals()[name]
            critic = critic_class(**params).to(DEVICE)
            critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr)

            # with torch.no_grad():
            #     critic.value_offset.copy_(data["returns"].mean())
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
                test_error[lr][name].append(actual_error.detach().mean().numpy())
            print(f"{name} average error: ", actual_error.mean().item())
            print(f"{name} max error: ", actual_error.max().item())

            if n_dims == 2:
                with torch.no_grad():
                    graphing_data[lr]["error"][name] = actual_error
                    graphing_data[lr]["critic_obs"][name] = data[0, :]["critic_obs"]
                    graphing_data[lr]["values"][name] = critic.evaluate(
                        data[0, :]["critic_obs"]
                    )
                    graphing_data[lr]["returns"][name] = data[0, :]["returns"]

# compare new and old critics
save_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "offline_critics_graph", time_str)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# plots
for lr, t_error in test_error.items():
    plot_learning_progress(
        t_error,
        fn=save_path + f"/{len(critic_names)}_lr {lr}",
        smoothing_window=50,
    )

g_data_no_ground_truth = generate_rosenbrock_g_data_dict(critic_names, learning_rates)
for lr, value in graphing_data.items():
    for name in all_graphing_names:
        if name == "Rosenbrock":
            continue
        g_data_no_ground_truth[lr]["critic_obs"][name] = value["critic_obs"][name]
        g_data_no_ground_truth[lr]["values"][name] = value["values"][name]
        g_data_no_ground_truth[lr]["returns"][name] = value["returns"][name]
        g_data_no_ground_truth[lr]["error"][name] = value["error"][name]

plot_binned_errors(g_data_no_ground_truth,
                    save_path + f"/rosenbrock",
                    title_add_on=f"Rosenbrock")

if n_dims == 2:
    for lr, g_data in graphing_data.items():
        plot_rosenbrock_multiple_critics_w_data(
            g_data["critic_obs"],
            g_data["values"],
            g_data["returns"],
            title=f"Learning a {n_dims + 1}D Rosenbrock Function \n Learning Rate: {lr}, Without Nonlinear Latent Activation",
            fn=save_path + f"/{len(critic_names)}_lr_{lr}",
            data=data[0, train_idx]["critic_obs"],
            grid_size=grid_resolution,
        )

this_file = os.path.join(LEGGED_GYM_ROOT_DIR, "scripts", "lr_rosenbrock.py")
params_file = os.path.join(LEGGED_GYM_ROOT_DIR, "scripts", "critic_params_osc.py")
shutil.copy(this_file, os.path.join(save_path, os.path.basename(this_file)))
shutil.copy(params_file, os.path.join(save_path, os.path.basename(params_file)))
