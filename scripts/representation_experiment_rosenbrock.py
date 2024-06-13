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
    plot_pendulum_multiple_critics_w_data,
    plot_learning_progress,  # noqa F401
    plot_rosenbrock_multiple_critics_w_data,
    plot_binned_errors,
)
from gym import LEGGED_GYM_ROOT_DIR
import os
import shutil

import torch
from torch import nn  # noqa F401
from representation_experiment_params import experiment_params
from utils import *
from tensordict import TensorDict


for experiment in experiment_params.values():
    experiment["critic_params"]["device"] = DEVICE

time_str = time.strftime("%Y%m%d_%H%M%S")

# generate data
n_dims = 3
grid_resolution = 50
total_data = grid_resolution**n_dims
x, target = generate_bounded_rosenbrock(n_dims, lb=0.0, ub=2.0, steps=grid_resolution)

# set up critics
critic_names = []
test_critics = {}
for name, experiment in experiment_params.items():
    critic_names.append(experiment["critic_name"])
    critic_class = globals()[experiment["critic_name"]]
    params = experiment["critic_params"]
    params["num_obs"] = n_dims
    test_critics[name] = critic_class(**params).to(DEVICE)
critic_optimizers = {
    name: torch.optim.Adam(critic.parameters(), lr=1e-3) # ! LR fixed?
    for name, critic in test_critics.items()
}

# set up training
tot_iter = 1
iter_offset = 0
iter_step = 1
max_gradient_steps = 1000 #200
# max_grad_norm = 1.0
batch_size = 128
n_training_data = int(0.6 * total_data)
n_validation_data = total_data - n_training_data
print(f"training data: {n_training_data}, validation data: {n_validation_data}")
rand_perm = torch.randperm(total_data)
train_idx = rand_perm[0:n_training_data]
test_idx = rand_perm[n_training_data:]

graphing_data = {data_name: {} for data_name in ["critic_obs", "values", "returns"]}
graphing_data["critic_obs"]["Rosenbrock"] = x
graphing_data["values"]["Rosenbrock"] = target
graphing_data["returns"]["Rosenbrock"] = target
data = TensorDict(
    {"critic_obs": x.unsqueeze(dim=0), "returns": target.unsqueeze(dim=0)},
    batch_size=(1, total_data),
    device=DEVICE,
)

mean_training_loss = {name: [] for name in experiment_params.keys()}
test_error = {name: [] for name in experiment_params.keys()}

for name, critic in test_critics.items():
    print("")
    # with torch.no_grad():
    #     critic.value_offset.copy_(episode_rollouts.mean())

    critic_optimizer = critic_optimizers[name]

    mean_value_loss = 0
    counter = 0
    generator = create_uniform_generator(
                data[:1, train_idx],
                batch_size,
                max_gradient_steps=max_gradient_steps,
            )
    for batch in generator:
        # print("critic name", name)
        # print("batch['critic_obs'].squeeze() shape", batch["critic_obs"].squeeze().shape)
        # print("batch['returns'].squeeze() shape", batch["returns"].squeeze().shape)
        value_loss = critic.loss_fn(
            batch["critic_obs"].squeeze(), batch["returns"].squeeze()
        )

        critic_optimizer.zero_grad()
        value_loss.backward()
        critic_optimizer.step()
        counter += 1
        with torch.no_grad():
            error = (
                (
                    data["returns"][0, test_idx].squeeze()
                    - critic.evaluate(data["critic_obs"][0, test_idx])
                ).pow(2)
            ).to("cpu")
        mean_training_loss[name].append(value_loss.item())
        test_error[name].append(error.detach().numpy())
    print(f"{name} average error: ", error.mean().item(), "+/- ", error.std().item())
    print(f"{name} max error: ", error.max().item())
    mean_value_loss /= counter


    with torch.no_grad():
        graphing_data["critic_obs"][name] = data[0, :]["critic_obs"]
        graphing_data["values"][name] = critic.evaluate(data[0, :]["critic_obs"])
        graphing_data["returns"][name] = data[0, :]["returns"]

# compare new and old critics
save_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "offline_critics_graph", time_str)
if not os.path.exists(save_path):
    os.makedirs(save_path)

if n_dims == 2:
    plot_rosenbrock_multiple_critics_w_data(
            graphing_data["critic_obs"],
            graphing_data["values"],
            graphing_data["returns"],
            title=f"Learning a {n_dims}D Rosenbrock Function With Different State Represntations \n",
            fn=save_path + f"/state_representation_rosenbrock",
            data=data[0, train_idx]["critic_obs"],
            grid_size=grid_resolution,
        )

critics_no_ground_truth = list(graphing_data["critic_obs"].keys())
critics_no_ground_truth.remove("Rosenbrock")
g_data_no_ground_truth = generate_rosenbrock_g_data_dict(critics_no_ground_truth, [0.001])

for name in list(graphing_data["critic_obs"].keys()):
    if "Rosenbrock" in name:
        continue
    g_data_no_ground_truth[0.001]["critic_obs"][name] = graphing_data["critic_obs"][name]
    g_data_no_ground_truth[0.001]["values"][name] = graphing_data["values"][name]
    g_data_no_ground_truth[0.001]["returns"][name] = graphing_data["returns"][name]
    g_data_no_ground_truth[0.001]["error"][name] = graphing_data["returns"][name].squeeze() - graphing_data["values"][name].squeeze()

plot_binned_errors(g_data_no_ground_truth,
                   save_path + f"/rosenbrock",
                   lb=0,
                   ub=15,
                   step=1,
                   tick_step=2,
                   title_add_on=f"{n_dims}D Rosenbrock Function with Different State Representations")


# plot_pendulum_multiple_critics_w_data(
#     graphing_data["critic_obs"],
#     graphing_data["values"],
#     graphing_data["returns"],
#     title=f"iteration{iteration}",
#     fn=save_path + f"/{len(experiment_params.keys())}_CRITIC_it{iteration}",
#     data=data[:num_steps, traj_idx]["critic_obs"],
# )

plt.close()
# plot_learning_progress(
#     test_error,
#     fn=save_path + f"/{len(experiment_params.keys())}_error_{iteration}",
#     smoothing_window=50,
# )
# plt.show()
this_file = os.path.join(LEGGED_GYM_ROOT_DIR, "scripts", "multiple_offline_critics.py")
params_file = os.path.join(LEGGED_GYM_ROOT_DIR, "scripts", "critic_params.py")
shutil.copy(this_file, os.path.join(save_path, os.path.basename(this_file)))
shutil.copy(params_file, os.path.join(save_path, os.path.basename(params_file)))

