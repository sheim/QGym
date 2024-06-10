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
    plot_dim_sweep,
)
from gym import LEGGED_GYM_ROOT_DIR
import os
import shutil

import torch
from torch import nn  # noqa F401
# from critic_params import critic_params
from critic_params_osc import critic_params
from optimizer_params import optimizer_params


DEVICE = "cuda:0"
for critic_param in critic_params.values():
    critic_param["device"] = DEVICE


# handle some bookkeeping
# run_name = "May15_16-20-21_standard_critic"
# run_name = "Jun06_00-51-58_standard_critic"
# run_name = "May22_11-11-03_standard_critic"
run_name = "Jun08_17-09-48_standard_critic" # oscillator

# log_dir = os.path.join(
#     LEGGED_GYM_ROOT_DIR, "logs", "pendulum_standard_critic", run_name
# )
log_dir = os.path.join(
    LEGGED_GYM_ROOT_DIR, "logs", "FullSend_standard_critic", run_name
)
time_str = time.strftime("%Y%m%d_%H%M%S")

save_path = os.path.join(
    LEGGED_GYM_ROOT_DIR, "logs", "offline_critics_graph", time_str
)
if not os.path.exists(save_path):
    os.makedirs(save_path)


gamma = 0.95
lam = 1.0
tot_iter = 200
iter_offset = 199
iter_step = 2
max_gradient_steps = 1000
# max_grad_norm = 1.0
batch_size = 128
num_steps = 10  # ! want this at 1
n_trajs = 256
rand_perm = torch.randperm(4096)
traj_idx = rand_perm[0:n_trajs]
test_idx = rand_perm[n_trajs : n_trajs + 1000]


# mean_training_loss = []
# test_error = []
# size = 4
# rel_range = torch.arange(2, 6)
# latent_range = torch.arange(0, size)
# relative_dim = torch.zeros(size, size)
# latent_dim = torch.zeros(size, size)
# for i, rel_dim in enumerate(rel_range):
#     for j, latent_dim_offset in enumerate(latent_range):
#         relative_dim[i, j] = rel_dim
#         latent_dim[i, j] = rel_dim + latent_dim_offset
# graphing_data = torch.zeros(size, size, 2)
size = 54
# x = torch.arange(1, size + 1)
# y = torch.arange(1, size + 1)
# xx, yy = torch.meshgrid(x, y, indexing="xy")
# graphing_data = torch.zeros(size, size, 2) - float("inf")
x = np.arange(1, size + 1)
y = np.arange(1, size + 1)
xx, yy = np.meshgrid(x, y, indexing="xy")
graphing_data = np.zeros((size, size, 2)) - float("inf")

for i in range(size):
    for j in range(size):
        np.savez(save_path + "/graphing_data.npz",
                 relative_dim=xx,
                 latent_dim=yy,
                 mean=graphing_data[..., 0],
                 max=graphing_data[..., 1])
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
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=optimizer_params[name]["lr"])

        for iteration in range(iter_offset, tot_iter, iter_step):
            # load data
            # base_data = torch.load(os.path.join(log_dir, "data_{}.pt".format(iteration))).to(
            #     DEVICE
            # )
            base_data = torch.load(os.path.join(log_dir, "data_{}.pt".format(500))).to(
                DEVICE
            )
            # compute ground-truth
            episode_rollouts = compute_MC_returns(base_data, gamma)
            print(f"Initializing value offset to: {episode_rollouts.mean().item()}")

            print("")
            # if hasattr(test_critics[name], "value_offset"):
            with torch.no_grad():
                critic.value_offset.copy_(episode_rollouts.mean())

            data = base_data.detach().clone()
            # train new critic
            data["values"] = critic.evaluate(data["critic_obs"])
            try:
                data["advantages"] = compute_generalized_advantages(data, gamma, lam, critic)
            except Exception as e:
                print("relative dim", rel_dim, "latent_dim", lat_dim)
            data["returns"] = data["advantages"] + data["values"]

            mean_value_loss = 0
            counter = 0
            generator = create_uniform_generator(
                data[:num_steps, traj_idx],
                batch_size,
                max_gradient_steps=max_gradient_steps,
            )
            for batch in generator:
                value_loss = critic.loss_fn(
                    batch["critic_obs"], batch["returns"], actions=batch["actions"]
                )
                critic_optimizer.zero_grad()
                value_loss.backward()
                # noqa F401.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
                critic_optimizer.step()
                counter += 1
                with torch.no_grad():
                    error = (
                        (
                            episode_rollouts[0, test_idx]
                            - critic.evaluate(data["critic_obs"][0, test_idx])
                        ).pow(2)
                    ).to("cpu")
                # mean_training_loss.append(value_loss.item())
                # test_error.append(error.detach().numpy())
            print(f"{name} average error: ", error.mean().item())
            print(f"{name} max error: ", error.max().item())
            mean_value_loss /= counter
            episode_rollouts = compute_MC_returns(data, gamma)
            with torch.no_grad():
                actual_error = (
                    critic.evaluate(data["critic_obs"][0]) - episode_rollouts[0]
                ).pow(2)
            mean_actual_error = actual_error.mean().item()
            max_actual_error = actual_error.max().item()
            graphing_data[i, j, 0] = mean_actual_error
            graphing_data[i, j, 1] = max_actual_error

        # plt.close()
        # plot_learning_progress(
        #     test_error,
        #     fn=save_path + f"/dim_sweep_error_{iteration}",
        #     smoothing_window=50,
        # )
# plot
print(graphing_data[0, 0]) # check that this is tensor([-inf, -inf])


plot_dim_sweep(
    xx,
    yy,
    graphing_data[..., 0],
    graphing_data[..., 1],
    fn=save_path + "/latent_relative_dim_sweep"
)


this_file = os.path.join(LEGGED_GYM_ROOT_DIR, "scripts", "multiple_offline_critics.py")
params_file = os.path.join(LEGGED_GYM_ROOT_DIR, "scripts", "critic_params.py")
shutil.copy(this_file, os.path.join(save_path, os.path.basename(this_file)))
shutil.copy(params_file, os.path.join(save_path, os.path.basename(params_file)))
