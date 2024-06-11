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
from learning.modules.lqrc.plotting import plot_binned_errors
from gym import LEGGED_GYM_ROOT_DIR
import os
import shutil

import torch
from torch import nn  # noqa F401

# from critic_params import critic_params
from critic_params_osc import critic_params


DEVICE = "cuda:0"
for critic_param in critic_params.values():
    critic_param["device"] = DEVICE


# handle some bookkeeping
# run_name = "May15_16-20-21_standard_critic"
# run_name = "Jun06_00-51-58_standard_critic"
# run_name = "May22_11-11-03_standard_critic"
run_name = "Jun08_17-09-48_standard_critic"  # oscillator

# log_dir = os.path.join(
#     LEGGED_GYM_ROOT_DIR, "logs", "pendulum_standard_critic", run_name
# )
log_dir = os.path.join(
    LEGGED_GYM_ROOT_DIR, "logs", "FullSend_standard_critic", run_name
)
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
# Instantiate the critics and add them to test_critics

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

# last_loss = {name: 0.0 for name in critic_names}
# mean_training_loss = {name: [] for name in critic_names}
# test_error = {name: [] for name in critic_names}
learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
graphing_data = {lr: {} for lr in learning_rates}
for lr in learning_rates:
    for iteration in range(iter_offset, tot_iter, iter_step):
        # load data
        # base_data = torch.load(os.path.join(log_dir,
        # "data_{}.pt".format(iteration))).to(
        #     DEVICE
        # )
        base_data = torch.load(os.path.join(log_dir, "data_{}.pt".format(500))).to(
            DEVICE
        )
        # compute ground-truth
        episode_rollouts = compute_MC_returns(base_data, gamma)
        print(f"Initializing value offset to: {episode_rollouts.mean().item()}")

        for name in critic_names:
            print("")
            # if hasattr(test_critics[name], "value_offset"):
            params = critic_params[name]
            if "critic_name" in params.keys():
                params.update(critic_params[params["critic_name"]])

            critic_class = globals()[name]
            critic = critic_class(**params).to(DEVICE)
            critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr)

            with torch.no_grad():
                critic.value_offset.copy_(episode_rollouts.mean())

            data = base_data.detach().clone()
            # train new critic
            data["values"] = critic.evaluate(data["critic_obs"])
            data["advantages"] = compute_generalized_advantages(
                data, gamma, lam, critic
            )
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

                # mean_training_loss[name].append(value_loss.item())
                # test_error[name].append(error.detach().numpy())

            print(f"{name} average error: ", error.mean().item())
            print(f"{name} max error: ", error.max().item())
            mean_value_loss /= counter

            with torch.no_grad():
                actual_error = (
                    critic.evaluate(data["critic_obs"][0]) - episode_rollouts[0]
                ).pow(2)
                # print("rollout average", episode_rollouts[0].mean())
            graphing_data[lr][name] = actual_error


# compare new and old critics
save_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "offline_critics_graph", time_str)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# PLOTS!!
plot_binned_errors(graphing_data, save_path + "/osc_test", title_add_on="Test")

this_file = os.path.join(LEGGED_GYM_ROOT_DIR, "scripts", "lr_osc.py")
params_file = os.path.join(LEGGED_GYM_ROOT_DIR, "scripts", "critic_params_osc.py")
shutil.copy(this_file, os.path.join(save_path, os.path.basename(this_file)))
shutil.copy(params_file, os.path.join(save_path, os.path.basename(params_file)))
