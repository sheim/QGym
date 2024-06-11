import time
import matplotlib.pyplot as plt  # noqa F401
import numpy as np  # noqa F401
from learning.modules.critic import Critic  # noqa F401
from learning.modules.lqrc import *  # noqa F401
from learning.utils import (
    compute_generalized_advantages,
    compute_MC_returns,
    create_uniform_generator,
)
from learning.modules.lqrc.plotting import (
    plot_pendulum_multiple_critics_w_data,
    plot_learning_progress,
    plot_binned_errors
)
from gym import LEGGED_GYM_ROOT_DIR
import os
import shutil

import torch
from torch import nn  # noqa F401
import torch.nn.functional as F
# from critic_params import critic_params
from critic_params_rosenbrock import critic_params
from torch.utils.data import DataLoader, TensorDataset

def generate_rosenbrock(n, lb, ub, steps):
    """
    Generates data based on Rosenbrock function
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """
    assert n > 1, "n must be > 1 for Rosenbrock"
    all_linspaces = [torch.linspace(lb, ub, steps, device=DEVICE) for i in range(n)]
    X = torch.cartesian_prod(*all_linspaces)
    term_1 = 100 * torch.square(X[:, 1:] - torch.square(X[:, :-1]))
    term_2 = torch.square(1 - X[:, :-1])
    y = torch.sum(term_1 + term_2, axis=1)
    return X, y.unsqueeze(1)

def generate_bounded_rosenbrock(n, lb, ub, steps):
    """
    Generates data based on Rosenbrock function
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """

    def bound_function_smoothly(y):
        a = 50.0  # Threshold
        c = 60.0  # Constant to transition to
        k = 0.1  # Sharpness of transition
        return y * (1 - 1 / (1 + torch.exp(-k * (y - a)))) + c * (
            1 / (1 + torch.exp(-k * (y - a)))
        )

    X, y = generate_rosenbrock(n, lb, ub, steps)
    y = bound_function_smoothly(y)
    return X, y

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
tot_iter = 500
iter_offset = 0
iter_step = 2
max_gradient_steps = 1000
# max_grad_norm = 1.0
batch_size = 128
num_steps = 10  # ! want this at 1
n_trajs = 3096
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
        # base_data = torch.load(os.path.join(log_dir, "data_{}.pt".format(500))).to(
        #     DEVICE
        # )
        torch.cuda.empty_cache()
        input, target = generate_bounded_rosenbrock(2, 0.0, 2.0, 64)
        # print(input.shape)
        # print(target[traj_idx].shape)
        # exit()
        # compute ground-truth
        # episode_rollouts = compute_MC_returns(base_data, gamma)
        print(f"Initializing value offset to: {target.mean().item()}")

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
                critic.value_offset.copy_(target.mean())

            data = target.detach().clone()
            # train new critic
            mean_value_loss = 0
            counter = 0
            # generator = create_uniform_generator(
            #     data[traj_idx],
            #     batch_size,
            #     max_gradient_steps=max_gradient_steps,
            # )
            
            training_data = TensorDataset(input[traj_idx], target[traj_idx])
            dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
            # exit()
            for X_batch, y_batch in dataloader:
                pred = critic.evaluate(X_batch)
                value_loss = F.mse_loss(pred, y_batch, reduction="mean")
                critic_optimizer.zero_grad()
                value_loss.backward()
                # noqa F401.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
                critic_optimizer.step()
                counter += 1

            with torch.no_grad():
                actual_error = (
                    critic.evaluate(input[test_idx]) - target[test_idx].squeeze()
                ).pow(2)
                # print("rollout average", episode_rollouts[0].mean())
            # print(actual_error.shape)
            # print(critic.evaluate(input[test_idx]).shape)
            # print(target[test_idx].shape)
            # exit()
            graphing_data[lr][name] = actual_error
            print(f"{name} average error: ", actual_error.mean().item())
            print(f"{name} max error: ", actual_error.max().item())


# compare new and old critics
save_path = os.path.join(
    LEGGED_GYM_ROOT_DIR, "logs", "offline_critics_graph", time_str
)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# PLOTS!!
plot_binned_errors(graphing_data,
                   save_path + f"/rosenbrock",
                   title_add_on=f"Rosenbrock")

this_file = os.path.join(LEGGED_GYM_ROOT_DIR, "scripts", "lr_rosenbrock.py")
params_file = os.path.join(LEGGED_GYM_ROOT_DIR, "scripts", "critic_params_osc.py")
shutil.copy(this_file, os.path.join(save_path, os.path.basename(this_file)))
shutil.copy(params_file, os.path.join(save_path, os.path.basename(params_file)))
