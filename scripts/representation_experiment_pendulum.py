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
)
from gym import LEGGED_GYM_ROOT_DIR
import os
import shutil

import torch
from torch import nn  # noqa F401
from representation_experiment_params import experiment_params


DEVICE = "cuda:0"
for experiment in experiment_params.values():
    experiment["critic_params"]["device"] = DEVICE


# handle some bookkeeping
# run_name = "May15_16-20-21_standard_critic"
run_name = "Jun06_00-51-58_standard_critic"
run_name = "May22_11-11-03_standard_critic"

log_dir = os.path.join(
    LEGGED_GYM_ROOT_DIR, "logs", "pendulum_standard_critic", run_name
)
time_str = time.strftime("%Y%m%d_%H%M%S")


test_critics = {}
for name, experiment in experiment_params.items():
    critic_class = globals()[experiment["critic_name"]]
    params = experiment["critic_params"]
    test_critics[name] = critic_class(**params).to(DEVICE)
critic_optimizers = {
    name: torch.optim.Adam(critic.parameters(), lr=1e-3)
    for name, critic in test_critics.items()
}

gamma = 0.95
lam = 1.0
iteration = 200
max_gradient_steps = 1000
batch_size = 128
num_steps = 50  # ! want this at 1
n_trajs = 256
rand_perm = torch.randperm(4096)
traj_idx = rand_perm[0:n_trajs]
test_idx = rand_perm[n_trajs : n_trajs + 1000]

# last_loss = {name: 0.0 for name in critic_names}
mean_training_loss = {name: [] for name in experiment_params.keys()}
test_error = {name: [] for name in experiment_params.keys()}

# load data
base_data = torch.load(os.path.join(log_dir, "data_{}.pt".format(iteration))).to(DEVICE)

# compute ground-truth
graphing_data = {data_name: {} for data_name in ["critic_obs", "values", "returns"]}

episode_rollouts = compute_MC_returns(base_data, gamma)
print(f"Initializing value offset to: {episode_rollouts.mean().item()}")
graphing_data["critic_obs"]["Ground Truth MC Returns"] = (
    base_data[0, :]["critic_obs"].detach().clone()
)
graphing_data["values"]["Ground Truth MC Returns"] = episode_rollouts[0, :]
graphing_data["returns"]["Ground Truth MC Returns"] = episode_rollouts[0, :]

for name, critic in test_critics.items():
    print("")
    with torch.no_grad():
        critic.value_offset.copy_(episode_rollouts.mean())

    critic_optimizer = critic_optimizers[name]
    data = base_data.detach().clone()
    # train new critic
    data["values"] = critic.evaluate(data["critic_obs"])
    data["advantages"] = compute_generalized_advantages(data, gamma, lam, critic)
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
        critic_optimizer.step()
        counter += 1
        with torch.no_grad():
            error = (
                (
                    episode_rollouts[0, test_idx]
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

plot_pendulum_multiple_critics_w_data(
    graphing_data["critic_obs"],
    graphing_data["values"],
    graphing_data["returns"],
    title=f"iteration{iteration}",
    fn=save_path + f"/{len(experiment_params.keys())}_CRITIC_it{iteration}",
    data=data[:num_steps, traj_idx]["critic_obs"],
)

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

# np.savez(
#     save_path + "/multiple_offline_critics_graphing_data.npz",
#     graphing_data=graphing_data,
#     data=data[:num_steps, traj_idx]["critic_obs"].detach().cpu().numpy(),
#     test_error=test_error,
# )
