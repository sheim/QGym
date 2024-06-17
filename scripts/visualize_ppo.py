import matplotlib.pyplot as plt  # noqa F401
from learning.modules.critic import Critic  # noqa F401

from learning.utils import (
    compute_generalized_advantages,
    compute_MC_returns,
)
from learning.modules.lqrc.plotting import plot_pendulum_multiple_critics_w_data
from gym import LEGGED_GYM_ROOT_DIR
import os
import shutil

import torch

# DEVICE = "cuda:0"
DEVICE = "cpu"

# * Setup
experiment_name = "pendulum"
run_name = "obs_no_norm"

log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", experiment_name, run_name)
plot_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "V_plots", run_name)
os.makedirs(plot_dir, exist_ok=True)

# * Critic Params
name = "PPO"
n_obs = 3  # [dof_pos_obs, dof_vel]
hidden_dims = [128, 64, 32]
activation = "tanh"
normalize_obs = False
n_envs = 4096

# * Params
gamma = 0.95
lam = 1.0
num_steps = 10  # ! want this at 1
n_trajs = 64
rand_perm = torch.randperm(n_envs)
traj_idx = rand_perm[0:n_trajs]
test_idx = rand_perm[n_trajs : n_trajs + 1000]

it_delta = 20
it_total = 100
it_range = range(it_delta, it_total + 1, it_delta)

for it in it_range:
    # load data
    base_data = torch.load(os.path.join(log_dir, "data_{}.pt".format(it))).to(DEVICE)

    dof_pos = base_data["dof_pos"].detach().clone()
    dof_vel = base_data["dof_vel"].detach().clone()
    graphing_obs = torch.cat((dof_pos, dof_vel), dim=2)

    # compute ground-truth
    graphing_data = {data_name: {} for data_name in ["obs", "values", "returns"]}

    episode_rollouts = compute_MC_returns(base_data, gamma)
    print(f"Initializing value offset to: {episode_rollouts.mean().item()}")
    graphing_data["obs"]["Ground Truth MC Returns"] = graphing_obs[0, :]
    graphing_data["values"]["Ground Truth MC Returns"] = episode_rollouts[0, :]
    graphing_data["returns"]["Ground Truth MC Returns"] = episode_rollouts[0, :]

    # load critic
    model = torch.load(os.path.join(log_dir, "model_{}.pt".format(it)))
    critic = Critic(n_obs, hidden_dims, activation, normalize_obs).to(DEVICE)
    critic.load_state_dict(model["critic_state_dict"])

    # compute values and returns
    data = base_data.detach().clone()
    data["values"] = critic.evaluate(data["critic_obs"])
    data["advantages"] = compute_generalized_advantages(data, gamma, lam, critic)
    data["returns"] = data["advantages"] + data["values"]

    with torch.no_grad():
        graphing_data["obs"][name] = graphing_obs[0, :]
        graphing_data["values"][name] = critic.evaluate(data[0, :]["critic_obs"])
        graphing_data["returns"][name] = data[0, :]["returns"]

    # generate plots
    plot_pendulum_multiple_critics_w_data(
        graphing_data["obs"],
        graphing_data["values"],
        graphing_data["returns"],
        title=f"iteration{it}",
        fn=plot_dir + f"/PPO_CRITIC_it{it}",
        data=graphing_obs[:num_steps, traj_idx],
    )

    plt.close()

this_file = os.path.join(LEGGED_GYM_ROOT_DIR, "scripts", "visualize_ppo.py")
shutil.copy(this_file, os.path.join(plot_dir, os.path.basename(this_file)))
