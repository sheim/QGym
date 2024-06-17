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
import numpy as np
import torch

# DEVICE = "cuda:0"
DEVICE = "cpu"

# * Setup
experiment_name = "sac_pendulum"
run_name = "1024envs"

log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", experiment_name, run_name)
plot_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "V_plots_sac", run_name)
os.makedirs(plot_dir, exist_ok=True)

# * Critic Params
name = "SAC"
n_obs = 4  # [dof_pos_obs, dof_vel, action]
hidden_dims = [128, 64, 32]
activation = "elu"
normalize_obs = False
n_envs = 1024

# * Params
gamma = 0.95
lam = 1.0
episode_steps = 100
visualize_steps = 10  # just to show rollouts
n_trajs = 64
rand_perm = torch.randperm(n_envs)
traj_idx = rand_perm[0:n_trajs]
test_idx = rand_perm[n_trajs : n_trajs + 1000]

it_delta = 2000
it_total = 30_000
it_range = range(it_delta, it_total + 1, it_delta)

for it in it_range:
    # load data
    base_data = torch.load(os.path.join(log_dir, "data_{}.pt".format(it))).to(DEVICE)
    # TODO: handle buffer differently?
    base_data = base_data[-episode_steps:, :]

    dof_pos = base_data["dof_pos"].detach().clone()
    dof_vel = base_data["dof_vel"].detach().clone()
    graphing_obs = torch.cat((dof_pos, dof_vel), dim=2)

    # compute ground-truth
    graphing_data = {
        data_name: {} for data_name in ["obs", "values", "returns", "actions"]
    }

    episode_rollouts = compute_MC_returns(base_data, gamma)
    print(f"Initializing value offset to: {episode_rollouts.mean().item()}")
    graphing_data["obs"]["Ground Truth MC Returns"] = graphing_obs[0, :]
    graphing_data["values"]["Ground Truth MC Returns"] = episode_rollouts[0, :]
    graphing_data["returns"]["Ground Truth MC Returns"] = episode_rollouts[0, :]

    # load model which includes both critics
    model = torch.load(os.path.join(log_dir, "model_{}.pt".format(it)))

    # line search to find best action
    data = base_data.detach().clone()
    data_shape = data["critic_obs"].shape  # [a, b, 3]

    # create a tensor of actions to evaluate
    N = 41
    actions_space = torch.linspace(-2, 2, N).to(DEVICE)
    actions = (
        actions_space.unsqueeze(0).unsqueeze(1).unsqueeze(-1)
    )  # Shape: [1, 1, N, 1]

    # repeat the actions for each entry in data
    actions = actions.repeat(data_shape[0], data_shape[1], 1, 1)  # Shape: [a, b, N, 1]

    # repeat the data for each action
    critic_obs = (
        data["critic_obs"].unsqueeze(2).repeat(1, 1, N, 1)
    )  # Shape: [a, b, N, 3]

    # concatenate the actions to the data
    critic_obs = torch.cat((critic_obs, actions), dim=3)  # Shape: [a, b, N, 4]

    # evaluate the critic for all actions and entries
    for critic_str in ["critic_1", "critic_2"]:
        critic_name = name + " " + critic_str
        critic = Critic(n_obs, hidden_dims, activation, normalize_obs).to(DEVICE)
        critic.load_state_dict(model[critic_str + "_state_dict"])
        q_values = critic.evaluate(critic_obs)  # Shape: [a, b, N]

        # find the best action for each entry
        best_actions_idx = torch.argmax(q_values, dim=2)  # Shape: [a, b]
        best_actions = actions_space[best_actions_idx]  # Shape: [a, b]
        best_actions = best_actions.unsqueeze(-1)  # Shape: [a, b, 1]

        # compute values and returns
        best_obs = torch.cat(
            (data["critic_obs"], best_actions), dim=2
        )  # Shape: [a, b, 4]
        data["values"] = critic.evaluate(best_obs)  # Shape: [a, b]
        data["next_critic_obs"] = best_obs  # needed for GAE
        data["advantages"] = compute_generalized_advantages(data, gamma, lam, critic)
        data["returns"] = data["advantages"] + data["values"]

        with torch.no_grad():
            graphing_data["obs"][critic_name] = graphing_obs[0, :]
            graphing_data["values"][critic_name] = critic.evaluate(best_obs[0, :])
            graphing_data["returns"][critic_name] = data[0, :]["returns"]
            graphing_data["actions"][critic_name] = best_actions[0, :]

    # generate plots
    grid_size = int(np.sqrt(n_envs))
    plot_pendulum_multiple_critics_w_data(
        graphing_data["obs"],
        graphing_data["values"],
        graphing_data["returns"],
        title=f"iteration{it}",
        fn=plot_dir + f"/{name}_CRITIC_it{it}",
        data=graphing_obs[:visualize_steps, traj_idx],
        grid_size=grid_size,
        actions=graphing_data["actions"],
    )

    plt.close()

this_file = os.path.join(LEGGED_GYM_ROOT_DIR, "scripts", "visualize_sac.py")
shutil.copy(this_file, os.path.join(plot_dir, os.path.basename(this_file)))
