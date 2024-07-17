import os
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt

from learning.modules.actor import Actor
from learning.modules.critic import Critic

from learning.utils import (
    compute_generalized_advantages,
    compute_MC_returns,
)
from learning.modules.lqrc.plotting import plot_pendulum_critics_with_data
from gym import LEGGED_GYM_ROOT_DIR

DEVICE = "cpu"

# * Setup
LOAD_RUN = "Jul17_17-22-37_IPG_nu1"
TITLE = "IPG nu=1.0"
IT_RANGE = range(20, 101, 20)

RUN_DIR = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "pendulum", LOAD_RUN)
PLOT_DIR = os.path.join(RUN_DIR, "visualize_critic")
os.makedirs(PLOT_DIR, exist_ok=True)

# * V-Critic, Q-Critic and Actor
critic_q = Critic(
    num_obs=4, hidden_dims=[128, 64, 32], activation="tanh", normalize_obs=False
).to(DEVICE)
critic_v = Critic(
    num_obs=3, hidden_dims=[128, 64, 32], activation="tanh", normalize_obs=False
).to(DEVICE)
actor = Actor(
    num_obs=3,
    num_actions=1,
    hidden_dims=[128, 64, 32],
    activation="tanh",
    normalize_obs=False,
).to(DEVICE)

# * Params
n_envs = 4096  # that were trained with
gamma = 0.95
lam = 1.0
episode_steps = 100
visualize_steps = 10  # just to show rollouts
n_trajs = 64
rand_perm = torch.randperm(n_envs)
traj_idx = rand_perm[0:n_trajs]
test_idx = rand_perm[n_trajs : n_trajs + 1000]

for it in IT_RANGE:
    # load data
    base_data = torch.load(os.path.join(RUN_DIR, "data_onpol{}.pt".format(it))).to(
        DEVICE
    )
    data = base_data.detach().clone()

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

    # load models
    model = torch.load(os.path.join(RUN_DIR, "model_{}.pt".format(it)))
    critic_v.load_state_dict(model["critic_v_state_dict"])
    critic_q.load_state_dict(model["critic_q_state_dict"])
    actor.load_state_dict(model["actor_state_dict"])

    # V-critic values and returns
    data["values"] = critic_v.evaluate(data["critic_obs"])
    data["advantages"] = compute_generalized_advantages(data, gamma, lam, critic_v)
    data["returns"] = data["advantages"] + data["values"]

    with torch.no_grad():
        graphing_data["obs"]["V-Critic"] = graphing_obs[0, :]
        graphing_data["values"]["V-Critic"] = critic_v.evaluate(
            data[0, :]["critic_obs"]
        )
        graphing_data["returns"]["V-Critic"] = data[0, :]["returns"]
        graphing_data["actions"] = actor(data[0, :]["actor_obs"])

    # Q-critic values and returns
    actions = actor(data["actor_obs"])
    critic_q_obs = torch.cat((data["critic_obs"], actions), dim=2)
    data["values"] = critic_q.evaluate(critic_q_obs)
    data["next_critic_obs"] = critic_q_obs  # needed for GAE
    data["advantages"] = compute_generalized_advantages(data, gamma, lam, critic_q)
    data["returns"] = data["advantages"] + data["values"]

    with torch.no_grad():
        graphing_data["obs"]["Q-Critic"] = graphing_obs[0, :]
        graphing_data["values"]["Q-Critic"] = critic_q.evaluate(critic_q_obs[0, :])
        graphing_data["returns"]["Q-Critic"] = data[0, :]["returns"]

    # generate plots
    grid_size = int(np.sqrt(n_envs))
    plot_pendulum_critics_with_data(
        x=graphing_data["obs"],
        predictions=graphing_data["values"],
        targets=graphing_data["returns"],
        actions=graphing_data["actions"],
        title=f"{TITLE} Iteration {it}",
        fn=PLOT_DIR + f"/IPG_it{it}",
        data=graphing_obs[:visualize_steps, traj_idx],
        grid_size=grid_size,
    )

    plt.close()

this_file = os.path.join(LEGGED_GYM_ROOT_DIR, "scripts", "visualize_ipg.py")
shutil.copy(this_file, os.path.join(PLOT_DIR, os.path.basename(this_file)))
