import os
import time

from gym.envs import __init__  # noqa: F401
from gym import LEGGED_GYM_ROOT_DIR
from gym.utils import get_args, task_registry
from learning import LEGGED_GYM_LQRC_DIR
from learning.modules import Critic
from learning.modules.lqrc.utils import get_load_path
from learning.modules.lqrc.plotting import (
    plot_value_func,
    plot_value_func_error,
)
from isaacgym import gymtorch

# torch needs to be imported after isaacgym imports in local source
import torch
import numpy as np


def model_switch(args):
    if args["model_type"] == "CholeskyPlusConst":
        return Critic(2, standard_nn=False).to(DEVICE)
    elif args["model_type"] == "StandardMLP":
        return Critic(2, [128, 64, 32], "tanh", standard_nn=True).to(DEVICE)
    else:
        raise KeyError("Specified model type is not supported for critic evaluation.")


def filter_state_dict(state_dict):
    critic_state_dict = {}
    for key, val in state_dict.items():
        if "critic." in key:
            critic_state_dict[key.replace("critic.", "")] = val.to(DEVICE)
    return critic_state_dict


def setup(args):
    env_cfg, train_cfg = task_registry.create_cfgs(args)
    env_cfg.env.num_envs = steps**2
    train_cfg.runner.num_steps_per_env = round(5.0 / (1.0 - train_cfg.algorithm.gamma))
    env_cfg.env.episode_length_s = 9999
    env_cfg.env.num_projectiles = 20
    task_registry.make_gym_and_sim()
    env = task_registry.make_env(args.task, env_cfg)
    env.cfg.init_state.reset_mode = "reset_to_basic"
    train_cfg.runner.resume = True
    train_cfg.logging.enable_local_saving = False
    runner = task_registry.make_alg_runner(env, train_cfg)

    # * switch to evaluation mode (dropout for example)
    runner.switch_to_eval()
    if EXPORT_POLICY:
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
        )
        runner.export(path)
    return env, runner, train_cfg


def create_logging_dict(runner, n_timesteps, reward_list=None):
    states_to_log = [
        "dof_pos",
        "dof_vel",
        "tau_ff",
    ]
    logs_dict = {}

    for state in states_to_log:
        array_dim = runner.get_obs_size(list([state]))
        logs_dict[state] = torch.zeros(
            (runner.env.num_envs, n_timesteps, array_dim),
            device=runner.env.device,
        )

    if reward_list is not None:
        for reward in reward_list:
            logs_dict["r_" + reward] = torch.zeros(
                (runner.env.num_envs, n_timesteps),
                device=runner.env.device,
            )

    return states_to_log, logs_dict


def get_ground_truth(env, runner, train_cfg, grid):
    env.dof_state[:, 0] = grid[:, 0]
    env.dof_state[:, 1] = grid[:, 1]
    env_ids_int32 = torch.arange(env.num_envs, device=env.device).to(dtype=torch.int32)
    env.gym.set_dof_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.dof_state),
        gymtorch.unwrap_tensor(env_ids_int32),
        len(env_ids_int32),
    )
    rewards_dict = {}
    rewards = np.zeros((runner.num_steps_per_env, env.num_envs))
    reward_list = runner.policy_cfg["reward"]["weights"]
    n_steps = runner.num_steps_per_env
    states_to_log, logs = create_logging_dict(runner, n_steps, reward_list)

    for i in range(runner.num_steps_per_env):
        runner.set_actions(
            runner.policy_cfg["actions"],
            runner.get_inference_actions(),
            runner.policy_cfg["disable_actions"],
        )
        env.step()
        terminated = runner.get_terminated()
        runner.update_rewards(rewards_dict, terminated)
        total_rewards = torch.stack(tuple(rewards_dict.values())).sum(dim=0)
        rewards[i, :] = total_rewards.detach().cpu().numpy()
        env.check_exit()
        for state in states_to_log:
            logs[state][:, i, :] = getattr(env, state)
        for reward in reward_list:
            logs["r_" + reward][:, i] = rewards_dict[reward]

    discount_factors = train_cfg.algorithm.gamma * np.ones(
        (1, runner.num_steps_per_env)
    )
    for i in range(runner.num_steps_per_env):
        discount_factors[0, i] = discount_factors[0, i] ** i
    returns = np.matmul(discount_factors, rewards)
    return returns, logs


def query_value_function(vf_args, grid):
    path = get_load_path(
        vf_args["experiment_name"], vf_args["load_run"], vf_args["checkpoint"]
    )
    model = model_switch(vf_args)

    loaded_dict = torch.load(path)
    critic_state_dict = loaded_dict["critic_state_dict"]
    model.load_state_dict(critic_state_dict)
    predicted_returns = np.zeros((grid.shape[0], 1))
    model.eval()
    for ix, X_batch in enumerate(grid):
        pred = model.evaluate(X_batch.unsqueeze(0))
        predicted_returns[ix, :] = pred.item()

    return predicted_returns


if __name__ == "__main__":
    args = get_args()

    DEVICE = "cuda:0"
    steps = 50
    npy_fn = (
        f"{LEGGED_GYM_LQRC_DIR}/logs/custom_training_data.npy"
        if args.custom_critic
        else f"{LEGGED_GYM_LQRC_DIR}/logs/standard_training_data.npy"
    )
    data = np.load(npy_fn)
    # ! data is normalized
    # dof_pos: 2.0 * torch.pi
    # dof_vel: 5.0
    dof_pos_rng = torch.linspace(-torch.pi, torch.pi, steps=steps, device=DEVICE)
    dof_vel_rng = torch.linspace(-8.0, 8.0, steps=steps, device=DEVICE)
    grid = torch.cartesian_prod(dof_pos_rng, dof_vel_rng)

    EXPORT_POLICY = False
    time_str = time.strftime("%b%d_%H-%M-%S")
    save_path = os.path.join(LEGGED_GYM_LQRC_DIR, f"logs/{time_str}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # get ground truth
    with torch.no_grad():
        env, runner, train_cfg = setup(args)

    ground_truth_returns, logs = get_ground_truth(env, runner, train_cfg, grid)
    logs = {k: v.detach().cpu().numpy() for k, v in logs.items()}
    np.savez_compressed(os.path.join(save_path, "data.npz"), **logs)
    print("saved logs to", os.path.join(save_path, "data.npz"))

    high_gt_returns = []
    for i in range(ground_truth_returns.shape[1] - 1):
        if ground_truth_returns[0, i] > 3.5 and ground_truth_returns[0, i + 1] < 2.0:
            high_gt_returns.append(
                torch.hstack((grid[i, :], grid[i + 1, :])).detach().cpu().numpy()
            )
        if ground_truth_returns[0, i] > 3.5 and ground_truth_returns[0, i + 1] < 2.0:
            high_gt_returns.append(
                torch.hstack((grid[i, :], grid[i + 1, :])).detach().cpu().numpy()
            )
    high_gt_returns = np.array(high_gt_returns)
    returns_save_path = (
        f"{LEGGED_GYM_LQRC_DIR}/logs/custom_high_returns.npy"
        if args.custom_critic
        else f"{LEGGED_GYM_LQRC_DIR}/logs/standard_high_returns.npy"
    )
    returns_save_path = (
        f"{LEGGED_GYM_LQRC_DIR}/logs/custom_high_returns.npy"
        if args.custom_critic
        else f"{LEGGED_GYM_LQRC_DIR}/logs/standard_high_returns.npy"
    )
    np.save(returns_save_path, high_gt_returns)
    print("Saved high returns to", returns_save_path)

    # get NN value functions
    custom_vf_args = {
        "experiment_name": "pendulum_custom_critic",
        "load_run": "Feb22_14-11-04_custom_critic",
        "checkpoint": -1,
        "model_type": "CholeskyPlusConst",
    }
    custom_critic_returns = query_value_function(custom_vf_args, grid)
    standard_vf_args = {
        "experiment_name": "pendulum_standard_critic",
        "load_run": "Feb22_14-07-29_standard_critic",
        "checkpoint": -1,
        "model_type": "StandardMLP",
    }
    standard_critic_returns = query_value_function(standard_vf_args, grid)

    plot_value_func_error(
        grid.detach().cpu().numpy(),
        custom_critic_returns - ground_truth_returns.T,
        standard_critic_returns - ground_truth_returns.T,
        ground_truth_returns.T,
        save_path + f"/value_func_error_{steps}_steps.png",
        contour=False,
    )

    plot_value_func(
        grid.detach().cpu().numpy(),
        custom_critic_returns,
        standard_critic_returns,
        ground_truth_returns.T,
        save_path + f"/value_func_{steps}_steps.png",
        contour=False,
    )

    # ! store data dist with model logs to ensure they're paired properly
    # plot_training_data_dist(npy_fn,
    #                         save_path + "/data_distribution.png")
