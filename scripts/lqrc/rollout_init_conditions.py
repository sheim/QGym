import numpy as np
from gym.envs import __init__  # noqa: F401
from gym.utils import get_args, task_registry
from gym.utils import KeyboardInterface
from gym.utils import VisualizationRecorder
from isaacgym import gymtorch

# torch needs to be imported after isaacgym imports in local source
import torch

from learning import LEGGED_GYM_LQRC_DIR
from learning.modules.lqrc.plotting import plot_theta_omega_polar, plot_trajectories


def setup(args, num_init_conditions):
    env_cfg, train_cfg = task_registry.create_cfgs(args)
    env_cfg.env.num_envs = num_init_conditions  # min(env_cfg.env.num_envs, 16)
    if hasattr(env_cfg, "push_robots"):
        env_cfg.push_robots.toggle = False
    if hasattr(env_cfg, "commands"):
        env_cfg.commands.resampling_time = 9999
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
    return env, runner, train_cfg


def play(env, runner, train_cfg, init_conditions):
    num_env_steps = 2000  # round(10.0 / (1.0 - 0.99))
    pos_traj = np.zeros((num_env_steps, env.num_envs))
    vel_traj = np.zeros((num_env_steps, env.num_envs))
    torques = np.zeros((num_env_steps, env.num_envs))
    rewards = np.zeros((num_env_steps, env.num_envs))
    rewards_dict = {}

    env.dof_state[:, 0] = torch.from_numpy(np.hstack((init_conditions[0, 0],
                                        init_conditions[0, 2]))).to(DEVICE)
    env.dof_state[:, 1] = torch.from_numpy(np.hstack((init_conditions[0, 1],
                                        init_conditions[0, 3]))).to(DEVICE)

    env_ids_int32 = torch.arange(env.num_envs, device=env.device).to(dtype=torch.int32)
    env.gym.set_dof_state_tensor_indexed(
            env.sim,
            gymtorch.unwrap_tensor(env.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
    )

    # * set up recording
    if env.cfg.viewer.record:
        recorder = VisualizationRecorder(
            env, train_cfg.runner.experiment_name, train_cfg.runner.load_run
        )

    for i in range(num_env_steps):
        if env.cfg.viewer.record:
            recorder.update(i)
        runner.set_actions(
            runner.policy_cfg["actions"],
            runner.get_inference_actions(),
            runner.policy_cfg["disable_actions"],
        )
        env.step()
        pos_traj[i, :] = env.dof_pos.detach().cpu().numpy().squeeze()
        vel_traj[i, :] = env.dof_vel.detach().cpu().numpy().squeeze()
        torques[i, :] = env.tau_ff.detach().cpu().numpy().squeeze()
        terminated = runner.get_terminated()
        runner.update_rewards(rewards_dict, terminated)
        total_rewards = torch.stack(tuple(rewards_dict.values())).sum(dim=0)
        rewards[i, :] = total_rewards.detach().cpu().numpy()
        env.check_exit()

    if env.cfg.viewer.record:
        recorder.save()

    return pos_traj, vel_traj, torques, rewards


if __name__ == "__main__":
    args = get_args()
    DEVICE = "cuda:0"
    npy_fn = f"{LEGGED_GYM_LQRC_DIR}/logs/custom_high_returns.npy" if args.custom_critic else f"{LEGGED_GYM_LQRC_DIR}/logs/standard_high_returns.npy"
    init_conditions = np.array([[0.25, -0.75, 0.2, -0.7]])  # np.load(npy_fn)
    num_init_conditions = 2
    with torch.no_grad():
        env, runner, train_cfg = setup(args, num_init_conditions)
        pos_traj, vel_traj, torques, rewards = play(env, runner, train_cfg, init_conditions)
    plot_trajectories(pos_traj, vel_traj, torques, rewards, f"{LEGGED_GYM_LQRC_DIR}/logs/trajectories.png", title=f"High Return Init (theta, omega) {np.hstack((init_conditions[0, 0], init_conditions[0, 2]))}, Low Return Init {np.hstack((init_conditions[0, 1], init_conditions[0, 3]))}")
    # plot_theta_omega_polar(np.vstack((init_conditions[:, 0], init_conditions[:, 2])), np.vstack((init_conditions[:, 1], init_conditions[:, 3])), npy_fn[:-4] + ".png")
    # print(init_conditions[0, :])
