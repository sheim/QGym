import numpy as np
from gym.envs import __init__  # noqa: F401
from gym.utils import get_args, task_registry
from gym.utils import KeyboardInterface
from gym.utils import VisualizationRecorder

# torch needs to be imported after isaacgym imports in local source
import torch

from learning import LEGGED_GYM_LQRC_DIR
from learning.modules.lqrc.plotting import plot_theta_omega_polar


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
    env.dof_pos = torch.from_numpy(init_conditions[:, 0]).to(DEVICE).unsqueeze(1)
    env.dof_vel = torch.from_numpy(init_conditions[:, 1]).to(DEVICE).unsqueeze(1)
    # * set up recording
    if env.cfg.viewer.record:
        recorder = VisualizationRecorder(
            env, train_cfg.runner.experiment_name, train_cfg.runner.load_run
        )

    for i in range(round(2.0 / (1.0 - 0.99))):
        if env.cfg.viewer.record:
            recorder.update(i)
        runner.set_actions(
            runner.policy_cfg["actions"],
            runner.get_inference_actions(),
            runner.policy_cfg["disable_actions"],
        )
        env.step()
        env.check_exit()


if __name__ == "__main__":
    args = get_args()
    DEVICE = "cuda:0"
    npy_fn = f"{LEGGED_GYM_LQRC_DIR}/logs/custom_high_returns.npy" if args.custom_critic else f"{LEGGED_GYM_LQRC_DIR}/logs/standard_high_returns.npy"
    init_conditions = np.load(npy_fn)
    num_init_conditions = init_conditions.shape[0]
    with torch.no_grad():
        env, runner, train_cfg = setup(args, num_init_conditions)
        play(env, runner, train_cfg, init_conditions)
    plot_theta_omega_polar(init_conditions[:, 0], init_conditions[:, 1], npy_fn[:-4] + ".png")
