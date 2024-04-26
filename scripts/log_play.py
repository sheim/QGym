from gym.envs import __init__  # noqa: F401
from gym.utils import get_args, task_registry
from gym.utils import KeyboardInterface
from gym.utils import VisualizationRecorder
from gym import LEGGED_GYM_ROOT_DIR

# torch needs to be imported after isaacgym imports in local source
import torch
import os
import numpy as np

TEST_TOTAL_TIMESTEPS = 1000


def create_logging_dict(runner, test_total_timesteps):
    states_to_log = [
        "dof_pos_target",
        "dof_pos_obs",
        "dof_vel",
        "torques",
        "commands",
        # 'base_lin_vel',
        # 'base_ang_vel',
        # 'oscillators',
        # 'grf',
        # 'base_height'
    ]

    states_to_log_dict = {}

    for state in states_to_log:
        array_dim = runner.get_obs_size(
            [
                state,
            ]
        )
        states_to_log_dict[state] = torch.zeros(
            (1, test_total_timesteps, array_dim), device=runner.env.device
        )
    return states_to_log_dict


def setup(args):
    env_cfg, train_cfg = task_registry.create_cfgs(args)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 16)
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


def play(env, runner, train_cfg):
    protocol_name = train_cfg.runner.experiment_name

    # * set up logging
    log_file_path = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        "gym",
        "smooth_exploration",
        "data_play",
        protocol_name + ".npz",
    )
    if not os.path.exists(os.path.dirname(log_file_path)):
        os.makedirs(os.path.dirname(log_file_path))

    states_to_log_dict = create_logging_dict(runner, TEST_TOTAL_TIMESTEPS)

    # * set up recording
    if env.cfg.viewer.record:
        recorder = VisualizationRecorder(
            env, train_cfg.runner.experiment_name, train_cfg.runner.load_run
        )

    # * set up interface: GamepadInterface(env) or KeyboardInterface(env)
    COMMANDS_INTERFACE = hasattr(env, "commands")
    if COMMANDS_INTERFACE:
        # interface = GamepadInterface(env)
        interface = KeyboardInterface(env)

    for t in range(TEST_TOTAL_TIMESTEPS):
        if COMMANDS_INTERFACE:
            interface.update(env)
        if env.cfg.viewer.record:
            recorder.update(t)
        runner.set_actions(
            runner.policy_cfg["actions"],
            runner.get_inference_actions(),
            runner.policy_cfg["disable_actions"],
        )
        env.step()
        env.check_exit()

        # * log
        for state in states_to_log_dict:
            states_to_log_dict[state][:, t, :] = getattr(env, state)[0, :]

    # * save data
    # first convert tensors to cpu
    log_dict_cpu = {k: v.cpu() for k, v in states_to_log_dict.items()}
    np.savez_compressed(log_file_path, **log_dict_cpu)
    print("saved to ", log_file_path)
    return states_to_log_dict


if __name__ == "__main__":
    args = get_args()
    with torch.no_grad():
        env, runner, train_cfg = setup(args)
        play(env, runner, train_cfg)
