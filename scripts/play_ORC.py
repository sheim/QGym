import os

from gym import LEGGED_GYM_ROOT_DIR
from gym.utils import get_args, task_registry
from gym.utils import KeyboardInterface

# torch needs to be imported after isaacgym imports in local source
import torch
import numpy as np


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
    train_cfg.runner.resume = False
    train_cfg.logging.enable_local_saving = False
    runner = task_registry.make_alg_runner(env, train_cfg)

    # * switch to evaluation mode (dropout for example)
    runner.switch_to_eval()
    return env, runner, train_cfg


def play(env, runner, train_cfg):
    saveLogs = False
    log = {
        "dof_pos_obs": [],
        "dof_vel": [],
        "torques": [],
        "grf": [],
        "oscillators": [],
        "base_lin_vel": [],
        "base_ang_vel": [],
        "commands": [],
        "dof_pos_error": [],
        "reward": [],
        "dof_names": [],
    }
    RECORD_FRAMES = False
    print(env.dof_names)

    # * set up interface: GamepadInterface(env) or KeyboardInterface(env)
    COMMANDS_INTERFACE = hasattr(env, "commands")
    if COMMANDS_INTERFACE:
        # interface = GamepadInterface(env)
        interface = KeyboardInterface(env)
    img_idx = 0

    for i in range(10 * int(env.max_episode_length)):
        if RECORD_FRAMES:
            if i % 5:
                filename = os.path.join(
                    LEGGED_GYM_ROOT_DIR,
                    "gym",
                    "scripts",
                    "gifs",
                    train_cfg.runner.experiment_name,
                    f"{img_idx}.png",
                )
                # print(filename)
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1

        # print(env.num_envs)
        # print(env.torques.size())
        log["dof_pos_obs"] += env.dof_pos_obs.tolist()
        log["dof_vel"] += env.dof_vel.tolist()
        log["torques"] += env.torques.tolist()
        log["grf"] += env.grf.tolist()
        log["oscillators"] += env.oscillators.tolist()
        log["base_lin_vel"] += env.base_lin_vel.tolist()
        log["base_ang_vel"] += env.base_ang_vel.tolist()
        log["commands"] += env.commands.tolist()
        log["dof_pos_error"] += (env.default_dof_pos - env.dof_pos).tolist()

        reward_weights = runner.policy_cfg["reward"]["weights"]
        log["reward"] += list(runner.get_rewards(reward_weights).items())

        # print(i)
        if i == 1000 and saveLogs:
            log["dof_names"] = env.dof_names
            np.savez("new_logs", **log)

        if COMMANDS_INTERFACE:
            interface.update(env)
        runner.set_actions(
            runner.policy_cfg["actions"],
            runner.get_inference_actions(),
            runner.policy_cfg["disable_actions"],
        )
        env.step()
        env.check_exit()


if __name__ == "__main__":
    EXPORT_POLICY = True
    args = get_args()
    with torch.no_grad():
        env, runner, train_cfg = setup(args)
        play(env, runner, train_cfg)
