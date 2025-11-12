from gym.envs import __init__  # noqa: F401
from gym.utils import get_args, task_registry
from gym.utils import KeyboardInterface
from gym.utils import VisualizationRecorder

# torch needs to be imported after isaacgym imports in local source
import torch
import numpy as np
import os


def setup(args):
    env_cfg, train_cfg = task_registry.create_cfgs(args)
    env_cfg.env.num_envs = 32
    if hasattr(env_cfg, "push_robots"):
        env_cfg.push_robots.toggle = True
    if hasattr(env_cfg, "commands"):
        env_cfg.commands.resampling_time = 9999
    env_cfg.env.episode_length_s = 50
    env_cfg.env.num_projectiles = 20
    task_registry.make_gym_and_sim()
    env_cfg.init_state.reset_mode = "reset_to_range"
    env = task_registry.make_env(args.task, env_cfg)
    train_cfg.runner.resume = True
    train_cfg.logging.enable_local_saving = False
    runner = task_registry.make_alg_runner(env, train_cfg)
    # runner.actor_cfg["disable_actions"] = True

    # * switch to evaluation mode (dropout for example)
    runner.switch_to_eval()
    return env, runner, train_cfg


def create_logging_dict(env, num_steps):
    """
    Creates a dictionary of tensors to store joint data for each timestep.
    """
    num_envs = env.num_envs
    num_dofs = env.num_dof

    log_dict = {
        "step": torch.zeros((num_steps,), dtype=torch.int32, device=env.device),
        "target_pos": torch.zeros((num_envs, num_steps, num_dofs), device=env.device),
        "actual_pos": torch.zeros((num_envs, num_steps, num_dofs), device=env.device),
        "torque": torch.zeros((num_envs, num_steps, num_dofs), device=env.device),
    }

    # Get joint names
    joint_names = (
        env.dof_names
        if hasattr(env, "dof_names")
        else [f"joint_{i}" for i in range(num_dofs)]
    )
    log_dict["joint_name"] = joint_names

    return log_dict


def play(env, runner, train_cfg):
    num_steps = int(env.max_episode_length)
    log_data = create_logging_dict(env, num_steps)

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

    try:
        for i in range(10 * int(env.max_episode_length)):
            if COMMANDS_INTERFACE:
                interface.update(env)
            if env.cfg.viewer.record:
                recorder.update(i)

            runner.set_actions(
                runner.actor_cfg["actions"],
                runner.get_inference_actions(),
                runner.actor_cfg["disable_actions"],
            )
            env.step()

            if i < num_steps:  # avoid overflow if loop is longer than episode
                log_data["step"][i] = i
                log_data["target_pos"][:, i, :] = env.dof_pos_target
                log_data["actual_pos"][:, i, :] = env.dof_pos
                log_data["torque"][:, i, :] = env.torques

            env.check_exit()  # user exit or viewer closed

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user, saving logs...")
    except SystemExit:
        print("\n[INFO] Viewer closed, saving logs...")
    finally:
        save_path = os.path.join(os.getcwd(), "joint_logs.npz")
        log_data_cpu = {
            k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v)
            for k, v in log_data.items()
        }
        np.savez_compressed(save_path, **log_data_cpu)
        print(f"\nSaved joint log to {save_path}")


if __name__ == "__main__":
    args = get_args()
    with torch.no_grad():
        env, runner, train_cfg = setup(args)
        play(env, runner, train_cfg)
