from gym.envs import __init__  # noqa: F401
from gym.utils import get_args, task_registry
from gym.utils import KeyboardInterface
from gym.utils import VisualizationRecorder

# torch needs to be imported after isaacgym imports in local source
import torch
import numpy as np


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


def create_obs_logging_dict(env, obs_vars, num_steps):
    """
    Create a dictionary to log raw and scaled observation data.

    Returns:
        obs_log: dict with keys:
                 {var}_raw  -> [num_envs, num_steps, ...]
                 {var}_scaled -> [num_envs, num_steps, ...]
    """
    num_envs = env.num_envs
    obs_log = {}

    for var in obs_vars:
        val = getattr(env, var)
        shape = (num_envs, num_steps) + val.shape[1:]  # preserve timestep array shape

        # allocate tensors
        obs_log[f"{var}_raw"] = torch.zeros(shape, device=val.device)
        obs_log[f"{var}_scaled"] = torch.zeros_like(obs_log[f"{var}_raw"])

    return obs_log


def log_obs_step(env, obs_log, obs_vars, step_idx):
    for var in obs_vars:
        obs_log[f"{var}_raw"][:, step_idx, ...] = getattr(env, var)
        obs_log[f"{var}_scaled"][:, step_idx, ...] = env.get_state(var)


def create_logging_dict(env, num_steps):
    # creates a dictionary of tensors to store joint data for each timestep.
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

    obs_vars = [
        "base_height",
        "base_lin_vel",
        "base_ang_vel",
        "projected_gravity",
        "commands",
        "dof_pos_obs",
        "dof_vel",
        "dof_pos_target",
    ]

    obs_log = create_obs_logging_dict(env, obs_vars, num_steps)

    # track actual number of simulation steps
    actual_steps = 0

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

            # log only actual simulation steps
            if actual_steps < num_steps:
                # log joints
                log_data["step"][actual_steps] = actual_steps
                log_data["target_pos"][:, actual_steps, :] = env.dof_pos_target
                log_data["actual_pos"][:, actual_steps, :] = env.dof_pos
                log_data["torque"][:, actual_steps, :] = env.torques

                # log observations
                log_obs_step(env, obs_log, obs_vars, actual_steps)

                actual_steps += 1

            env.check_exit()  # user exit or viewer closed

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user, saving logs...")
    except SystemExit:
        print("\n[INFO] Viewer closed, saving logs...")
    finally:
        # slice to actual steps before saving
        log_data_cpu = {
            k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v)
            for k, v in log_data.items()
        }
        for key in ["step", "target_pos", "actual_pos", "torque"]:
            log_data_cpu[key] = log_data_cpu[key][:actual_steps]

        np.savez_compressed("joint_logs.npz", **log_data_cpu)
        print(f"\nSaved joint log to joint_logs.npz ({actual_steps} steps)")

        obs_log_cpu = {
            k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v)
            for k, v in obs_log.items()
        }
        for key in obs_log_cpu.keys():
            obs_log_cpu[key] = obs_log_cpu[key][:, :actual_steps, ...]

        np.savez_compressed("obs_logs.npz", **obs_log_cpu)
        print(f"\nSaved obs log to obs_logs.npz ({actual_steps} steps)")


if __name__ == "__main__":
    args = get_args()
    with torch.no_grad():
        env, runner, train_cfg = setup(args)
        play(env, runner, train_cfg)
