from gym.envs import __init__  # noqa: F401
from gym.utils import get_args, task_registry
from gym.utils import KeyboardInterface
from gym.utils import VisualizationRecorder

# torch needs to be imported after isaacgym imports in local source
import torch
import numpy as np
import os

BASE_HEIGHT_REF = 1.3


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

    # --- height reward logging ---
    height_rew = np.zeros((num_steps,), dtype=np.float32)
    height_actual = np.zeros((num_steps,), dtype=np.float32)
    height_target = np.zeros((num_steps,), dtype=np.float32)

    # ---------- record init pos/height ----------
    OUT_DIR = "play_logs"
    os.makedirs(OUT_DIR, exist_ok=True)

    # num dofs (don't rely on env.num_dofs, use tensor shape)
    num_dofs = env.dof_pos.shape[1]

    # Capture *one-time* init snapshot for all envs
    # root z: prefer root_states[:,2] if it exists; otherwise fall back to base_height
    if hasattr(env, "root_states"):
        init_root_z = env.root_states[:, 2].detach().cpu().numpy().astype(np.float32)
    else:
        init_root_z = env.base_height.detach().cpu().numpy().astype(np.float32)

    # commanded height at start
    init_cmd_h = env.commands[:, 3].detach().cpu().numpy().astype(np.float32)
    # starting joint positions
    init_dof_pos = env.dof_pos.detach().cpu().numpy().astype(np.float32)

    # Build a single 2D table: (num_envs, 2 + num_dofs)
    # col0 = init_root_z, col1 = init_cmd_h, col2.. = dof_pos
    init_table = np.zeros((env.num_envs, 2 + num_dofs), dtype=np.float32)
    init_table[:, 0] = init_root_z
    init_table[:, 1] = init_cmd_h
    init_table[:, 2:] = init_dof_pos

    # Column names/joint names
    if hasattr(env, "dof_names"):
        dof_names = list(env.dof_names)
    else:
        # fallback: generic names
        dof_names = [f"dof_{i}" for i in range(num_dofs)]

    col_names = ["init_root_z", "init_cmd_height"] + [f"init_{n}" for n in dof_names]

    # track actual number of simulation steps
    actual_steps = 0
    # track and print commanded height changes
    last_height_cmd = None

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

        # resample height command only
        h_min, h_max = env.cfg.commands.ranges.height
        env.commands[:, 3] = h_min + (h_max - h_min) * torch.rand(
            (env.num_envs,), device=env.device
        )

    try:
        for i in range(10 * int(env.max_episode_length)):
            if COMMANDS_INTERFACE:
                interface.update(env)
            if env.cfg.viewer.record:
                recorder.update(i)

            # print target/actual height (m)
            target_height = env.commands[0, 3].item()
            if (last_height_cmd is None) or (
                abs(target_height - last_height_cmd) > 1e-6
            ):
                actual_height = env.base_height[0].item()

                print(
                    f"[HEIGHT] actual = {actual_height:.3f} m | "
                    f"target = {target_height:.3f} m | "
                    f"error = {actual_height - target_height:.3f} | "
                    f"descend mode = {(env.commands[:, 3][0]) < 1.0}"
                )
                last_height_cmd = target_height

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

                # _reward_tracking_height returns (num_envs,) tensor
                r = env._reward_tracking_height()  # torch tensor
                height_rew[actual_steps] = r[0].item()
                height_actual[actual_steps] = env.base_height[0].item()
                height_target[actual_steps] = env.commands[0, 3].item()

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

    # ---------- save init data ----------
    init_npz_path = os.path.join(OUT_DIR, "init_snapshot.npz")
    np.savez_compressed(
        init_npz_path,
        init_table=init_table,
        col_names=np.array(col_names, dtype=object),
    )
    print(f"[saved] init snapshot -> {init_npz_path}")

    # save as csv
    init_csv_path = os.path.join(OUT_DIR, "init_snapshot.csv")
    with open(init_csv_path, "w") as f:
        f.write(",".join(col_names) + "\n")
        np.savetxt(f, init_table, delimiter=",", fmt="%.6f")
    print(f"[saved] init snapshot csv -> {init_csv_path}")


if __name__ == "__main__":
    args = get_args()
    with torch.no_grad():
        env, runner, train_cfg = setup(args)
        play(env, runner, train_cfg)
