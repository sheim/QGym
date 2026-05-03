from gym.envs import __init__  # noqa: F401
from gym.utils import get_args, task_registry
from gym.utils import KeyboardInterface
from gym.utils import VisualizationRecorder

# torch needs to be imported after isaacgym imports in local source
import torch
import numpy as np
import os
import csv

BASE_HEIGHT_REF = 1.0


# stability / success tracking
class RecoveryTracker:
    def __init__(self, env, output_file="recovery_results.csv"):  # noqa: F811
        self.env = env
        self.output_file = output_file

        self.num_envs = env.num_envs
        self.dt = env.dt

        # success criteria settings
        self.height_threshold = 0.8 * BASE_HEIGHT_REF
        self.orientation_threshold = 0.6
        self.required_stable_time = 0.4  # seconds
        self.required_stable_steps = int(self.required_stable_time / self.dt)

        # per-env tracking
        self.stable_counter = torch.zeros(
            self.num_envs, device=env.device, dtype=torch.long
        )

        self.success = torch.zeros(self.num_envs, device=env.device, dtype=torch.bool)

        self.time_to_stand = torch.full(
            (self.num_envs,),
            -1.0,
            device=env.device,
            dtype=torch.float,
        )

        self.max_tilt = torch.zeros(
            self.num_envs,
            device=env.device,
            dtype=torch.float,
        )

        self.fall = torch.zeros(
            self.num_envs,
            device=env.device,
            dtype=torch.bool,
        )

        self.elapsed_time = 0.0

        self._init_csv()

    def _init_csv(self):
        file_exists = os.path.isfile(self.output_file)

        if not file_exists:
            with open(self.output_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "torque_scale",
                        "latency_steps",
                        "env_id",
                        "success",
                        "time_to_stand",
                        "max_tilt",
                        "final_height",
                        "fall",
                    ]
                )

    def update(self):
        # call once every control step during play loop

        env = self.env

        height = env.base_height.flatten()

        tilt = torch.norm(env.projected_gravity[:, :2], dim=1)

        grf = env._compute_grf(grf_norm=True)
        contact = grf > env.cfg.osc.grf_threshold
        feet_contact_count = contact.float().sum(dim=1)

        # stable standing criteria
        height_ok = height > self.height_threshold
        orientation_ok = tilt < self.orientation_threshold
        feet_ok = feet_contact_count >= 3

        stable_now = height_ok & orientation_ok & feet_ok

        # consecutive stable steps
        self.stable_counter[stable_now] += 1
        self.stable_counter[~stable_now] = 0

        newly_successful = (self.stable_counter >= self.required_stable_steps) & (
            ~self.success
        )

        self.success[newly_successful] = True
        self.time_to_stand[newly_successful] = self.elapsed_time

        # max tilt
        self.max_tilt = torch.maximum(self.max_tilt, tilt)

        # fall detection
        if hasattr(env, "to_be_reset"):
            self.fall |= env.to_be_reset.clone()

        self.elapsed_time += self.dt

    def save_results(self):
        # call once at end of rollout
        env = self.env

        torque_scale = (
            env.cfg.perturbations.torque_scale
            if env.cfg.perturbations.reduced_torque_enabled
            else 1.0
        )

        latency_steps = (
            env.cfg.perturbations.latency_steps
            if env.cfg.perturbations.latency_enabled
            else 0
        )

        final_height = env.base_height.flatten().detach().cpu()
        success = self.success.detach().cpu()
        time_to_stand = self.time_to_stand.detach().cpu()
        max_tilt = self.max_tilt.detach().cpu()
        fall = self.fall.detach().cpu()

        with open(self.output_file, "a", newline="") as f:
            writer = csv.writer(f)

            for i in range(self.num_envs):
                writer.writerow(
                    [
                        float(torque_scale),
                        int(latency_steps),
                        int(i),
                        int(success[i].item()),
                        float(time_to_stand[i].item()),
                        float(max_tilt[i].item()),
                        float(final_height[i].item()),
                        int(fall[i].item()),
                    ]
                )

        print(f"Saved results to: {self.output_file}")


def make_results_filename(env):
    torque_scale = (
        env.cfg.perturbations.torque_scale
        if env.cfg.perturbations.reduced_torque_enabled
        else 1.0
    )

    latency_steps = (
        env.cfg.perturbations.latency_steps
        if env.cfg.perturbations.latency_enabled
        else 0
    )

    # 1.0 -> 1 and 0.5 stays 0.5
    if float(torque_scale).is_integer():
        torque_str = str(int(torque_scale))
    else:
        torque_str = str(torque_scale)

    filename = (
        f"recovery_results_" f"torque_{torque_str}_" f"latency_{int(latency_steps)}.csv"
    )

    return filename


def get_reward_fns(env):
    requested = [
        "tracking_height",
        "tendon_constraints",
        "tracking_lin_vel",
        "swing_grf",
        "stance_grf",
        "cursorial",
    ]

    out = {}
    for name in requested:
        fn_name = f"_reward_{name}"
        if hasattr(env, fn_name):
            out[name] = getattr(env, fn_name)
        else:
            print(f"reward not found (skipping): {fn_name}")
    return out


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
    env.set_belay(True)
    # * switch to evaluation mode (dropout for example)
    runner.switch_to_eval()

    # enable perturbations
    env.cfg.perturbations.enabled = True

    env.cfg.perturbations.reduced_torque_enabled = True
    env.cfg.perturbations.torque_scale = args.torque_scale

    env.cfg.perturbations.latency_enabled = True
    env.cfg.perturbations.latency_steps = args.latency_steps

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

    reward_fns = get_reward_fns(env)
    reward_names = list(reward_fns.keys())

    reward_log = {
        "reward_names": np.array(reward_names, dtype=object),
        "height_command": np.zeros((num_steps,), dtype=np.float32),
        "height_actual": np.zeros((num_steps,), dtype=np.float32),
        "switch_height": np.zeros((num_steps,), dtype=np.float32),
        "total_reward": np.zeros((num_steps,), dtype=np.float32),
    }

    for name in reward_names:
        reward_log[name] = np.zeros((num_steps,), dtype=np.float32)

    os.makedirs(args.results_dir, exist_ok=True)

    results_file = os.path.join(
        args.results_dir,
        f"recovery_results_torque_{args.torque_scale:g}_latency_{args.latency_steps}.csv",
    )

    tracker = RecoveryTracker(env, output_file=results_file)

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

    # height reward logging
    height_rew = np.zeros((num_steps,), dtype=np.float32)
    height_actual = np.zeros((num_steps,), dtype=np.float32)
    height_target = np.zeros((num_steps,), dtype=np.float32)

    # record init pos/height
    OUT_DIR = "play_logs"
    os.makedirs(OUT_DIR, exist_ok=True)

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
        env.commands[:, 3] = BASE_HEIGHT_REF

    max_play_steps = getattr(args, "max_steps", 10 * int(env.max_episode_length))
    headless = getattr(args, "headless", False)

    try:
        for i in range(max_play_steps):
            if COMMANDS_INTERFACE and not headless:
                interface.update(env)

            if env.cfg.viewer.record and not headless:
                recorder.update(i)

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
            tracker.update()

            if not headless and hasattr(env, "draw_belay_debug"):
                env.draw_belay_debug(force_scale=0.002)

            if actual_steps < num_steps:
                log_data["step"][actual_steps] = actual_steps
                log_data["target_pos"][:, actual_steps, :] = env.dof_pos_target
                log_data["actual_pos"][:, actual_steps, :] = env.dof_pos
                log_data["torque"][:, actual_steps, :] = env.torques

                log_obs_step(env, obs_log, obs_vars, actual_steps)

                r = env._reward_tracking_height()
                height_rew[actual_steps] = r[0].item()
                height_actual[actual_steps] = env.base_height[0].item()
                height_target[actual_steps] = env.commands[0, 3].item()

                reward_log["height_actual"][actual_steps] = env.base_height[0].item()
                reward_log["height_command"][actual_steps] = env.commands[0, 3].item()
                reward_log["switch_height"][actual_steps] = env._switch_height()[
                    0
                ].item()

                total = 0.0
                for name, fn in reward_fns.items():
                    r = fn()
                    r0 = r[0].item()
                    reward_log[name][actual_steps] = r0
                    total += r0

                reward_log["total_reward"][actual_steps] = total
                actual_steps += 1

            if not headless:
                env.check_exit()

        print(f"\n[INFO] Finished fixed play run: {max_play_steps} steps")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user, saving logs...")

    except SystemExit:
        print("\n[INFO] Viewer closed, saving logs...")

    finally:
        if env.cfg.viewer.record:
            if hasattr(recorder, "close"):
                recorder.close()
                print("[RECORD] recorder.close() called")
            elif hasattr(recorder, "save"):
                recorder.save()
                print("[RECORD] recorder.save() called")
            elif hasattr(recorder, "finish"):
                recorder.finish()
                print("[RECORD] recorder.finish() called")
        # slice to actual steps before saving
        """
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

        # save rewards
        reward_log_cpu = {}
        for k, v in reward_log.items():
            if isinstance(v, np.ndarray) and v.shape[0] == num_steps:
                reward_log_cpu[k] = v[:actual_steps]
            else:
                reward_log_cpu[k] = v
        np.savez_compressed("reward_logs.npz", **reward_log_cpu)
        """

        # save recovery results
        tracker.save_results()


if __name__ == "__main__":
    args = get_args()
    with torch.no_grad():
        env, runner, train_cfg = setup(args)
        play(env, runner, train_cfg)
