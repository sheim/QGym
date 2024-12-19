from gym.envs import __init__  # noqa: F401
from gym.utils import get_args, task_registry
from gym.utils import KeyboardInterface
from gym.utils import VisualizationRecorder
import numpy as np
import matplotlib.pyplot as plt
from gym import LEGGED_GYM_ROOT_DIR
import os
from sklearn.decomposition import PCA
from isaacgym.torch_utils import (
    get_axis_params,
    torch_rand_float,
    quat_rotate_inverse,
    to_torch,
    quat_from_euler_xyz,
)

# torch needs to be imported after isaacgym imports in local source
import torch
from torch.multiprocessing import Process
from torch.multiprocessing import set_start_method
from scipy.interpolate import interp1d


def setup(args, run_name):
    # print("SETTING UP")
    # print(run_name)
    EXPORT_POLICY = False
    args.experiment_name = "network_sweep/osc/test"  # "ctrl_freq_sweep_new/osc/test"#
    args.load_run = run_name
    # args.ctrl_frequency = int(run_name.split("_")[-1])
    size = int((run_name.split("_")[-1]).split(" ")[0][1:-1])
    env_cfg, train_cfg = task_registry.create_cfgs(args)
    print("inside")
    print(size)
    print(run_name)
    if len(run_name.split("_")[-1].split(" ")) == 3:
        train_cfg.policy.actor_hidden_dims = [size, size, size // 2]
        train_cfg.policy.critic_hidden_dims = [size, size, size // 2]
    elif len(run_name.split("_")[-1].split(" ")) == 2:
        train_cfg.policy.actor_hidden_dims = [size, size]
        train_cfg.policy.critic_hidden_dims = [size, size]
    else:
        train_cfg.policy.actor_hidden_dims = [size]
        train_cfg.policy.critic_hidden_dims = [size]

    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 16)
    if hasattr(env_cfg, "push_robots"):
        env_cfg.push_robots.toggle = False
    if hasattr(env_cfg, "commands"):
        env_cfg.commands.resampling_time = 9999
    env_cfg.env.episode_length_s = 9999
    env_cfg.env.num_projectiles = 20
    task_registry.make_gym_and_sim()
    env = task_registry.make_env(args.task, env_cfg)
    # env.cfg.init_state.reset_mode = "reset_to_basic"
    train_cfg.runner.resume = True
    train_cfg.logging.enable_local_saving = False
    runner = task_registry.make_alg_runner(env, train_cfg)

    # * switch to evaluation mode (dropout for example)
    runner.switch_to_eval()

    if EXPORT_POLICY:
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name, "exported"
        )
        runner.export(path)
    return env, runner, train_cfg


def play(env, runner, train_cfg, vel, run_name):
    print(vel)

    # * set up recording
    if env.cfg.viewer.record:
        recorder = VisualizationRecorder(
            env, train_cfg.runner.experiment_name, train_cfg.runner.load_run
        )
    saveLogs = True
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
        "pca_scalings": [],
        "termination": [],
    }

    # * set up interface: GamepadInterface(env) or KeyboardInterface(env)
    COMMANDS_INTERFACE = hasattr(env, "commands")
    if COMMANDS_INTERFACE:
        # interface = GamepadInterface(env)
        interface = KeyboardInterface(env)
    env.timed_out[:] = True
    env.reset()
    env.commands[:, :] = 0
    print((10 / env.dt))
    foot_ids = [8, 16, 4, 12]
    foot_pos = {
        "lf": np.empty((16, 0, 3)),
        "lh": np.empty((16, 0, 3)),
        "rf": np.empty((16, 0, 3)),
        "rh": np.empty((16, 0, 3)),
    }
    bod_pos = np.empty((16, 0, 3))
    # 150 sim, 15 ctrl freq
    # command_freq = int(run_name.split("_")[-1])
    command_freq = 150
    for ctrl_i in range(int(10 / env.dt)):
        if (10 / env.dt) % (10 / env.dt * 1 / command_freq) == 0:
            env.commands[:, 0] = vel[0]
            env.commands[:, 2] = vel[1]
            if COMMANDS_INTERFACE:
                interface.update(env)
            if env.cfg.viewer.record:
                recorder.update(ctrl_i)
            runner.set_actions(
                runner.policy_cfg["actions"],
                # torch.randn(1,6).repeat(env.num_envs, 1).to(device = env.device),
                runner.get_inference_actions(),
                runner.policy_cfg["disable_actions"],
            )

        env.step()
        if saveLogs and ctrl_i > 250:
            # log["dof_pos_obs"] += env.dof_pos_obs.tolist()
            log["dof_vel"] += env.dof_vel.tolist()
            log["torques"] += env.torques.tolist()
            log["grf"] += [env.grf.tolist()]
            log["oscillators"] += env.oscillators.tolist()
            log["base_lin_vel"] += env.base_lin_vel.tolist()
            log["base_ang_vel"] += env.base_ang_vel.tolist()
            log["commands"] += env.commands.tolist()
            log["termination"] += [(env.terminated.float()).tolist()]

            for i, key in enumerate(foot_pos.keys()):
                pos = env._rigid_body_pos[:, foot_ids[i], 0:3] - env.root_states[:, 0:3]
                temp = quat_rotate_inverse(env.root_states[:, 3:7], pos).cpu().numpy()
                temp = np.reshape(temp, (16, 1, 3))
                foot_pos[key] = np.hstack((foot_pos[key], temp))

            # print(foot_pos['lf'].shape)
            # print(current_pos.shape)
            # log['foot_pos'] += [env._rigid_body_pos[:,i,:] for i in foot_ids]
            # log["dof_pos_error"] += (env.default_dof_pos - env.dof_pos).tolist()
            # log["pca_scalings"] += (env.pca_scalings.tolist())
            # reward_weights = runner.policy_cfg["reward"]["weights"]
            # log["reward"] += runner.get_rewards(reward_weights).tolist()

    return eval(log, vel, env, run_name, foot_pos)


def eval(log, vel, env, run_name, foot_pos):
    # evaluation
    # convert to numpy array
    for key in log.keys():
        log[key] = np.array(log[key])

    log["grf"][log["grf"] <= 1e-2] = 0

    log["base_lin_vel"][
        np.logical_and(log["base_lin_vel"] >= -1e-1, log["base_lin_vel"] <= 1e-1)
    ] = 0
    log["base_ang_vel"][
        np.logical_and(log["base_ang_vel"] >= -3e-1, log["base_ang_vel"] <= 3e-1)
    ] = 0
    base_lin_vel_avg = np.mean(log["base_lin_vel"], axis=0)
    base_ang_vel_avg = np.mean(log["base_ang_vel"], axis=0)
    print(base_lin_vel_avg)
    power = np.abs(base_lin_vel_avg[:1] * base_ang_vel_avg[2:3])
    # difference in signs between average vel and commanded vel (abs val)
    vel_dir = np.dot(base_lin_vel_avg[:1], vel[0])
    # norm
    vel_mag = np.array(
        [
            np.mean(
                (
                    log["base_lin_vel"][:, :1]
                    - np.ones_like(log["base_lin_vel"][:, :1]) * vel[0]
                )
                ** 2
            )
        ]
    )
    termination = np.array([np.sum((log["termination"]) > 0)])

    print(log["grf"].shape)
    print(power)
    print(vel_dir)
    print(vel_mag)
    print(termination)

    # ONLY FOOT 4 INTERPOLATED SEGMENTS
    plot_foot_dof(foot_pos, log["grf"], run_name, vel)
    start_groups = plotgrf(
        run_name, log, vel, env.cfg.control.ctrl_frequency, show=True
    )
    if start_groups is not None and vel[1] == 0:
        # calculate deltas for 1st 10 steps
        g0 = np.array([lst[0:10] for lst in start_groups[0].values()])
        g1 = np.array([lst[0:10] for lst in start_groups[1].values()])
        g2 = np.array([lst[0:10] for lst in start_groups[2].values()])
        g3 = np.array([lst[0:10] for lst in start_groups[3].values()])
        delta_rf_lh = np.abs(((g0 - g3) * env.dt / (1 / env.cfg.osc.omega))).flatten()
        delta_rf_lf = np.abs(((g0 - g1) * env.dt / (1 / env.cfg.osc.omega))).flatten()
        delta_rf_rh = np.abs(((g0 - g2) * env.dt / (1 / env.cfg.osc.omega))).flatten()
        deltas = np.vstack((delta_rf_lh, delta_rf_lf, delta_rf_rh)).T
    else:
        deltas = np.array([0, 0, 0])

    return {
        "power": power,
        "vel_dir": vel_dir,
        "vel_mag": vel_mag,
        "termination": termination,
        "deltas": deltas,
    }


def worker(run_name):
    args = get_args()
    # direction = [[0.5,0], [1.5,1.0]]
    # velocity_sweep = [[1.25,0.5], [1.5,1.0]] #0.25, 0.5, 1.0,
    start = -2
    end = 2
    step = 1.0

    # Generate the range of values for x and y
    x_values = np.arange(start, end + step, step)
    y_values = np.arange(start, end + step, step)

    # Create the velocity_sweep list with all combinations of x and y
    velocity_sweep = np.array([[x, y] for x in x_values for y in y_values])
    # velocity_sweep = np.array([[0,0]])
    # velocity_sweep = np.array([[-2, 0], [-1, 0], [1, 0], [2, 0]])
    print(velocity_sweep)
    successes = {
        "power": np.empty((1,)),
        "vel_dir": np.empty((1,)),
        "vel_mag": np.empty((1,)),
        "termination": np.empty((1,)),
        "deltas": np.empty((3,)),
    }
    with torch.no_grad():
        env, runner, train_cfg = setup(args, run_name)
    for vel in velocity_sweep:
        with torch.no_grad():
            temp = play(env, runner, train_cfg, vel, run_name)
            for key in successes.keys():
                print(temp[key].shape)
                successes[key] = np.vstack((successes[key], temp[key]))

    print(successes)
    plot = True
    np.save("deltas/" + run_name, successes["deltas"])

    scalings = {
        "power": [0, 2.0],
        "vel_dir": [-1.0, 1.0],
        "vel_mag": [0, 1.0],
        "termination": [0, 1.0],
    }
    if plot:
        for key in successes.keys():
            if key != "deltas":
                try:
                    yaw = np.unique(velocity_sweep[:, 1])
                    forward_backward = np.unique(velocity_sweep[:, 0])

                    # Create a grid for the heatmap
                    success_rate_grid = np.zeros((len(forward_backward), len(yaw)))
                    # Populate the grid with the success rates
                    for i, y in enumerate(yaw):
                        for j, x in enumerate(forward_backward):
                            # Find the index of the (x, y) pair in the original data
                            index = np.where(
                                (velocity_sweep[:, 0] == x)
                                & (velocity_sweep[:, 1] == y)
                            )
                            if index[0].size > 0:
                                success_rate_grid[i, j] = successes[key][index[0]]

                    # Create the heatmap
                    plt.imshow(
                        success_rate_grid,
                        aspect="auto",
                        cmap="viridis",
                        origin="lower",
                        extent=[
                            yaw.min(),
                            yaw.max(),
                            forward_backward.min(),
                            forward_backward.max(),
                        ],
                        vmin=scalings[key][0],
                        vmax=scalings[key][1],
                    )
                    # Add color bar to indicate the success rate
                    plt.colorbar(label="penalty")

                    # Add labels and title
                    plt.xlabel("Straight")
                    plt.ylabel("Yaw")
                    plt.title(f"Penalty {key} for Varying Commands")

                    file_path = os.path.join(
                        "/home/aileen/QGym/scripts/gridplots/" + run_name
                    )
                    if not os.path.exists(file_path):
                        os.makedirs(file_path)

                    file_path = os.path.join(file_path, run_name + "_" + key + ".png")
                    plt.savefig(file_path)
                    plt.close()
                except RuntimeWarning as e:
                    print(
                        f"Failed to plot heatmap for key {key} due to runtime warning: {e}"
                    )


if __name__ == "__main__":
    set_start_method("spawn")

    # Assign directory
    directory = r"/home/aileen/QGym/logs/network_sweep/osc/test"
    # directory = r"/home/aileen/QGym/logs/ctrl_freq_sweep_new/osc/test"

    # Iterate over files in directory
    for run_name in os.listdir(directory):
        print("START")
        print(run_name)
        p = Process(target=worker, args=(run_name,))
        p.start()
        p.join()  # Wait for the process to finish

        p.terminate()
        p.close()


def plotgrf(run_name, log, vel, sim_freq, show=False):
    print(log["grf"].shape)
    zeros = log["grf"] < 1e-5
    if vel[0] != 0 and np.sum(zeros) > 0:
        start = (
            log["grf"]
            * np.vstack(
                (
                    np.zeros((1, log["grf"].shape[1], log["grf"].shape[2])),
                    zeros[0:-1, :, :],
                )
            )
            > 0
        )
        end = (
            log["grf"]
            * np.vstack(
                (
                    zeros[1:, :, :],
                    np.zeros((1, log["grf"].shape[1], log["grf"].shape[2])),
                )
            )
            > 0
        )

        start = np.vstack(
            (start[1:, :, :], np.zeros((1, log["grf"].shape[1], log["grf"].shape[2])))
        )
        end = np.vstack(
            (np.zeros((1, log["grf"].shape[1], log["grf"].shape[2])), end[:-1, :, :])
        )

        startpos = np.nonzero(start[:, :, :] > 0)
        endpos = np.nonzero(end[:, :, :] > 0)

        start_row_indices, start_col_indices, foot_start = startpos
        end_row_indices, end_col_indices, foot_end = endpos

        start_indices = np.array(
            list(zip(start_row_indices, start_col_indices, foot_start))
        )
        end_indices = np.array(list(zip(end_row_indices, end_col_indices, foot_end)))

        # Sort start and end indices by column first, then by row
        start_indices_sorted = start_indices[
            np.lexsort((start_indices[:, 0], start_indices[:, 1], start_indices[:, 2]))
        ]
        end_indices_sorted = end_indices[
            np.lexsort((end_indices[:, 0], end_indices[:, 1], end_indices[:, 2]))
        ]

        start_indices_sorted = np.array(start_indices_sorted)
        end_indices_sorted = np.array(end_indices_sorted)
        start_groups = [{} for _ in range(4)]
        end_groups = [{} for _ in range(4)]
        if show:
            plt.figure(figsize=(10, 6))
        maxlen = 0
        for foot_index in range(4):
            # Grouping by columns within each foot
            start_groups[foot_index] = {
                col: []
                for col in np.unique(
                    start_indices_sorted[start_indices_sorted[:, 2] == foot_index, 1]
                )
            }
            end_groups[foot_index] = {
                col: []
                for col in np.unique(
                    end_indices_sorted[end_indices_sorted[:, 2] == foot_index, 1]
                )
            }

            for row, col, foot in start_indices_sorted:
                if foot == foot_index:
                    start_groups[foot_index][col].append(row)

            for row, col, foot in end_indices_sorted:
                if foot == foot_index:
                    end_groups[foot_index][col].append(row)

            # interp_length = 100
            for col in sorted(start_groups[foot_index].keys()):
                # Ensure the end groups do not start before the start groups
                if (
                    end_groups[foot_index][col]
                    and end_groups[foot_index][col][0]
                    < start_groups[foot_index][col][0]
                ):
                    end_groups[foot_index][col] = end_groups[foot_index][col][1:]

                # Ensure both groups have the same length
                min_length = min(
                    len(start_groups[foot_index][col]), len(end_groups[foot_index][col])
                )
                start_groups[foot_index][col] = start_groups[foot_index][col][
                    :min_length
                ]
                end_groups[foot_index][col] = end_groups[foot_index][col][:min_length]
                diff = np.max(
                    np.array(end_groups[foot_index][col])
                    - np.array(start_groups[foot_index][col])
                )
                if diff > maxlen:
                    maxlen = diff
        for foot_index in range(4):
            all_interpolated_segments = []
            for col in sorted(start_groups[foot_index].keys()):
                for start_row, end_row in zip(
                    start_groups[foot_index][col], end_groups[foot_index][col]
                ):
                    if start_row < end_row:  # Ensure valid segment
                        segment = np.append(
                            log["grf"][start_row:end_row, col, foot_index],
                            np.zeros((150 - (end_row - start_row),)),
                        )  # 42
                        all_interpolated_segments.append(segment)
                        # if len(segment) > 1:  # Ensure there are enough points to interpolate
                        #     x_original = np.linspace(0, 1, len(segment))
                        #     x_interp = np.linspace(0, 1, interp_length)
                        #     interpolator = interp1d(x_original, segment, kind='linear')
                        #     interpolated_segment = interpolator(x_interp)
                        #     all_interpolated_segments.append(interpolated_segment)

            # Compute the average of all interpolated segments
            if show and all_interpolated_segments:
                average_segment = np.mean(all_interpolated_segments, axis=0)
                # Plot the averaged segment
                x = np.arange(0, average_segment.shape[0] * 1 / sim_freq, 1 / sim_freq)
                plt.plot(x, average_segment, label=f"Foot {foot_index}")
            else:
                print(f"No valid segments for foot {foot_index}")
        if show:
            plt.legend()
            # plt.title('Averaged Segment Across All Columns and Feet')
            plt.xlabel("Time (s)")
            plt.ylabel("GRF Value")
            file_path = f"/home/aileen/QGym/scripts/grf/{run_name}"
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            file_name = f"{vel[0]}_{vel[1]}.png"
            file_path = os.path.join(file_path, file_name)
            # plt.show()
            plt.savefig(file_path)
            plt.close()
        return start_groups
    else:
        print("no steps")
        return None


def phase_plot(successes):
    plt.cla()
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(
        successes["deltas"][:, 0],
        successes["deltas"][:, 1],
        successes["deltas"][:, 2],
        alpha=0.5,
    )
    # Adding the red triangle marker
    ax.scatter(0, 0.5, 0.5, marker="^", c="red", s=500, label="Target Point")

    # Set labels
    ax.set_xlabel("Delta RF, LH")
    ax.set_ylabel("Delta RF, LF")
    ax.set_zlabel("Delta RF, RH")
    ax.view_init(elev=30, azim=-135)

    file_path = os.path.join(run_name)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    fig.savefig(run_name + "_delta.png")
    plt.close()


def plot_foot_dof(foot_pos, grf, run_name, vel):
    plt.cla()
    # fig3d = plt.figure(figsize=(14,8))
    # ax3d = fig3d.add_subplot(projection='3d')
    fig = plt.figure()
    ax = fig.add_subplot()
    grf = grf[:, 0, 0] > 0
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(projection="3d")
    colors = ["red", "green", "blue", "orange"]
    for i, key in enumerate(foot_pos.keys()):
        print(foot_pos[key][0, :, :].shape)
        print(
            np.tile(
                np.mean(foot_pos[key][0, :, :], axis=0),
                (foot_pos[key][0, :, :].shape[0], 1),
            ).shape
        )
        foot_pos_norm = foot_pos[key][0, :, :] - np.tile(
            np.mean(foot_pos[key][0, :, :], axis=0),
            (foot_pos[key][0, :, :].shape[0], 1),
        )
        proj_data, eigvecs, var = sklearnpca(foot_pos_norm)
        print(foot_pos_norm.shape)
        print(eigvecs)
        print(proj_data.shape)
        ax3d.plot3D(
            foot_pos_norm[:, 0],
            foot_pos_norm[:, 1],
            foot_pos_norm[:, 2],
            label=key,
            c=colors[i],
        )
        ax3d.scatter3D(
            foot_pos_norm[:, 0] * grf,
            foot_pos_norm[:, 1] * grf,
            foot_pos_norm[:, 2] * grf,
            s=5,
            alpha=0.2,
            c="black",
            zorder=2,
        )
        ax3d.plot3D(
            [0, eigvecs[0, 0] * 0.1],
            [0, eigvecs[1, 0] * 0.1],
            [0, eigvecs[2, 0] * 0.1],
            label=key,
            c=colors[i],
        )
        ax3d.plot3D(
            [0, eigvecs[0, 1] * 0.1],
            [0, eigvecs[1, 1] * 0.1],
            [0, eigvecs[2, 1] * 0.1],
            label=key,
            c=colors[i],
        )
        ax.scatter(
            proj_data[0, :] * grf,
            proj_data[1, :] * grf,
            s=5,
            alpha=0.2,
            c="black",
            zorder=2,
        )
        ax.plot(proj_data[0, :], proj_data[1, :], label=key, zorder=1)
    ax.legend(loc="upper right")
    ax3d.legend()
    file_path = f"/home/aileen/QGym/scripts/dof/{run_name}"
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    fig.savefig(f"{file_path}/{vel[0]}_{vel[1]}_dof_pos.png")
    fig3d.savefig(f"{file_path}/{vel[0]}_{vel[1]}_dof_pos_3d.png")
    plt.close()

    # do PCA to find plane that foot pos is on. then plot in that x,y axis over time.


def sklearnpca(normalized):
    pca = PCA(n_components=2)
    proj_data = pca.fit_transform(normalized).T
    print("sklearn pca var" + str(pca.explained_variance_ratio_))
    print(np.sum(pca.explained_variance_ratio_))

    # np.save("pca_components_ref_withpcascaling", pca.components_.T)
    eigvecs = pca.components_.T
    var = pca.explained_variance_ratio_
    print(eigvecs.shape)
    return proj_data, eigvecs, var
