from gym.envs import __init__  # noqa: F401
from gym.utils import get_args, task_registry
from gym.utils import KeyboardInterface
from gym.utils import VisualizationRecorder
import numpy as np
import matplotlib.pyplot as plt
from gym import LEGGED_GYM_ROOT_DIR
import os

# torch needs to be imported after isaacgym imports in local source
import torch


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


def play(env, runner, train_cfg):
    saveLogs = True

    # * set up recording
    if env.cfg.viewer.record:
        recorder = VisualizationRecorder(
            env, train_cfg.runner.experiment_name, train_cfg.runner.load_run
        )
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
    }

    # * set up interface: GamepadInterface(env) or KeyboardInterface(env)

    pca_scalings_logged = torch.zeros((0, 2)).to(device=env.device)
    noiseplots = False
    count = False
    env.commands[:, :] = 0
    for i in range(10 * int(env.max_episode_length)):
        # print(env.pca_scalings[0,:])
        # env.pca_scalings = torch.randn(1,6).repeat(env.num_envs, 1)
        # print(env.pca_scalings.shape)
        # pca_scalings_logged = torch.vstack((pca_scalings_logged, env.pca_scalings[0,0:2]))
        # print(i)
        if saveLogs:
            log["dof_pos_obs"] += env.dof_pos_obs.tolist()
            log["dof_vel"] += env.dof_vel.tolist()
            log["torques"] += env.torques.tolist()
            # log["grf"] += env.grf.tolist()
            # log["oscillators"] += env.oscillators.tolist()
            # log["base_lin_vel"] += env.base_lin_vel.tolist()
            # log["base_ang_vel"] += env.base_ang_vel.tolist()
            # log["commands"] += env.commands.tolist()
            # log["dof_pos_error"] += (env.default_dof_pos - env.dof_pos).tolist()
            # log["pca_scalings"] += (env.pca_scalings.tolist())
            # reward_weights = runner.policy_cfg["reward"]["weights"]
            # log["reward"] += runner.get_rewards(reward_weights).tolist()

            if i == 3000:
                log["dof_names"] = env.dof_names
                np.savez("data_source_randomized", **log)
                print("done")
                if noiseplots:
                    plt.plot(
                        pca_scalings_logged[:, 1].cpu(), pca_scalings_logged[:, 0].cpu()
                    )
                    plt.xlabel("PCA scaling 1", fontsize=20)
                    plt.ylabel("PCA scaling 2", fontsize=20)
                    plt.show()

            if i % 150 == 0:
                command_dt, ang_dt = 0.1, 0.1
                ideal_command = ((torch.rand(1) * 8) - 4).to("cuda")
                if abs(ideal_command) < 0.1:
                    command_dt = abs(ideal_command)
                ideal_angvel = ((torch.rand(1) * 4) - 2).to("cuda")
                if abs(ideal_angvel) < 0.1:
                    ang_dt = abs(ideal_angvel)
            if ideal_command > 0:
                env.commands[:, 0] = torch.clamp(
                    env.commands[:, 0] + command_dt, max=ideal_command
                )
            else:
                env.commands[:, 0] = torch.clamp(
                    env.commands[:, 0] - command_dt, min=ideal_command
                )

            if ideal_angvel > 0:
                env.commands[:, 2] = torch.clamp(
                    env.commands[:, 2] + ang_dt,
                    max=ideal_angvel,
                )
            else:
                env.commands[:, 2] = torch.clamp(
                    env.commands[:, 2] - ang_dt,
                    min=ideal_angvel,
                )

        if env.cfg.viewer.record:
            recorder.update(i)
        runner.set_actions(
            runner.policy_cfg["actions"],
            # torch.randn(1,6).repeat(env.num_envs, 1).to(device = env.device),
            runner.get_inference_actions(),
            runner.policy_cfg["disable_actions"],
        )

        if i == 500 and count:
            # * get the body_name to body_index dict
            body_dict = env.gym.get_actor_rigid_body_dict(
                env.envs[0], env.actor_handles[0]
            )
            # * extract a list of body_names where the index is the id number
            body_names = [
                body_tuple[0]
                for body_tuple in sorted(
                    body_dict.items(), key=lambda body_tuple: body_tuple[1]
                )
            ]

            body_id = [
                body_names.index(body_name)
                for body_name in body_names
                if "base" in body_name
            ]

            body_pos = torch.zeros(16, 1, 3)
            body_pos = env._rigid_body_lin_vel[:, body_id]
            print(body_pos)
            success = abs(body_pos)[:, :, 0] >= 0.5 * torch.ones_like(body_pos)[:, :, 0]
            success2 = torch.zeros_like(success)
            success3 = torch.zeros_like(success)

            success2 = (
                abs(body_pos)[:, :, 1] <= 0.5 * torch.ones_like(body_pos)[:, :, 1]
            )
            success3 = (
                abs(body_pos)[:, :, 2] <= 0.5 * torch.ones_like(body_pos)[:, :, 2]
            )

            success = torch.logical_and(success, success2)
            success = torch.logical_and(success, success3)
            print(sum(success))
        env.step()
        env.check_exit()


if __name__ == "__main__":
    EXPORT_POLICY = True
    args = get_args()
    with torch.no_grad():
        env, runner, train_cfg = setup(args)
        play(env, runner, train_cfg)
