from gym.envs import __init__  # noqa: F401
from gym.utils import get_args, task_registry
from gym.utils import KeyboardInterface
from gym.utils import VisualizationRecorder
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import proj3d

# torch needs to be imported after isaacgym imports in local source
import torch


def setup(args):
    num = 1
    env_cfg, train_cfg = task_registry.create_cfgs(args)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 16)
    if hasattr(env_cfg, "push_robots"):
        env_cfg.push_robots.toggle = False
    if hasattr(env_cfg, "commands"):
        env_cfg.commands.resampling_time = 9999
    env_cfg.env.episode_length_s = 9999
    env_cfg.env.num_projectiles = 20
    if num == 1:
        env_cfg.asset.fix_base_link = False
    else:
        env_cfg.asset.fix_base_link = True
    task_registry.make_gym_and_sim()

    env_cfg.init_state.pos = [0, 0, 0.35]

    env_cfg.init_state.reset_mode = "reset_to_basic"
    env = task_registry.make_env(args.task, env_cfg)
    train_cfg.runner.resume = True
    train_cfg.logging.enable_local_saving = False
    runner = task_registry.make_alg_runner(env, train_cfg)

    # * switch to evaluation mode (dropout for example)
    runner.switch_to_eval()
    return env, runner, train_cfg


def play(env, runner, train_cfg):
    num = 1
    # * set up recording
    if env.cfg.viewer.record:
        recorder = VisualizationRecorder(
            env, train_cfg.runner.experiment_name, train_cfg.runner.load_run
        )
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
        "pca_scalings": [],
    }

    # # * set up interface: GamepadInterface(env) or KeyboardInterface(env)
    # COMMANDS_INTERFACE = hasattr(env, "commands")
    # if COMMANDS_INTERFACE:
    #     # interface = GamepadInterface(env)
    #     interface = KeyboardInterface(env)
    dof_logged = torch.zeros((0, 3)).to(device=env.device)
    dof_logged_2 = torch.zeros((0, 3)).to(device=env.device)
    dof_logged_3 = torch.zeros((0, 3)).to(device=env.device)
    dof_logged_4 = torch.zeros((0, 3)).to(device=env.device)

    dist = torch.zeros((0, 3)).to(device=env.device)
    plt.figure()
    ax = plt.axes(projection="3d")
    noiseplots = False

    env.commands[:] = 0.0
    for i in range(1000):
        # print(env.pca_scalings.shape)
        # env.pca_scalings = torch.randn(1,6).repeat(env.num_envs, 1)
        # print(env.pca_scalings.shape)

        # print(i)
        if saveLogs:
            log["dof_pos_obs"] += env.dof_pos_obs.tolist()
            log["dof_vel"] += env.dof_vel.tolist()
            log["torques"] += env.torques.tolist()
            log["grf"] += env.grf.tolist()
            log["oscillators"] += env.oscillators.tolist()
            log["base_lin_vel"] += env.base_lin_vel.tolist()
            log["base_ang_vel"] += env.base_ang_vel.tolist()
            log["commands"] += env.commands.tolist()
            log["dof_pos_error"] += (env.default_dof_pos - env.dof_pos).tolist()
            log["pca_scalings"] += env.pca_scalings.tolist()
            # reward_weights = runner.policy_cfg["reward"]["weights"]
            # log["reward"] += runner.get_rewards(reward_weights).tolist()

            if i == 1000:
                log["dof_names"] = env.dof_names
                np.savez("gaussian_scalings_logs", **log)

        if i == 1000 and noiseplots:
            # print(dist.shape)
            # m,_ = torch.max(obsdist,dim=0)
            # min,_ = torch.min(obsdist,dim=0)
            # dist*=(m-min)
            dof_logged = dof_logged.cpu().numpy()
            ax.plot3D(dof_logged[:, 0], dof_logged[:, 1], dof_logged[:, 2])
            ax.scatter3D(
                dist[:, 0].cpu(),
                dist[:, 1].cpu(),
                dist[:, 2].cpu(),
                c="r",
                marker=".",
                s=5,
            )
            ax.set_xlabel("x", fontsize=10)
            ax.set_ylabel("y", fontsize=10)
            ax.set_zlabel("z", fontsize=10)

            plt.show()
            # plt.savefig("exploration3d.png")

        env.commands[:, 0] = torch.clamp(
            env.commands[:, 0] + 0.5,
            max=4.0,
        )
        # if COMMANDS_INTERFACE:
        #     interface.update(env)

        if num == 1:
            dof_logged = torch.vstack((dof_logged, env._rigid_body_pos[0, 4, :]))
            dof_logged_2 = torch.vstack((dof_logged_2, env._rigid_body_pos[0, 8, :]))
            dof_logged_3 = torch.vstack((dof_logged_3, env._rigid_body_pos[0, 12, :]))
            dof_logged_4 = torch.vstack((dof_logged_4, env._rigid_body_pos[0, 16, :]))

            runner.set_actions(
                runner.policy_cfg["actions"],
                runner.get_inference_actions(),
                runner.policy_cfg["disable_actions"],
            )
        else:
            env.dof_pos_target = torch.randn_like(env.dof_pos_target)

            # action_dist=action_dist[0,1:3]

            in_contact = torch.gt(
                torch.norm(env.contact_forces[:, 4, :], dim=-1),
                50.0,
            )[0]
            # print(in_contact)
            dist = torch.vstack((dist, env._rigid_body_pos[0, 4, :]))
        env.step()
        env.check_exit()
    # np.save("dof_logged_baseline", dof_logged.cpu().numpy())
    return dof_logged, dof_logged_2, dof_logged_3, dof_logged_4, dist


if __name__ == "__main__":
    args = get_args()

    with torch.no_grad():
        env, runner, train_cfg = setup(args)
        dof_logged, dof_logged_2, dof_logged_3, dof_logged_4, _ = play(
            env, runner, train_cfg
        )
    dof_logged = dof_logged.cpu().numpy()
    dof_logged_2 = dof_logged_2.cpu().numpy()
    dof_logged_3 = dof_logged_3.cpu().numpy()
    dof_logged_4 = dof_logged_4.cpu().numpy()
    print("done")
    # args = get_args()
    # with torch.no_grad():
    #     env, runner, train_cfg = setup(args)
    #     _,dist = play(env, runner, train_cfg)

    plt.figure()
    ax = plt.axes(projection="3d")
    # dof_logged_baseline = np.load("dof_logged_baseline.npy")
    ax.plot3D(dof_logged[1:, 0], dof_logged[1:, 1], dof_logged[1:, 2], label="Leg 1")
    ax.plot3D(
        dof_logged_2[1:, 0], dof_logged_2[1:, 1], dof_logged_2[1:, 2], label="Leg 2"
    )
    ax.plot3D(
        dof_logged_3[1:, 0], dof_logged_3[1:, 1], dof_logged_3[1:, 2], label="Leg 3"
    )
    ax.plot3D(
        dof_logged_4[1:, 0], dof_logged_4[1:, 1], dof_logged_4[1:, 2], label="Leg 4"
    )
    # ax.legend()
    # print(dist.shape)
    # ax.scatter3D(dist[:,0].cpu(),dist[:,1].cpu(), dist[:,2].cpu(), c='r',marker='.',s=5)
    ax.set_title("Emergent Asymmetry PCA Foot Positions During Trajectory")
    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("y", fontsize=10)
    ax.set_zlabel("z", fontsize=10)
    plt.show()

    # plt.figure()
    # ax = plt.axes(projection='3d')
    # dof_logged_baseline = np.load("dof_logged_baseline.npy")
    # ax.plot3D(dof_logged[1:,0],dof_logged[1:,1],dof_logged[1:,2], label='PCA')
    # ax.plot3D(dof_logged_baseline[1:,0],dof_logged_baseline[1:,1],dof_logged_baseline[1:,2], label='Baseline')

    # # print(dist.shape)
    # # ax.scatter3D(dist[:,0].cpu(),dist[:,1].cpu(), dist[:,2].cpu(), c='r',marker='.',s=5)
    # ax.set_xlabel("x", fontsize=10)
    # ax.set_ylabel("y", fontsize=10)
    # ax.set_zlabel("z", fontsize=10)
    # #plt.show()
    # plt.savefig("exploration3d.png")

    # plt.figure()
    # ax = plt.axes()
    # # dof_logged = np.load("dof_logged.npy")
    # ax.plot(dof_logged[1:,0],dof_logged[1:,2])
    # ax.plot(dof_logged_baseline[1:,1],dof_logged_baseline[1:,2])

    # # print(dist.shape)
    # # ax.scatter3D(dist[:,0].cpu(),dist[:,1].cpu(), dist[:,2].cpu(), c='r',marker='.',s=5)
    # ax.set_xlabel("y", fontsize=10)
    # ax.set_ylabel("z", fontsize=10)

    # #plt.savefig("exploration3d.png")
    # #print("saved")
    # plt.figure()
    # ax = plt.axes()
    # # dof_logged = np.load("dof_logged.npy")
    # ax.plot(dof_logged[1:,0],dof_logged[1:,2])
    # ax.plot(dof_logged_baseline[1:,0],dof_logged_baseline[1:,2])

    # # print(dist.shape)
    # # ax.scatter3D(dist[:,0].cpu(),dist[:,1].cpu(), dist[:,2].cpu(), c='r',marker='.',s=5)
    # ax.set_xlabel("x", fontsize=10)
    # ax.set_ylabel("z", fontsize=10)
    # plt.show()
    # #plt.savefig("exploration3d.png")
    # #print("saved")
