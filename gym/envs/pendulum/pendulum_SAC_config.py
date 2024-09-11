import torch
from gym.envs.base.fixed_robot_config import FixedRobotCfgPPO
from gym.envs.pendulum.pendulum_config import PendulumCfg


class PendulumSACCfg(PendulumCfg):
    class env(PendulumCfg.env):
        num_envs = 1024
        episode_length_s = 10

    class init_state(PendulumCfg.init_state):
        reset_mode = "reset_to_range"
        default_joint_angles = {"theta": 0.0}
        dof_pos_range = {
            "theta": [-torch.pi, torch.pi],
        }
        dof_vel_range = {"theta": [-5, 5]}

    class control(PendulumCfg.control):
        ctrl_frequency = 100
        desired_sim_frequency = 100

    class asset(PendulumCfg.asset):
        joint_damping = 0.1

    class reward_settings(PendulumCfg.reward_settings):
        tracking_sigma = 0.25

    class scaling(PendulumCfg.scaling):
        dof_vel = 5.0
        dof_pos = 2.0 * torch.pi
        tau_ff = 1.0


class PendulumSACRunnerCfg(FixedRobotCfgPPO):
    seed = -1
    runner_class_name = "OffPolicyRunner"

    class actor(FixedRobotCfgPPO.actor):
        frequency = 10
        latent_nn = {"hidden_dims": [128, 64], "activation": "elu", "layer_norm": True}
        mean_nn = {"hidden_dims": [32], "activation": "elu", "layer_norm": True}
        std_nn = {"hidden_dims": [32], "activation": "elu", "layer_norm": True}
        nn_params = {"latent": latent_nn, "mean": mean_nn, "std": std_nn}

        normalize_obs = False
        obs = ["dof_pos_obs", "dof_vel"]
        actions = ["tau_ff"]
        disable_actions = False

        class noise:
            dof_pos = 0.0
            dof_vel = 0.0

    class critic(FixedRobotCfgPPO.critic):
        obs = [
            "dof_pos_obs",
            "dof_vel",
        ]
        hidden_dims = [128, 64, 32]
        # * can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = "elu"
        # TODO[lm]: Current normalization uses torch.no_grad, this should be changed
        normalize_obs = False

        class reward:
            class weights:
                theta = 0.0
                omega = 0.0
                equilibrium = 2.0
                energy = 0.5
                dof_vel = 0.0
                torques = 0.025

            class termination_weight:
                termination = 0.0

    class algorithm(FixedRobotCfgPPO.algorithm):
        initial_fill = 0
        storage_size = 100 * 1024  # 10**6  # 17
        batch_size = 256  # 4096
        max_gradient_steps = 1  # 10 # SB3: 1
        action_max = 2.0
        action_min = -2.0
        actor_noise_std = 1.0
        log_std_max = 4.0
        log_std_min = -20.0
        alpha = 0.2
        target_entropy = -1.0
        max_grad_norm = 1.0
        polyak = 0.995  # flipped compared to stable-baselines3 (polyak == 1-tau)
        gamma = 0.99
        alpha_lr = 1e-4
        actor_lr = 1e-3
        critic_lr = 1e-3

    class runner(FixedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "sac_pendulum"
        max_iterations = 30_000  # number of policy updates
        algorithm_class_name = "SAC"
        save_interval = 5000
        num_steps_per_env = 1
