import torch
from gym.envs.base.fixed_robot_config import FixedRobotCfgPPO
from gym.envs.pendulum.pendulum_config import PendulumCfg


class PendulumSACCfg(PendulumCfg):
    class env(PendulumCfg.env):
        num_envs = 256
        episode_length_s = 2.5

    class init_state(PendulumCfg.init_state):
        reset_mode = "reset_to_basic"
        default_joint_angles = {"theta": 0.0}
        dof_pos_range = {
            "theta": [-torch.pi / 2, torch.pi / 2],
        }
        dof_vel_range = {"theta": [-1, 1]}

    class control(PendulumCfg.control):
        ctrl_frequency = 25
        desired_sim_frequency = 200

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

    class actor:
        hidden_dims = {
            "latent": [400],
            "mean": [300],
            "std": [300],
        }
        # * can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = {
            "latent": "elu",
            "mean": "elu",
            "std": "elu",
        }

        normalize_obs = False
        obs = [
            "dof_pos_obs",
            "dof_vel",
        ]
        actions = ["tau_ff"]
        disable_actions = False

        class noise:
            dof_pos = 0.0
            dof_vel = 0.0

    class critic:
        obs = [
            "dof_pos_obs",
            "dof_vel",
        ]
        hidden_dims = [256, 256]
        # * can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = "elu"
        normalize_obs = True

        class reward:
            class weights:
                theta = 0.0
                omega = 0.0
                equilibrium = 2.0
                energy = 1.0
                dof_vel = 0.0
                torques = 0.01

            class termination_weight:
                termination = 0.0

    class algorithm(FixedRobotCfgPPO.algorithm):
        initial_fill = 10**3
        storage_size = 10**6  # 17
        batch_size = 256  # 4096
        max_gradient_steps = 1  # 10 # SB3: 1
        action_max = 1.0
        action_min = -1.0
        actor_noise_std = 1.0
        log_std_max = 4.0
        log_std_min = -20.0
        alpha = 0.8
        target_entropy = -1.0
        max_grad_norm = 1.0
        polyak = 0.98  # flipped compared to stable-baselines3 (polyak == 1-tau)
        gamma = 0.98
        alpha_lr = 3e-5
        actor_lr = 3e-5
        critic_lr = 3e-5

    class runner(FixedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "pendulum"
        max_iterations = 10000  # number of policy updates
        algorithm_class_name = "SAC"
        save_interval = 250
        num_steps_per_env = 1
