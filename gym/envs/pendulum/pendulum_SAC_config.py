from gym.envs.base.fixed_robot_config import FixedRobotCfgPPO


class PendulumSACRunnerCfg(FixedRobotCfgPPO):
    seed = -1
    runner_class_name = "OffPolicyRunner"

    class actor:
        hidden_dims = {
            "latent": [128, 128],
            "mean": [64, 32],
            "std": [64, 32],
        }
        # * can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = {
            "latent": "elu",
            "mean": "elu",
            "std": "tanh",
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
        normalize_obs = False

        class reward:
            class weights:
                theta = 0.0
                omega = 0.0
                equilibrium = 10.0
                energy = 1.0
                dof_vel = 0.0
                torques = 0.001

            class termination_weight:
                termination = 0.0

    class algorithm(FixedRobotCfgPPO.algorithm):
        initial_fill = 2
        storage_size = 10**6  # 17
        batch_size = 256
        max_gradient_steps = 10  # 10 # SB3: 1
        action_max = 1.0
        action_min = -1.0
        actor_noise_std = 1.0
        log_std_max = 4.0
        log_std_min = -20.0
        alpha = 0.2
        target_entropy = 1.0  # -1.0
        max_grad_norm = 1.0
        polyak = 0.98  # flipped compared to stable-baselines3 (polyak == 1-tau)
        gamma = 0.98
        alpha_lr = 3e-5
        actor_lr = 3e-5
        critic_lr = 3e-5

    class runner(FixedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "pendulum"
        max_iterations = 500  # number of policy updates
        algorithm_class_name = "SAC"
        save_interval = 50
        num_steps_per_env = 1
