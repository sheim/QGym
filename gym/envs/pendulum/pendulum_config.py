import torch

from gym.envs.base.fixed_robot_config import FixedRobotCfg, FixedRobotCfgPPO


class PendulumCfg(FixedRobotCfg):
    class env(FixedRobotCfg.env):
        num_envs = 4096
        num_actuators = 1  # 1 for theta connecting base and pole
        episode_length_s = 5.0

    class terrain(FixedRobotCfg.terrain):
        pass

    class init_state(FixedRobotCfg.init_state):
        default_joint_angles = {"theta": 0.0}  # -torch.pi / 2.0}

        # * default setup chooses how the initial conditions are chosen.
        # * "reset_to_basic" = a single position
        # * "reset_to_range" = uniformly random from a range defined below
        reset_mode = "reset_to_range"

        # * initial conditions for reset_to_range
        dof_pos_range = {
            "theta": [-torch.pi, torch.pi],
        }
        dof_vel_range = {"theta": [-5, 5]}

    class control(FixedRobotCfg.control):
        actuated_joints_mask = [1]  # angle
        ctrl_frequency = 100
        desired_sim_frequency = 200
        stiffness = {"theta": 0.0}  # [N*m/rad]
        damping = {"theta": 0.0}  # [N*m*s/rad]

    class asset(FixedRobotCfg.asset):
        # * Things that differ
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/" + "pendulum/urdf/pendulum.urdf"
        disable_gravity = False
        disable_motors = False  # all torques set to 0
        joint_damping = 0.1

    class reward_settings(FixedRobotCfg.reward_settings):
        tracking_sigma = 0.25

    class scaling(FixedRobotCfg.scaling):
        dof_vel = 5.0
        dof_pos = 2.0 * torch.pi
        # * Action scales
        tau_ff = 1.0


class PendulumRunnerCfg(FixedRobotCfgPPO):
    seed = -1
    runner_class_name = "DataLoggingRunner"

    class policy(FixedRobotCfgPPO.policy):
        actor_obs = [
            "dof_pos",
            "dof_vel",
        ]

        critic_obs = actor_obs

        actions = ["tau_ff"]
        disable_actions = False
        standard_critic_nn = True

        class noise:
            dof_pos = 0.0
            dof_vel = 0.0

        class reward:
            class weights:
                theta = 0.1
                omega = 0.1
                equilibrium = 5.0
                energy = 1.0
                dof_vel = 0.01
                torques = 0.025

            class termination_weight:
                termination = 0.0

    class algorithm(FixedRobotCfgPPO.algorithm):
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 6
        # * mini batch size = num_envs*nsteps / nminibatches
        num_mini_batches = 4
        learning_rate = 1.0e-3
        schedule = "fixed"  # could be adaptive, fixed
        discount_horizon = 2.0  # [s]
        lam = 0.98
        # GAE_bootstrap_horizon = .0  # [s]
        desired_kl = 0.01
        max_grad_norm = 1.0
        standard_loss = True
        plus_c_penalty = 0.1

    class runner(FixedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "pendulum"
        max_iterations = 500  # number of policy updates
        algorithm_class_name = "PPO"
        num_steps_per_env = round(2.0 / (1.0 - 0.99))  # 32
