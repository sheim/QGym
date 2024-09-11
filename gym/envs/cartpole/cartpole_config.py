import torch

from gym.envs.base.fixed_robot_config import FixedRobotCfg, FixedRobotCfgPPO


class CartpoleCfg(FixedRobotCfg):
    class env(FixedRobotCfg.env):
        num_envs = 4024
        num_actuators = 1  # 1 for the cart force
        episode_length_s = 5.0

    class terrain(FixedRobotCfg.terrain):
        pass

    class init_state(FixedRobotCfg.init_state):
        default_joint_angles = {"slider_to_cart": 0.0, "cart_to_pole": 0.0}

        # * default setup chooses how the initial conditions are chosen.
        # * "reset_to_basic" = a single position
        # * "reset_to_range" = uniformly random from a range defined below
        reset_mode = "reset_to_range"

        # * initial conditions for reset_to_range
        dof_pos_range = {
            "slider_to_cart": [-2.5, 2.5],
            "cart_to_pole": [-torch.pi, torch.pi],
        }
        dof_vel_range = {
            "slider_to_cart": [-1.0, 1.0],
            "cart_to_pole": [-1.0, 1.0],
        }

    class control(FixedRobotCfg.control):
        actuated_joints_mask = [
            1,  # slider_to_cart
            0,
        ]  # cart_to_pole
        stiffness = {"slider_to_cart": 0.0}
        damping = {"slider_to_cart": 0.0}
        ctrl_frequency = 250
        desired_sim_frequency = 500

    class asset(FixedRobotCfg.asset):
        # * Things that differ
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/" + "cartpole/urdf/cartpole.urdf"
        flip_visual_attachments = False
        disable_gravity = False
        disable_motors = False  # all torques set to 0

    class reward_settings(FixedRobotCfg.reward_settings):
        pass

    class scaling(FixedRobotCfg.scaling):
        dof_pos = [4.0, 3.14]
        dof_vel = [5.0, 5.0]

        cart_obs = 5.0
        dof_vel = 5.0
        cart_vel_square = 10.0
        pole_vel_square = 25.0

        # * Action scales
        tau_ff = 10


class CartpoleRunnerCfg(FixedRobotCfgPPO):
    # We need random experiments to run
    seed = -1
    runner_class_name = "OnPolicyRunner"

    class actor(FixedRobotCfgPPO.actor):
        init_noise_std = 1.0
        num_layers = 2
        num_units = 32
        hidden_dims = [num_units] * num_layers
        activation = "elu"

        obs = [
            "cart_obs",
            "pole_trig_obs",
            "dof_vel",
            "cart_vel_square",
            "pole_vel_square",
        ]

        actions = ["tau_ff"]

        class noise:
            cart_pos = 0.001
            pole_pos = 0.001
            cart_vel = 0.010
            pole_vel = 0.010
            actuation = 0.00

    class critic:
        num_layers = 2
        num_units = 32
        hidden_dims = [num_units] * num_layers
        activation = "elu"

        obs = [
            "cart_obs",
            "pole_trig_obs",
            "dof_vel",
            "cart_vel_square",
            "pole_vel_square",
        ]

        class reward:
            class weights:
                pole_pos = 5
                pole_vel = 0.025
                cart_pos = 2
                torques = 0.1
                dof_vel = 0.1
                upright_pole = 25.0
                # energy = 10

            class termination_weight:
                termination = 0.0

    class algorithm(FixedRobotCfgPPO.algorithm):
        pass

    class runner(FixedRobotCfgPPO.runner):
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO2"
        num_steps_per_env = 32  # per iteration
        max_iterations = 500  # number of policy updates

        # * logging
        # * check for potential saves every this many iterations
        save_interval = 50
        run_name = ""
        experiment_name = "cartpole"

        # * load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
