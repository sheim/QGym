import torch

from gym.envs.base.fixed_robot_config import FixedRobotCfg, FixedRobotCfgPPO


class PendulumCfg(FixedRobotCfg):
    class env(FixedRobotCfg.env):
        num_envs = 2**13
        num_actuators = 1  # 1 for theta connecting base and pole
        episode_length_s = 5.0

    class terrain(FixedRobotCfg.terrain):
        pass

    class viewer:
        ref_env = 0
        pos = [10.0, 5.0, 10.0]  # [m]
        lookat = [0.0, 0.0, 0.0]  # [m]

    class init_state(FixedRobotCfg.init_state):
        default_joint_angles = {"theta": 0.0}  # -torch.pi / 2.0}

        # * default setup chooses how the initial conditions are chosen.
        # * "reset_to_basic" = a single position
        # * "reset_to_range" = uniformly random from a range defined below
        reset_mode = "reset_to_range"

        # * initial conditions for reset_to_range
        dof_pos_range = {
            "theta": [-torch.pi / 4, torch.pi / 4],
        }
        dof_vel_range = {"theta": [-1, 1]}

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
        joint_damping = 0.5

    class reward_settings(FixedRobotCfg.reward_settings):
        tracking_sigma = 0.25

    class scaling(FixedRobotCfg.scaling):
        dof_vel = 5.0
        dof_pos = 2.0 * torch.pi
        # * Action scales
        tau_ff = 5.0


class PendulumRunnerCfg(FixedRobotCfgPPO):
    seed = -1
    runner_class_name = "OnPolicyRunner"

    class actor:
        hidden_dims = [128, 64, 32]
        # * can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = "tanh"

        obs = [
            "dof_pos",
            "dof_vel",
        ]

        actions = ["tau_ff"]
        disable_actions = False

        class noise:
            dof_pos = 0.0
            dof_vel = 0.0

    class critic:
        obs = [
            "dof_pos",
            "dof_vel",
        ]
        hidden_dims = [128, 64, 32]
        # * can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = "tanh"

        class reward:
            class weights:
                theta = 0.0
                omega = 0.0
                equilibrium = 1.0
                energy = 0.05
                dof_vel = 0.0
                torques = 0.01

            class termination_weight:
                termination = 0.0

    class algorithm(FixedRobotCfgPPO.algorithm):
        # both
        gamma = 0.99
        lam = 0.95
        # shared
        batch_size = 2**15
        max_grad_steps = 10
        # new
        storage_size = 2**17  # new
        batch_size = 2**15  #  new

        clip_param = 0.2
        learning_rate = 1.0e-3
        max_grad_norm = 1.0
        # Critic
        use_clipped_value_loss = True
        # Actor
        entropy_coef = 0.1
        schedule = "adaptive"  # could be adaptive, fixed
        desired_kl = 0.01

    class runner(FixedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "pendulum"
        max_iterations = 500  # number of policy updates
        algorithm_class_name = "PPO2"
        num_steps_per_env = 32
