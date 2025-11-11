from gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotRunnerCfg,
)

BASE_HEIGHT_REF = 1.3


class MiniCheetahCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 2**12
        num_actuators = 1 + 4 * 5
        episode_length_s = 20

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"

    class init_state(LeggedRobotCfg.init_state):
        default_joint_angles = {
            "haa": 0.0,
            "hfe": 0.0,
            "kfe": 0.0,
            "pfe": 0.0,
            "pastern_to_foot": 0.0,
            "base_joint": 0.0,
        }

        # * reset setup chooses how the initial conditions are chosen.
        # * "reset_to_basic" = a single position
        # * "reset_to_range" = uniformly random from a range defined below
        reset_mode = "reset_to_range"

        # * default COM for basic initialization
        pos = [0.0, 0.0, 1.3]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        # * initialization for random range setup
        dof_pos_range = {
            "haa": [-0.2, 0.2],
            "hfe": [-0.5, 0.5],
            "kfe": [0.0, 0.0],
            "pfe": [0.0, 0.0],
            "pastern_to_foot": [0.0, 0.0],
            "base_joint": [0.0, 0.0],
        }
        dof_vel_range = {
            "haa": [0.0, 0.0],
            "hfe": [0.0, 0.0],
            "kfe": [0.0, 0.0],
            "pfe": [0.0, 0.0],
            "pastern_to_foot": [0.0, 0.0],
            "base_joint": [0.0, 0.0],
        }

        root_pos_range = [
            [0.0, 0.0],  # x
            [0.0, 0.0],  # y
            [1.3, 1.3],  # z
            [0.0, 0.0],  # roll
            [0.0, 0.0],  # pitch
            [0.0, 0.0],  # yaw
        ]
        root_vel_range = [
            [-0.5, 2.0],  # x
            [0.0, 0.0],  # y
            [-0.05, 0.05],  # z
            [0.0, 0.0],  # roll
            [0.0, 0.0],  # pitch
            [0.0, 0.0],  # yaw
        ]

    class control(LeggedRobotCfg.control):
        # * PD Drive parameters:
        stiffness = {
            "haa": 4000,
            "hfe": 4000,
            "kfe": 4000,
            "pfe": 4000,
            "pastern_to_foot": 4000,
            "base_joint": 50,
        }
        damping = {
            "haa": 250,
            "hfe": 250,
            "kfe": 250,
            "pfe": 250,
            "pastern_to_foot": 250,
            "base_joint": 10,
        }
        ctrl_frequency = 500  # how often the PDF controller/action updates run
        desired_sim_frequency = 1000  # how often the physics is calculated

    class commands:
        # * time before command are changed[s]
        resampling_time = 3.0

        class ranges:
            lin_vel_x = [-2.0, 3.0]  # min max [m/s]
            lin_vel_y = 1.0  # max [m/s]
            yaw_vel = 3  # max [rad/s]
            height = [0.61, 1.30]  # m

    class push_robots:
        toggle = False
        interval_s = 1
        max_push_vel_xy = 0.5
        push_box_dims = [0.3, 0.1, 0.1]  # x,y,z [m]

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 5.0]
        randomize_base_mass = False
        added_mass_range = [-1.0, 1.0]

    class asset(LeggedRobotCfg.asset):
        file = (
            "{LEGGED_GYM_ROOT_DIR}/resources/robots/"
            + "mini_cheetah/urdf/mini_cheetah_simple.urdf"
        )
        foot_name = "foot"
        penalize_contacts_on = ["shank"]
        terminate_after_contacts_on = ["base"]
        end_effector_names = ["foot"]
        collapse_fixed_joints = False
        self_collisions = 1
        flip_visual_attachments = False
        disable_gravity = False
        disable_motors = False
        joint_damping = 0.3
        fix_base_link = False
        rotor_inertia = [0.002268] + 4 * (
            [0.002268, 0.002268, 0.005484, 0.005484, 0.005484]
        )

    class reward_settings(LeggedRobotCfg.reward_settings):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 600.0
        base_height_target = BASE_HEIGHT_REF
        tracking_sigma = 0.25

    class scaling(LeggedRobotCfg.scaling):
        base_ang_vel = 0.3
        base_lin_vel = BASE_HEIGHT_REF
        dof_vel = [2.0] + (4 * [0.5, 0.5, 0.5, 0.5, 0.5])
        base_height = BASE_HEIGHT_REF
        dof_pos = [0.2] + (
            4 * [1.0, 1.0, 1.0, 1.0, 1.0]
        )  # i think these values were too small?
        dof_pos_obs = dof_pos
        dof_pos_target = [0.2] + (4 * [0.1, 0.1, 0.1, 0.1, 0.1])  # target joint angles
        tau_ff = [3600] + (4 * [3600, 3600, 400000, 400000, 5800])  # not being used
        commands = [3, 1, 3, 1]  # add height as a command


class MiniCheetahRunnerCfg(LeggedRobotRunnerCfg):
    seed = -1
    runner_class_name = "OnPolicyRunner"

    class actor(LeggedRobotRunnerCfg.actor):
        hidden_dims = [256, 256, 128]
        # * can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = "elu"
        obs = [
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_vel",
            "dof_pos_target",
        ]
        normalize_obs = True
        actions = ["dof_pos_target"]
        add_noise = True
        disable_actions = False

        class noise:
            scale = 1.0
            dof_pos_obs = 0.01
            base_ang_vel = 0.01
            dof_pos = 0.005
            dof_vel = 0.005
            lin_vel = 0.05
            ang_vel = [0.3, 0.15, 0.4]
            gravity_vec = 0.1

    class critic(LeggedRobotRunnerCfg.critic):
        hidden_dims = [128, 64]
        # * can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = "elu"
        obs = [
            "base_height",
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_vel",
            "dof_pos_target",
        ]
        normalize_obs = True

        class reward:
            class weights:
                tracking_lin_vel = 4.0
                tracking_ang_vel = 2.0
                lin_vel_z = 0.0
                ang_vel_xy = 0.01
                orientation = 1.0
                torques = 5.0e-6
                dof_vel = 0.0
                min_base_height = 1.5
                action_rate = 0.1
                action_rate2 = 0.01
                stand_still = 0.0
                dof_pos_limits = 0.0
                feet_contact_forces = 0.0
                dof_near_home = 0.0
                tracking_height = 1.5

            class termination_weight:
                termination = 0.01

    class algorithm(LeggedRobotRunnerCfg.algorithm):
        pass

    class runner(LeggedRobotRunnerCfg.runner):
        run_name = ""
        experiment_name = "mini_cheetah"
        max_iterations = 500
        algorithm_class_name = "PPO2"
        num_steps_per_env = 32
