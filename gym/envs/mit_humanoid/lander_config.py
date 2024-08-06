from gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotRunnerCfg,
)

BASE_HEIGHT_REF = 0.80


class LanderCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_actuators = 18
        episode_length_s = 5  # episode length in seconds

        sampled_history_length = 3  # n samples
        sampled_history_frequency = 10  # [Hz]

    class terrain(LeggedRobotCfg.terrain):
        pass

    class init_state(LeggedRobotCfg.init_state):
        # * default setup chooses how the initial conditions are chosen.
        # * "reset_to_basic" = a single position with added randomized noise.
        # * "reset_to_range" = a range of joint positions and velocities.
        # * "reset_to_traj" = feed in a trajectory to sample from.
        reset_mode = "reset_to_range"

        default_joint_angles = {
            "hip_yaw": 0.0,
            "hip_abad": 0.0,
            "hip_pitch": -0.667751,
            "knee": 1.4087,
            "ankle": -0.708876,
            "shoulder_pitch": 0.0,
            "shoulder_abad": 0.0,
            "shoulder_yaw": 0.0,
            "elbow": -1.25,
        }

        # * default COM for basic initialization
        pos = [0.0, 0.0, 0.6]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        # * initialization for random range setup

        dof_pos_range = {
            "hip_yaw": [-0.0, 0.0],
            "hip_abad": [-0.0, 0.0],
            "hip_pitch": [-0.667751, -0.667751],
            "knee": [1.4087, 1.4087],
            "ankle": [-0.708876, -0.708876],
            "shoulder_pitch": [0.0, 0.0],
            "shoulder_abad": [0.0, 0.0],
            "shoulder_yaw": [0.0, 0.0],
            "elbow": [0, 0],
        }
        dof_vel_range = {
            "hip_yaw": [-0.1, 0.1],
            "hip_abad": [-0.1, 0.1],
            "hip_pitch": [-0.1, 0.1],
            "knee": [-0.1, 0.1],
            "ankle": [-0.1, 0.1],
            "shoulder_pitch": [0.0, 0.0],
            "shoulder_abad": [0.0, 0.0],
            "shoulder_yaw": [0.0, 0.0],
            "elbow": [0.0, 0.0],
        }

        root_pos_range = [
            [0.0, 0.0],  # x
            [0.0, 0.0],  # y
            [1.0, 1.5],  # z
            [-0.1, 0.1],  # roll
            [-0.1, 0.1],  # pitch
            [-0.1, 0.1],
        ]  # yaw

        root_vel_range = [
            [-0.75, 2.75],  # x
            [-0.55, 0.55],  # y
            [-2.5, 0.25],  # z
            [-0.35, 0.35],  # roll
            [-0.35, 0.35],  # pitch
            [-0.35, 0.35],  # yaw
        ]

    class control(LeggedRobotCfg.control):
        # * PD Drive parameters:
        stiffness = {
            "hip_yaw": 30.0,
            "hip_abad": 30.0,
            "hip_pitch": 30.0,
            "knee": 30.0,
            "ankle": 30.0,
            "shoulder_pitch": 40.0,
            "shoulder_abad": 40.0,
            "shoulder_yaw": 40.0,
            "elbow": 50.0,
        }  # [N*m/rad]
        damping = {
            "hip_yaw": 2.0,
            "hip_abad": 2.0,
            "hip_pitch": 2.0,
            "knee": 2.0,
            "ankle": 2.0,
            "shoulder_pitch": 2.0,
            "shoulder_abad": 2.0,
            "shoulder_yaw": 2.0,
            "elbow": 1.0,
        }  # [N*m*s/rad]

        ctrl_frequency = 100
        desired_sim_frequency = 1000

    # class oscillator:
    #     base_frequency = 3.0  # [Hz]

    class commands:
        resampling_time = 10.0  # time before command are changed[s]

        class ranges:
            lin_vel_x = [-0.0, 0.0]  # min max [m/s] [-0.75, 0.75]
            lin_vel_y = 0.0  # max [m/s]
            yaw_vel = 0.0  # max [rad/s]

    class push_robots:
        toggle = True
        interval_s = 1
        max_push_vel_xy = 0.5
        push_box_dims = [0.1, 0.1, 0.3]  # x,y,z [m]

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1.0, 1.0]

    class asset(LeggedRobotCfg.asset):
        file = (
            "{LEGGED_GYM_ROOT_DIR}/resources/robots/"
            + "mit_humanoid/urdf/humanoid_F_sf_learnt.urdf"
        )
        # foot_collisionbox_names = ["foot"]
        foot_name = "foot"
        penalize_contacts_on = ["arm"]
        terminate_after_contacts_on = ["base"]
        end_effector_names = ["hand", "foot"]  # ??
        flip_visual_attachments = False
        self_collisions = 0  # 1 to disagble, 0 to enable...bitwise filter
        collapse_fixed_joints = False
        # * see GymDofDriveModeFlags
        # * (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        default_dof_drive_mode = 3
        fix_base_link = False
        disable_gravity = False
        disable_motors = False
        total_mass = 25.0

    class reward_settings(LeggedRobotCfg.reward_settings):
        soft_dof_pos_limit = 0.8
        soft_dof_vel_limit = 0.8
        soft_torque_limit = 0.8
        max_contact_force = 1500.0
        base_height_target = BASE_HEIGHT_REF
        tracking_sigma = 0.25

        # a smooth switch based on |cmd| (commanded velocity).
        switch_scale = 0.5
        switch_threshold = 0.2

    class scaling(LeggedRobotCfg.scaling):
        base_ang_vel = 2.5
        base_lin_vel = 1.5
        commands = 1
        base_height = BASE_HEIGHT_REF
        dof_pos = [
            0.1,
            0.2,
            0.8,
            0.8,
            0.8,
            0.1,
            0.2,
            0.8,
            0.8,
            0.8,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
        ]
        # # * Action scales
        dof_pos_target = dof_pos
        dof_vel = [
            0.5,
            1.0,
            4.0,
            4.0,
            2.0,
            0.5,
            1.0,
            4.0,
            4.0,
            2.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]
        dof_pos_history = 3 * dof_pos


class LanderRunnerCfg(LeggedRobotRunnerCfg):
    seed = -1
    runner_class_name = "OnPolicyRunner"

    class actor(LeggedRobotRunnerCfg.actor):
        init_noise_std = 1.0
        hidden_dims = [512, 256, 128]
        # * can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = "elu"
        smooth_exploration = False

        obs = [
            "base_height",
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            # "commands",
            "dof_pos_obs",
            "dof_vel",
            "dof_pos_history",
            # "sampled_history_dof_pos",
            # "sampled_history_dof_vel",
            # "sampled_history_dof_pos_target",
            # "oscillator_obs",
        ]
        normalize_obs = True

        actions = ["dof_pos_target"]
        disable_actions = False

        class noise:
            dof_pos = 0.005
            dof_vel = 0.05
            base_ang_vel = 0.025
            base_lin_vel = 0.025
            projected_gravity = 0.01
            feet_contact_state = 0.025

    class critic(LeggedRobotRunnerCfg.critic):
        hidden_dims = [512, 256, 128]
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
            "dof_pos_history",
        ]
        normalize_obs = True

        class reward:
            class weights:
                # tracking_lin_vel = 0.0
                # tracking_ang_vel = 1.5
                # orientation = 1.0
                torques = 5.0e-4
                # min_base_height = 1.5
                action_rate = 1e-3
                action_rate2 = 1e-3
                lin_vel_z = 0.0
                ang_vel_xy = 0.0
                # dof_vel = 0.25
                # stand_still = 0.25
                dof_pos_limits = 0.25
                dof_near_home = 0.25
                # stance = 1.0
                # swing = 1.0
                hips_forward = 0.0
                # walk_freq = 0.0  # 2.5

            class termination_weight:
                termination = 15

    class algorithm(LeggedRobotRunnerCfg.algorithm):
        # both
        gamma = 0.99
        lam = 0.95
        # shared
        batch_size = 2**15
        max_gradient_steps = 24
        # new
        storage_size = 2**17  # new
        batch_size = 2**15  #  new

        clip_param = 0.2
        learning_rate = 1.0e-3
        max_grad_norm = 1.0
        # Critic
        use_clipped_value_loss = True
        # Actor
        entropy_coef = 0.01
        schedule = "adaptive"  # could be adaptive, fixed
        desired_kl = 0.01
        lr_range = [1e-5, 1e-2]
        lr_ratio = 1.5

    class runner(LeggedRobotRunnerCfg.runner):
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO2"
        num_steps_per_env = 24
        max_iterations = 1000
        run_name = "Standing"
        experiment_name = "Humanoid"
        save_interval = 50
