from gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotRunnerCfg,
)

BASE_HEIGHT_REF = 0.80


class MITHumanoidCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 49 + 3 * 18  # 121
        num_actuators = 18
        episode_length_s = 5  # episode length in seconds

        history_length = 3  # n samples
        history_frequency = 10  # [Hz]

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
            "hip_pitch": -0.4,
            "knee": 0.9,
            "ankle": -0.45,
            "shoulder_pitch": 0.0,
            "shoulder_abad": 0.0,
            "shoulder_yaw": 0.0,
            "elbow": 0.0,
        }

        # * default COM for basic initialization
        pos = [0.0, 0.0, 0.66]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        # * initialization for random range setup

        dof_pos_range = {
            "hip_yaw": [0.0, 0.0],
            "hip_abad": [0.0, 0.0],
            "hip_pitch": [-0.29, -0.25],
            "knee": [0.67, 0.71],
            "ankle": [-0.43, -0.39],
            "shoulder_pitch": [0.0, 0.0],
            "shoulder_abad": [0.0, 0.0],
            "shoulder_yaw": [0.0, 0.0],
            "elbow": [0.0, 0.0],
        }
        dof_vel_range = {
            "hip_yaw": [-0.0, 0.1],
            "hip_abad": [-0.0, 0.1],
            "hip_pitch": [-0.1, -0.1],
            "knee": [-0.05, 0.05],
            "ankle": [-0.05, 0.05],
            "shoulder_pitch": [0.0, 0.0],
            "shoulder_abad": [0.0, 0.0],
            "shoulder_yaw": [0.0, 0.0],
            "elbow": [0.0, 0.0],
        }

        root_pos_range = [
            [0.0, 0.0],  # x
            [0.0, 0.0],  # y
            [0.7, 0.8],  # z
            [-0.1, 0.1],  # roll
            [-0.1, 0.1],  # pitch
            [-0.1, 0.1],
        ]  # yaw

        root_vel_range = [
            [-0.5, 2.5],  # x
            [-0.5, 0.5],  # y
            [-0.35, 0.1],  # z
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
            "elbow": 40.0,
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

        filter_gain = 0.1586  # 1: no filtering, 0: wall

    class oscillator:
        base_frequency = 1.5  # [Hz]

    class commands:
        resampling_time = 10.0  # time before command are changed[s]

        class ranges:
            lin_vel_x = [-1.0, 4.0]  # min max [m/s]
            lin_vel_y = 1.0  # max [m/s]
            yaw_vel = 1  # max [rad/s]

    class push_robots:
        toggle = False
        interval_s = 2
        max_push_vel_xy = 0.6
        push_box_dims = [0.1, 0.1, 0.3]  # x,y,z [m]

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1.0, 1.0]

    class asset(LeggedRobotCfg.asset):
        file = (
            "{LEGGED_GYM_ROOT_DIR}/resources/robots/"
            + "mit_humanoid/urdf/humanoid_R_sf.urdf"
        )
        # foot_collisionbox_names = ["foot"]
        foot_name = "foot"
        penalize_contacts_on = ["base", "arm"]
        terminate_after_contacts_on = ["base"]
        end_effector_names = ["hand", "foot"]
        flip_visual_attachments = False
        self_collisions = 1  # 1 to disagble, 0 to enable...bitwise filter
        collapse_fixed_joints = False
        # * see GymDofDriveModeFlags
        # * (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        default_dof_drive_mode = 3
        fix_base_link = False
        disable_gravity = False
        disable_motors = False
        total_mass = 25.0

    class reward_settings(LeggedRobotCfg.reward_settings):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 1500.0
        base_height_target = BASE_HEIGHT_REF
        tracking_sigma = 0.5

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
        dof_pos_obs = dof_pos
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
        dof_pos_history = 3 * dof_pos_obs


class MITHumanoidRunnerCfg(LeggedRobotRunnerCfg):
    seed = -1
    runner_class_name = "OnPolicyRunner"

    class policy(LeggedRobotRunnerCfg.policy):
        disable_actions = False
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        # * can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = "tanh"

        actor_obs = [
            "base_height",
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_vel",
            "dof_pos_history",
            "oscillator_obs",
        ]
        critic_obs = actor_obs

        actions = ["dof_pos_target"]

        class noise:
            base_height = 0.05
            dof_pos_obs = 0.0
            dof_vel = 0.0
            base_lin_vel = 0.1
            base_ang_vel = 0.2
            projected_gravity = 0.05
            height_measurements = 0.1

        class reward:
            class weights:
                tracking_ang_vel = 1.5
                tracking_lin_vel = 3.0
                orientation = 1.0
                torques = 5.0e-4
                min_base_height = 1.5
                action_rate = 0.01
                action_rate2 = 0.001
                lin_vel_z = 0.0
                ang_vel_xy = 0.0
                dof_vel = 0.25
                stand_still = 0.25
                dof_pos_limits = 0.25
                dof_near_home = 0.25
                stance = 1.0
                swing = 1.0
                walk_freq = 0.0

            class termination_weight:
                termination = 15

    class algorithm(LeggedRobotRunnerCfg.algorithm):
        # * training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.002
        num_learning_epochs = 5
        # * mini batch size = num_envs*nsteps / nminibatches
        num_mini_batches = 4
        learning_rate = 5.0e-4
        schedule = "adaptive"  # could be adaptive, fixed
        gamma = 0.999
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner(LeggedRobotRunnerCfg.runner):
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO2"
        num_steps_per_env = 32
        max_iterations = 1000
        run_name = "Standing"
        experiment_name = "Humanoid"
        save_interval = 50
