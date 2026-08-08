from gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotRunnerCfg,
)

BASE_HEIGHT_REF = 0.4

GO2_DOF_NAMES = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]

GO2_FOOT_NAMES = [
    "FL_foot",
    "FR_foot",
    "RL_foot",
    "RR_foot",
]


class Go2Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 2**12
        num_actuators = 12
        episode_length_s = 5

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"

    class init_state(LeggedRobotCfg.init_state):
        # URDF joint-range midpoints. Relative position/target zero is therefore
        # equally far from each joint's lower and upper position limit.
        # default_joint_angles = {
        #     "hip_joint": 0.0,
        #     "calf_joint": -1.0,
        #     "FL_thigh_joint": 0.5995,
        #     "FR_thigh_joint": 0.5995,
        #     "RL_thigh_joint": 0.35,
        #     "RR_thigh_joint": 0.35,
        # }
        default_joint_angles = {
            "hip_joint": 0.0,
            "calf_joint": 0.0,
            "FL_thigh_joint": 0.0,
            "FR_thigh_joint": 0.0,
            "RL_thigh_joint": 0.0,
            "RR_thigh_joint": 0.0,
        }

        # * reset setup chooses how the initial conditions are chosen.
        # * "reset_to_basic" = a single position
        # * "reset_to_range" = uniformly random from a range defined below
        reset_mode = "reset_to_basic"

        # * default COM for basic initialization
        pos = [0.0, 0.0, 0.40]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        # * initialization for random range setup
        dof_pos_range = {
            "hip": [-0.01, 0.01],
            "thigh": [0.65, 0.67],
            "calf": [-1.37, -1.35],
        }
        dof_vel_range = {"hip": [0.0, 0.0], "thigh": [0.0, 0.0], "calf": [0.0, 0.0]}
        root_pos_range = [
            [0.0, 0.0],  # x
            [0.0, 0.0],  # y
            [0.450, 0.50],  # z
            [0.0, 0.0],  # roll
            [0.0, 0.0],  # pitch
            [0.0, 0.0],  # yaw
        ]
        root_vel_range = [
            [-0.5, 3.0],  # x
            [-0.1, 0.1],  # y
            [-0.05, 0.05],  # z
            [0.0, 0.0],  # roll
            [0.0, 0.0],  # pitch
            [0.0, 0.0],  # yaw
        ]

    class control(LeggedRobotCfg.control):
        # * PD Drive parameters:
        stiffness = {"hip": 20.0, "thigh": 20.0, "calf": 20.0}
        damping = {"hip": 0.5, "thigh": 0.5, "calf": 0.5}
        ctrl_frequency = 100
        desired_sim_frequency = 500
        gait_freq = [1.0, 3.0]  # oscillator frequency range [Hz]
        # Cycle offsets define a trot: front-left/rear-right move together,
        # half a cycle away from front-right/rear-left.
        gait_phase_offsets = {
            "FL_foot": 0.0,
            "FR_foot": 0.5,
            "RL_foot": 0.5,
            "RR_foot": 0.0,
        }
        # Canonical order is FL, FR, RL, RR; hip, thigh, calf within each leg.
        # q_ref = offset + amplitude * sin(phase + leg_phase).
        # These are relative PD targets; LeggedRobot adds default_dof_pos.
        # The thigh/calf amplitudes approximately preserve fore-aft foot
        # position while alternately extending the stance diagonal and
        # shortening the swing diagonal.
        gait_joint_offsets = 4 * [0.0, 0.96, -1.36]
        gait_joint_amplitudes = 4 * [0.0, -0.15, 0.30]

    class commands:
        resampling_time = 3.0
        var = 1.0

        class ranges:
            lin_vel_x = [-1.0, 0.0, 1.0, 3.0]
            lin_vel_y = 1.0  # max [m/s]
            yaw_vel = 3  # max [rad/s]

    class push_robots:
        toggle = True
        interval_s = 5
        max_push_vel_xy = 0.5
        push_box_dims = [0.3, 0.1, 0.1]  # x,y,z [m]

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.0]
        randomize_base_mass = False
        added_mass_range = [-1.0, 1.0]

    class asset(LeggedRobotCfg.asset):
        file = "{GYM_ROOT_DIR}/resources/robots/" + "go2/urdf/go2.urdf"
        foot_name = "foot"
        penalize_contacts_on = ["calf", "hip"]
        terminate_after_contacts_on = ["base", "Head_upper", "Head_lower"]
        end_effector_names = ["foot"]
        fix_base_link = False
        disable_gravity = False
        disable_motors = False
        joint_damping = 0.01
        rotor_inertia = [0.002268, 0.002268, 0.005484] * 4
        total_mass = 16.087  # sum of nominal URDF link masses [kg]

        class robot_layout:
            version = "go2_v1"
            dof_names = GO2_DOF_NAMES
            actuated_dof_names = GO2_DOF_NAMES
            body_groups = {"feet": GO2_FOOT_NAMES}

    class reward_settings(LeggedRobotCfg.reward_settings):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 600.0
        base_height_target = 0.9 * BASE_HEIGHT_REF
        tracking_sigma = 0.25

    class scaling(LeggedRobotCfg.scaling):
        # Canonical RobotLayout order is FL, FR, RL, RR, with
        # hip, thigh, calf inside each leg. Backends map native order to it.
        base_ang_vel = 0.3
        base_lin_vel = BASE_HEIGHT_REF
        # dof_vel = 4 * [30.1, 30.1, 15.7]
        dof_vel = 4 * [2.0, 2.0, 4.0]
        base_height = 0.3
        dof_pos = 4 * [1.0472, 2.53075, 0.94247]
        # dof_pos = 4 * [0.2, 0.3, 0.3]  # old
        dof_pos_obs = dof_pos
        dof_pos_target = [0.5 * x for x in dof_pos]
        tau_ff = 4 * [23.7, 23.7, 45.43]
        commands = [3, 1, 3]

    class mjspec_attributes:
        njmax = 130

    class mjspec_option_attributes:
        ccd_iterations = 50


class Go2RunnerCfg(LeggedRobotRunnerCfg):
    seed = -1
    runner_class_name = "OnPolicyRunner"

    class actor(LeggedRobotRunnerCfg.actor):
        hidden_dims = [256, 256, 128]
        # * can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = "elu"
        obs = [
            "base_height",
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_pos_history",
            "dof_vel",
            "dof_pos_target",
            "phase_obs",
            "phase_frequency",
        ]
        normalize_obs = False
        actions = ["dof_pos_target"]
        add_noise = False
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
            "phase_obs",
            "phase_frequency",
        ]
        normalize_obs = False

        class reward:
            class weights:
                tracking_lin_vel = 4.0
                tracking_ang_vel = 2.0
                lin_vel_z = 0.0
                ang_vel_xy = 0.01
                orientation = 1.0
                torques = 5.0e-6
                dof_vel = 0.0
                min_base_height = 0.5
                action_rate = 0.1
                action_rate2 = 0.01
                stand_still = 0.0
                dof_pos_limits = 0.0
                feet_contact_forces = 0.0
                dof_near_home = 0.0
                # Preserve the old combined term's approximate +/-0.625 range,
                # while making both stance feet necessary for positive credit.
                trot_support = 0.625
                swing_contact = 1.25

            class termination_weight:
                termination = 0.01

    class algorithm(LeggedRobotRunnerCfg.algorithm):
        # both
        gamma = 0.99
        lam = 0.95
        # shared
        batch_size = 2**15
        rollout_size = 2**16
        max_gradient_steps = 32
        # new
        clip_param = 0.2
        learning_rate = 1.0e-3
        max_grad_norm = 1.0
        # Critic
        use_clipped_value_loss = True
        # Actor
        entropy_coef = 0.01
        schedule = "adaptive"  # could be adaptive, fixed
        desired_kl = 0.01
        lr_range = [2e-5, 1e-2]
        lr_ratio = 1.5

    class runner(LeggedRobotRunnerCfg.runner):
        run_name = ""
        experiment_name = "go2"
        max_iterations = 500
        algorithm_class_name = "PPO2"
