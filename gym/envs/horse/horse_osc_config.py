from gym.envs.horse.horse_config import (
    HorseCfg,
    HorseRunnerCfg,
)

BASE_HEIGHT_REF = 1.3


class HorseOscCfg(HorseCfg):
    class viewer:
        ref_env = 0
        pos = [-2.0, 3, 2]  # [m]
        lookat = [0.0, 1.0, 0.5]  # [m]

    class env(HorseCfg.env):
        num_envs = 2**12
        num_actuators = 1 + 4 * 5
        episode_length_s = 10
        env_spacing = 3.0

    class terrain(HorseCfg.terrain):
        mesh_type = "plane"
        # mesh_type = 'trimesh'  # none, plane, heightfield or trimesh

    class init_state(HorseCfg.init_state):
        timeout_reset_ratio = 0.75
        reset_mode = "reset_to_range"
        # * default COM for basic initialization
        pos = [0.0, 0.0, 1.4]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {
            "haa": 0.0,
            "hfe": 0.0,
            "kfe": 0.0,
            "pfe": 0.0,
            "pastern_to_foot": 0.0,
            "base_joint": 0.0,
        }

        # * initialization for random range setup
        # these are the physical limits in the URDF as of 17 Nov 2025
        # dof_pos_range = {
        #     "haa": [-0.2, 0.2],
        #     "hfe": [-0.7, 0.6],
        #     "kfe": [-1.3, 0.1],
        #     "pfe": [-0.3, 2.2],
        #     "pastern_to_foot": [-0.3, 1.8],
        #     "base_joint": [-0.2, 0.2],
        # }
        dof_pos_range = {
            "haa": [-0.2, 0.2],
            "hfe": [-0.7, 0.6],
            "kfe": [-1.3, 0.1],
            "pfe": [-0.3, 2.2],
            "pastern_to_foot": [-0.3, 1.8],
            "base_joint": [-0.2, 0.2],
        }
        dof_vel_range = {
            "haa": [-0.2, 0.2],
            "hfe": [-0.2, 0.2],
            "kfe": [-0.2, 0.2],
            "pfe": [-0.2, 0.2],
            "pastern_to_foot": [-0.2, 0.2],
            "base_joint": [-0.2, 0.2],
        }

        root_pos_range = [
            [0.0, 0.0],  # x
            [0.0, 0.0],  # y
            [1.3, 1.3],  # z
            [-0.2, 0.2],  # roll
            [-0.2, 0.2],  # pitch
            [-0.2, 0.2],  # yaw
        ]
        root_vel_range = [
            [-0.5, 5.0],  # x
            [0.0, 0.0],  # y
            [-0.05, 0.05],  # z
            [0.0, 0.0],  # roll
            [0.0, 0.0],  # pitch
            [0.0, 0.0],  # yaw
        ]

    class control(HorseCfg.control):
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
        ctrl_frequency = 250  # how often the PDF controller/action updates run
        desired_sim_frequency = 500  # how often the physics is calculated

    class osc:  # <-------------------most likely needs tuning
        process_noise_std = 0.25
        grf_threshold = 0.1  # 20. # Normalized to body weight
        # oscillator parameters
        omega = 3  # gets overwritten
        coupling = 1  # gets overwritten
        osc_bool = False  # not used in paper
        grf_bool = False  # not used in paper
        randomize_osc_params = False
        omega_range = [1.0, 4.0]  # [0.0, 10.]
        coupling_range = [0.0, 1.0]
        offset_range = [0.0, 0.0]
        stop_threshold = 0.5
        omega_stop = 1.0
        omega_step = 2.0
        omega_slope = 1.0
        omega_max = 4.0
        omega_var = 0.25
        # coupling_step = 0.
        # coupling_stop = 0.
        coupling_stop = 4.0
        coupling_step = 1.0
        coupling_slope = 0.0
        coupling_max = 1.0
        offset = 1.0
        coupling_var = 0.25

        init_to = "random"
        init_w_offset = True

    class commands:
        resampling_time = 3.0  # * time before command are changed[s]
        var = 1.0

        class ranges:
            lin_vel_x = [-1.0, 0.0, 1.0, 3.0]  # min max [m/s]
            lin_vel_y = 1.0  # max [m/s]
            yaw_vel = 6  # max [rad/s]
            height = [0.61, 1.30]  # m

    class push_robots:
        toggle = False
        interval_s = 1
        max_push_vel_xy = 0.5
        push_box_dims = [0.3, 0.1, 0.1]  # x,y,z [m]

    class domain_rand:
        randomize_friction = True
        friction_range = [0.4, 1.0]
        randomize_base_mass = False
        lower_mass_offset = -0.5  # kg
        upper_mass_offset = 2.0
        lower_z_offset = 0.0  # m
        upper_z_offset = 0.2
        lower_x_offset = 0.0
        upper_x_offset = 0.0

    class asset(HorseCfg.asset):
        shank_length_diff = 0  # Units in cm
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/" + "horse/urdf/horse.urdf"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "shank", "pastern"]
        terminate_after_contacts_on = ["base"]
        collapse_fixed_joints = False
        fix_base_link = False
        self_collisions = 1
        flip_visual_attachments = False
        disable_gravity = False
        disable_motors = False
        joint_damping = 0.3

    class reward_settings(HorseCfg.reward_settings):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 600.0
        base_height_target = BASE_HEIGHT_REF + 0.03
        tracking_sigma = 0.25
        switch_scale = 0.5

    class scaling(HorseCfg.scaling):
        base_ang_vel = [0.3, 0.3, 0.1]
        base_lin_vel = BASE_HEIGHT_REF
        dof_vel = [2.0] + (4 * [0.5, 0.5, 0.5, 0.5, 0.5])
        base_height = BASE_HEIGHT_REF
        dof_pos = [0.2] + (4 * [0.4, 1.3, 1.4, 2.5, 2.1])
        dof_pos_obs = dof_pos
        dof_pos_target = [2.0 * x for x in dof_pos]  # target joint angles
        tau_ff = [1100] + (4 * [1000, 1000, 1000, 500, 300])  # not being used
        commands = [3, 1, 3, 1]  # add height as a command


class HorseOscRunnerCfg(HorseRunnerCfg):
    seed = -1
    runner_class_name = "OnPolicyRunner"

    class actor(HorseRunnerCfg.actor):
        hidden_dims = [256, 256, 128]
        # * can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = "elu"
        # obs = [
        #     "base_height",
        #     "base_lin_vel",
        #     "base_ang_vel",
        #     "projected_gravity",
        #     "commands",
        #     "dof_pos_obs",
        #     "dof_vel",
        #     "dof_pos_target",
        # ]
        obs = [
            "base_height",
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_vel",
            "oscillator_obs",
            "dof_pos_target",
            #  "osc_omega",
            #  "osc_coupling"
            #  "oscillators_vel",
            #  "grf",
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

    class critic(HorseRunnerCfg.critic):
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
            "oscillator_obs",
            "oscillators_vel",
            "dof_pos_target",
        ]
        normalize_obs = True

        class reward:
            class weights:
                tracking_lin_vel = 4.0
                tracking_ang_vel = 2.0
                lin_vel_z = 0.0
                ang_vel_xy = 0.0
                orientation = 1.0
                torques = 5.0e-6
                dof_vel = 0.0
                min_base_height = 1.0
                collision = 0
                action_rate = 0.1  # -0.01
                action_rate2 = 0.01  # -0.001
                stand_still = 0.0
                dof_pos_limits = 0.0
                feet_contact_forces = 0.0
                dof_near_home = 0.0
                swing_grf = 1.0
                stance_grf = 1.0
                swing_velocity = 0.0
                stance_velocity = 0.0
                coupled_grf = 0.0  # 8.
                enc_pace = 0.0
                cursorial = 0.25
                standing_torques = 0.0  # 1.e-5

            class termination_weight:
                termination = 15.0 / 100.0

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 4
        # mini batch size = num_envs*nsteps/nminibatches
        num_mini_batches = 8
        max_gradient_steps = 32
        learning_rate = 1.0e-4
        schedule = "adaptive"  # can be adaptive, fixed
        discount_horizon = 1.0
        GAE_bootstrap_horizon = 2.0
        desired_kl = 0.01
        max_grad_norm = 1.0
        lr_range = [1e-5, 5e-3]
        lr_ratio = 1.5

    class runner(HorseRunnerCfg.runner):
        run_name = ""
        experiment_name = "horse_osc"
        max_iterations = 500
        algorithm_class_name = "PPO2"
        num_steps_per_env = 32
