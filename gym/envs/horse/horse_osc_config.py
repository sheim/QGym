from gym.envs.horse.horse_config import (
    HorseCfg,
    HorseRunnerCfg,
)

BASE_HEIGHT_REF = 1.0


class HorseOscCfg(HorseCfg):
    class env(HorseCfg.env):
        num_envs = 2**12
        num_actuators = 1 + 4 * 5
        episode_length_s = 10

    class terrain(HorseCfg.terrain):
        mesh_type = "plane"
        # mesh_type = 'trimesh'  # none, plane, heightfield or trimesh

    class init_state(HorseCfg.init_state):
        timeout_reset_ratio = 0.75
        reset_mode = "reset_to_range"
        # * default COM for basic initialization
        pos = [0.0, 0.0, 0.6]  # x,y,z [m]
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
        # these are the physical limits in the URDF as of 11 Dec 2025
        # dof_pos_range = {
        #   haa": [-0.2, 0.2],
        #   f_hfe": [-1.0, 0.6],
        #   h_hfe": [-1.5, 0.5],
        #   f_kfe": [-1.5, 0.1],
        #   h_kfe": [-0.2, 1.0],
        #   f_pfe": [-0.3, 3.0],
        #   h_pfe": [-1.2, 2.5],
        #   f_pastern_to_foot": [-0.3, 1.8],
        #   h_pastern_to_foot": [-0.3, 1.8],
        #   base_joint": [-0.2, 0.2],
        # }
        # dof_pos_range = {
        #     "haa": [-0.2, 0.2],
        #     "hfe": [-1.0, 0.5],
        #     "kfe": [-0.2, 0.1],
        #     "pfe": [-0.3, 2.5],
        #     "pastern_to_foot": [-0.3, 1.8],
        #     "base_joint": [-0.0, 0.0],
        # }
        dof_pos_range = {
            "haa": [-0.2, 0.2],
            "h_hfe": [-1.5, -1.5],
            "f_hfe": [-1.0, -1.0],
            "h_kfe": [1.0, 1.0],
            "f_kfe": [-1.5, -1.5],
            "h_pfe": [-1.2, -1.2],
            "f_pfe": [1.5, 1.5],
            "pastern_to_foot": [-0.3, 1.8],
            "base_joint": [-0.0, 0.0],
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
            [0.6, 0.6],  # z
            [-0.0, 0.0],  # roll
            [-0.0, 0.0],  # pitch
            [-0.2, 0.2],  # yaw
        ]
        root_vel_range = [
            [0.0, 0.0],  # x
            [0.0, 0.0],  # y
            [0.0, 0.0],  # z
            [0.0, 0.0],  # roll
            [0.0, 0.0],  # pitch
            [0.0, 0.0],  # yaw
        ]

    class control(HorseCfg.control):
        # * PD Drive parameters:
        # HorseOscCfg.control
        stiffness = {
            "haa": 75,
            "hfe": 40,
            "kfe": 38,
            "pfe": 10,
            "pastern_to_foot": 5,
            "base_joint": 50,  # still needs modifying
        }
        damping = {
            "haa": 0.001,
            "hfe": 0.001,
            "kfe": 0.001,
            "pfe": 0.5,
            "pastern_to_foot": 0.5,
            "base_joint": 10,
        }
        ctrl_frequency = 250  # how often the PDF controller/action updates run
        desired_sim_frequency = 1000  # how often the physics is calculated

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
        # coupling_step = 0.
        # coupling_stop = 0.
        coupling_stop = 4.0
        coupling_step = 1.0
        coupling_slope = 0.0
        coupling_max = 1.0
        offset = 1.0

        init_to = "random"

    class commands:
        resampling_time = 3.0  # * time before command are changed[s]
        var = 1.0

        class ranges:
            lin_vel_x = [0.0, 0.0]  # min max [m/s]
            lin_vel_y = 0.0  # max [m/s]
            yaw_vel = 0  # max [rad/s]
            height = [0.6, 1.0]  # m

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
        penalize_contacts_on = ["thigh"]  # "thigh", "shank", "pastern"
        terminate_after_contacts_on = ["top"]
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
        max_contact_force = 3000.0  # testing, this was too low for a horse weight
        base_height_target = 1.0 + 0.03
        tracking_sigma = 0.25
        switch_scale = 0.5
        switch_scale_height = 0.05  # drops to near 0 when cmd_h is 0.6

    class scaling(HorseCfg.scaling):
        base_ang_vel = [0.3, 0.3, 0.1]
        base_lin_vel = BASE_HEIGHT_REF
        dof_vel = [2.0] + (4 * [0.5, 0.5, 0.5, 0.5, 0.5])
        base_height = BASE_HEIGHT_REF
        dof_pos = [0.2] + (
            4 * [0.4, 1.3, 1.4, 2.5, 2.1]
        )  # reducing this to be alot smaller 2.1 or 2.5
        dof_pos_obs = dof_pos
        # dof_pos_target = [2.0 * x for x in dof_pos]  # target joint angles
        dof_pos_target = [0.4] + (4 * [0.8, 2.6, 2.8, 5.0, 4.2])
        tau_ff = [1100] + (4 * [1000, 1000, 1000, 500, 300])  # not being used
        commands = [3, 1, 3, 1]  # add height as a command


class HorseOscRunnerCfg(HorseRunnerCfg):
    seed = -1
    runner_class_name = "OnPolicyRunner"

    class actor(HorseRunnerCfg.actor):
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
                ang_vel_xy = 0.01
                orientation = 4.0
                torques = 5.0e-10
                dof_vel = 0.0
                min_base_height = 0.0
                action_rate = 1e-5
                action_rate2 = 1e-6
                stand_still = 0.0
                dof_pos_limits = 0.0
                feet_contact_forces = 0.0
                dof_near_home = 0.0
                swing_grf = 5.0
                stance_grf = 5.0
                swing_velocity = 0.0
                stance_velocity = 0.0
                coupled_grf = 1.0  # penalize for grf during swing, no grf during stance
                enc_pace = 0.0
                cursorial = 1.0  # enourage legs to stay under body, don't splay out
                standing_torques = 0.0  # 1.e-5
                tracking_height = 10.0
                hind_kfe_tendon = 2.0
                hind_pfe_tendon = 2.0
                front_kfe_tendon = 3
                front_pfe_tendon = 3
                feet_contact_count = 3
                feet_support_upright = 10
                hfe_upright = 10

            class termination_weight:
                termination = 50.0 / 100.0

    class algorithm:
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
        lr_range = [2e-4, 1e-2]
        lr_ratio = 1.3

    class runner(HorseRunnerCfg.runner):
        run_name = ""
        experiment_name = "horse_osc"
        max_iterations = 1000
        algorithm_class_name = "PPO2"
        num_steps_per_env = 32
