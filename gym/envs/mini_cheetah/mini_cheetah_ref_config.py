from gym.envs.mini_cheetah.mini_cheetah_config import (
    MiniCheetahCfg,
    MiniCheetahRunnerCfg,
)

BASE_HEIGHT_REF = 0.33


class MiniCheetahRefCfg(MiniCheetahCfg):
    class env(MiniCheetahCfg.env):
        num_envs = 4096
        num_actuators = 12
        episode_length_s = 5.0

    class terrain(MiniCheetahCfg.terrain):
        pass

    class init_state(MiniCheetahCfg.init_state):
        ref_traj = (
            "{LEGGED_GYM_ROOT_DIR}/resources/robots/"
            + "mini_cheetah/trajectories/single_leg.csv"
        )

    class control(MiniCheetahCfg.control):
        # * PD Drive parameters:
        stiffness = {"haa": 20.0, "hfe": 20.0, "kfe": 20.0}
        damping = {"haa": 0.5, "hfe": 0.5, "kfe": 0.5}
        gait_freq = 2.5
        ctrl_frequency = 100
        desired_sim_frequency = 300

    class commands(MiniCheetahCfg.commands):
        pass

    class push_robots(MiniCheetahCfg.push_robots):
        pass

    class domain_rand(MiniCheetahCfg.domain_rand):
        pass

    class asset(MiniCheetahCfg.asset):
        file = (
            "{LEGGED_GYM_ROOT_DIR}/resources/robots/"
            + "mini_cheetah/urdf/mini_cheetah_simple.urdf"
        )
        foot_name = "foot"
        penalize_contacts_on = ["shank"]
        terminate_after_contacts_on = ["base", "thigh"]
        collapse_fixed_joints = False
        fix_base_link = False
        self_collisions = 1
        flip_visual_attachments = False
        disable_gravity = False
        disable_motors = False

    class reward_settings(MiniCheetahCfg.reward_settings):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 600.0
        base_height_target = 0.3
        tracking_sigma = 0.25

    class scaling(MiniCheetahCfg.scaling):
        pass


class MiniCheetahRefRunnerCfg(MiniCheetahRunnerCfg):
    seed = -1
    runner_class_name = "MyRunner"

    class actor(MiniCheetahRunnerCfg.actor):
        hidden_dims = [256, 256, 128]
        # * can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        smooth_exploration = False
        # exploration_sample_freq = 16
        activation = ["elu", "elu", "tanh"]
        obs = [
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_vel",
            "phase_obs",
            "dof_pos_target",
        ]
        normalize_obs = False

        actions = ["dof_pos_target"]
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

    class critic:
        hidden_dims = [256, 128]
        # * can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = ["elu", "tanh"]
        obs = [
            "base_height",
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_vel",
            "phase_obs",
            "dof_pos_target",
        ]
        normalize_obs = False
        # critic_class_name = "Critic"  # "DenseSpectralLatent"
        critic_class_name = "DenseSpectralLatent"
        # * some class-specific params
        minimize = False
        relative_dim = 4  # 16
        latent_dim = 12  # 18,
        latent_hidden_dims = [256, 256]
        latent_activation = "elu"

        class reward:
            class weights:
                tracking_lin_vel = 4.0
                tracking_ang_vel = 2.0
                lin_vel_z = 0.0
                ang_vel_xy = 0.01
                orientation = 1.0
                torques = 5.0e-7
                dof_vel = 0.0
                min_base_height = 1.5
                collision = 0.0
                action_rate = 0.01
                action_rate2 = 0.001
                stand_still = 0.0
                dof_pos_limits = 0.0
                feet_contact_forces = 0.0
                dof_near_home = 0.0
                reference_traj = 0.0
                swing_grf = 1.5
                stance_grf = 1.5

            class termination_weight:
                termination = 0.15

    class algorithm(MiniCheetahRunnerCfg.algorithm):
        # both
        gamma = 0.99
        lam = 0.75
        # shared
        batch_size = 2**15
        max_gradient_steps = 36
        # new
        storage_size = 2**17  # new
        batch_size = 2**15  #  new

        clip_param = 0.2
        learning_rate = 1.0e-2
        max_grad_norm = 1.0
        # Critic
        use_clipped_value_loss = True
        # Actor
        entropy_coef = 0.01
        schedule = "adaptive"  # could be adaptive, fixed
        desired_kl = 0.01
        lr_range = [2e-5, 1e-2]
        lr_ratio = 1.5

    class runner(MiniCheetahRunnerCfg.runner):
        run_name = ""
        experiment_name = "mini_cheetah_ref"
        max_iterations = 500  # number of policy updates
        algorithm_class_name = "PPO2"
        num_steps_per_env = 20  # deprecate
