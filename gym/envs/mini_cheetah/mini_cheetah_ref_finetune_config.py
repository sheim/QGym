from gym.envs.mini_cheetah.mini_cheetah_ref_config import (
    MiniCheetahRefCfg,
    MiniCheetahRefRunnerCfg,
)

BASE_HEIGHT_REF = 0.33


class MiniCheetahRefFinetuneCfg(MiniCheetahRefCfg):
    class env(MiniCheetahRefCfg.env):
        num_envs = 1
        num_actuators = 12
        episode_length_s = 20.0

    class terrain(MiniCheetahRefCfg.terrain):
        pass

    class init_state(MiniCheetahRefCfg.init_state):
        pass

    class control(MiniCheetahRefCfg.control):
        # * PD Drive parameters:
        stiffness = {"haa": 20.0, "hfe": 20.0, "kfe": 20.0}
        damping = {"haa": 0.5, "hfe": 0.5, "kfe": 0.5}
        gait_freq = 3.0
        ctrl_frequency = 50
        desired_sim_frequency = 500

    class commands(MiniCheetahRefCfg.commands):
        pass

    class push_robots(MiniCheetahRefCfg.push_robots):
        pass

    class domain_rand(MiniCheetahRefCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.0]
        randomize_base_mass = True
        added_mass_range = [5, 5]

    class asset(MiniCheetahRefCfg.asset):
        pass

    class reward_settings(MiniCheetahRefCfg.reward_settings):
        pass

    class scaling(MiniCheetahRefCfg.scaling):
        pass


class MiniCheetahRefFinetuneRunnerCfg(MiniCheetahRefRunnerCfg):
    seed = -1
    runner_class_name = "IPGRunner"

    class actor:
        hidden_dims = [256, 256, 128]
        # * can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = "elu"
        obs = [
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_vel",
            "phase_obs",
        ]
        normalize_obs = True

        actions = ["dof_pos_target"]
        disable_actions = False

        class noise:
            noise_multiplier = 10.0  # Fine tuning: multiplies all noise

            scale = 1.0
            dof_pos_obs = 0.01
            base_ang_vel = 0.01
            dof_pos = 0.005
            dof_vel = 0.005
            lin_vel = 0.05
            ang_vel = [0.3, 0.15, 0.4]
            gravity_vec = 0.1

    class critic:
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
            "phase_obs",
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
                reference_traj = 0.0  # 1.5
                swing_grf = 1.5
                stance_grf = 1.5

            class termination_weight:
                termination = 0.15

    class algorithm(MiniCheetahRefRunnerCfg.algorithm):
        # both
        gamma = 0.99
        lam = 0.95
        # shared
        batch_size = 1000  # same as num_steps_per_env
        max_gradient_steps = 10
        # new
        storage_size = 8 * 1000 * 1  # policies * steps * envs

        clip_param = 0.2
        learning_rate = 1.0e-4
        max_grad_norm = 1.0
        # Critic
        use_clipped_value_loss = True
        # Actor
        entropy_coef = 0.01
        schedule = "fixed"  # could be adaptive, fixed
        desired_kl = 0.01

        # GePPO
        vtrace = True
        normalize_advantages = False  # weighted normalization in GePPO loss
        recursive_advantages = True  # applies to vtrace
        is_trunc = 1.0

        # IPG
        polyak = 0.995
        use_cv = False  # control variate
        inter_nu = 0.2
        beta = "off_policy"

    class runner(MiniCheetahRefRunnerCfg.runner):
        run_name = ""
        experiment_name = "mini_cheetah_ref"
        max_iterations = 30  # number of policy updates
        algorithm_class_name = "PPO_IPG"
        num_steps_per_env = 1000  # deprecate
        num_old_policies = 4

        # Fine tuning
        resume = True
        load_run = "Jul03_09-45-30_IPG_4096_32_8"
