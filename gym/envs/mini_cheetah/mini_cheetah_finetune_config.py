from gym.envs.mini_cheetah.mini_cheetah_ref_config import (
    MiniCheetahRefCfg,
    MiniCheetahRefRunnerCfg,
)

BASE_HEIGHT_REF = 0.33


class MiniCheetahFineTuneCfg(MiniCheetahRefCfg):
    class env(MiniCheetahRefCfg.env):
        num_envs = 4096
        num_actuators = 12
        episode_length_s = 30.0

    class terrain(MiniCheetahRefCfg.terrain):
        pass

    class init_state(MiniCheetahRefCfg.init_state):
        pass

    class control(MiniCheetahRefCfg.control):
        # * PD Drive parameters:
        stiffness = {"haa": 20.0, "hfe": 20.0, "kfe": 20.0}
        damping = {"haa": 0.5, "hfe": 0.5, "kfe": 0.5}
        gait_freq = 3.0
        ctrl_frequency = 100
        desired_sim_frequency = 500

    class commands(MiniCheetahRefCfg.commands):
        pass

    class push_robots(MiniCheetahRefCfg.push_robots):
        pass

    class domain_rand(MiniCheetahRefCfg.domain_rand):
        pass

    class asset(MiniCheetahRefCfg.asset):
        pass

    class reward_settings(MiniCheetahRefCfg.reward_settings):
        pass

    class scaling(MiniCheetahRefCfg.scaling):
        pass


class MiniCheetahFineTuneRunnerCfg(MiniCheetahRefRunnerCfg):
    seed = -1
    runner_class_name = "IPGRunner"

    class actor:
        hidden_dims = [256, 256, 128]
        # * can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = "elu"
        smooth_exploration = True
        exploration_sample_freq = 16

        normalize_obs = True
        obs = [
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_vel",
            "phase_obs",
        ]

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
        hidden_dims = [256, 256, 128]
        # * can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = "elu"

        # TODO: Check normalization, SAC/IPG need gradient to pass back through actor
        normalize_obs = False
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

        class reward:
            class weights:
                tracking_lin_vel = 4.0
                tracking_ang_vel = 2.0
                orientation = 1.0
                min_base_height = 1.5
                stand_still = 2.0
                swing_grf = 3.0
                stance_grf = 3.0
                action_rate = 0.01
                action_rate2 = 0.001

            class termination_weight:
                termination = 0.15

    class state_estimator:
        class network:
            hidden_dims = [128, 128]
            activation = "tanh"
            dropouts = None

        obs = [
            "base_ang_vel",
            "projected_gravity",
            "dof_pos_obs",
            "dof_vel",
            "torques",
            "phase_obs",
        ]
        targets = ["base_height", "base_lin_vel", "grf"]
        normalize_obs = True

    class algorithm(MiniCheetahRefRunnerCfg.algorithm):
        desired_kl = 0.02  # 0.02 for smooth-exploration, else 0.01

        # IPG
        polyak = 0.995
        use_cv = False
        inter_nu = 0.9
        beta = "off_policy"
        storage_size = 30000

        # Finetuning
        clip_param = 0.2
        max_gradient_steps = 4
        batch_size = 30000
        learning_rate = 1e-4
        schedule = "fixed"

    class runner(MiniCheetahRefRunnerCfg.runner):
        run_name = ""
        experiment_name = "mini_cheetah_ref"
        max_iterations = 20  # number of policy updates
        algorithm_class_name = "PPO_IPG"
        num_steps_per_env = 32

        # Finetuning
        resume = True
        load_run = "Jul23_00-14-23_nu02_B8"
        checkpoint = 1000
        save_interval = 1
