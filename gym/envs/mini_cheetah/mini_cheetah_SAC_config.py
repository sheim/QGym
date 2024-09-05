from gym.envs.base.legged_robot_config import LeggedRobotRunnerCfg
from gym.envs.mini_cheetah.mini_cheetah_ref_config import MiniCheetahRefCfg

BASE_HEIGHT_REF = 0.3


class MiniCheetahSACCfg(MiniCheetahRefCfg):
    class env(MiniCheetahRefCfg.env):
        num_envs = 1
        episode_length_s = 4  # TODO

    class terrain(MiniCheetahRefCfg.terrain):
        pass

    class init_state(MiniCheetahRefCfg.init_state):
        pass

    class control(MiniCheetahRefCfg.control):
        # * PD Drive parameters:
        stiffness = {"haa": 20.0, "hfe": 20.0, "kfe": 20.0}
        damping = {"haa": 0.5, "hfe": 0.5, "kfe": 0.5}
        gait_freq = 3.0
        ctrl_frequency = 20  # TODO
        desired_sim_frequency = 100

    class commands(MiniCheetahRefCfg.commands):
        pass

    class push_robots(MiniCheetahRefCfg.push_robots):
        pass

    class domain_rand(MiniCheetahRefCfg.domain_rand):
        pass

    class asset(MiniCheetahRefCfg.asset):
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
        joint_damping = 0.1
        rotor_inertia = [0.002268, 0.002268, 0.005484] * 4

    class reward_settings(MiniCheetahRefCfg.reward_settings):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 600.0
        base_height_target = BASE_HEIGHT_REF
        tracking_sigma = 0.25

    class scaling(MiniCheetahRefCfg.scaling):
        base_ang_vel = 0.3
        base_lin_vel = BASE_HEIGHT_REF
        dof_vel = 4 * [2.0, 2.0, 4.0]
        base_height = 0.3
        dof_pos = 4 * [0.2, 0.3, 0.3]
        dof_pos_obs = dof_pos
        dof_pos_target = 4 * [0.2, 0.3, 0.3]
        tau_ff = 4 * [18, 18, 28]
        commands = [3, 1, 3]


class MiniCheetahSACRunnerCfg(LeggedRobotRunnerCfg):
    seed = -1
    runner_class_name = "OffPolicyRunner"

    class actor(LeggedRobotRunnerCfg.actor):
        hidden_dims = {
            "latent": [128, 128],
            "mean": [64],
            "std": [64],
        }
        # * can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = {
            "latent": "elu",
            "mean": "elu",
            "std": "elu",
        }

        # TODO[lm]: Handle normalization
        normalize_obs = False
        obs = [
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_vel",
            "phase_obs",
        ]

        actions = ["dof_pos_target"]
        add_noise = False  # TODO
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
        hidden_dims = [128, 128, 64]
        # * can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = "elu"

        # TODO[lm]: Handle normalization
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
                reference_traj = 1.5
                swing_grf = 1.5
                stance_grf = 1.5

            class termination_weight:
                termination = 0.15

    class algorithm(LeggedRobotRunnerCfg.algorithm):
        # Taken from SAC pendulum
        initial_fill = 500
        storage_size = 10**6
        batch_size = 256
        max_gradient_steps = 1  # 10
        action_max = 1.0  # TODO
        action_min = -1.0  # TODO
        actor_noise_std = 0.5  # TODO
        log_std_max = 4.0
        log_std_min = -20.0
        alpha = 0.2
        target_entropy = -12.0  # -action_dim
        max_grad_norm = 1.0
        polyak = 0.995  # flipped compared to SB3 (polyak == 1-tau)
        gamma = 0.99
        alpha_lr = 1e-4
        actor_lr = 1e-4
        critic_lr = 1e-4

    class runner(LeggedRobotRunnerCfg.runner):
        run_name = ""
        experiment_name = "sac_mini_cheetah"
        max_iterations = 50_000
        algorithm_class_name = "SAC"
        save_interval = 10_000
        num_steps_per_env = 1
