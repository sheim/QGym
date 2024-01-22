import torch

from gym.envs.base.fixed_robot_config import FixedRobotCfg, FixedRobotCfgPPO


class PendulumCfg(FixedRobotCfg):
    class env(FixedRobotCfg.env):
        num_envs = 4096
        num_actuators = 1  # 1 for theta connecting base and pole
        episode_length_s = 10.0

    class terrain(FixedRobotCfg.terrain):
        pass

    class init_state(FixedRobotCfg.init_state):
        default_joint_angles = {"theta": 0.0}

        # * default setup chooses how the initial conditions are chosen.
        # * "reset_to_basic" = a single position
        # * "reset_to_range" = uniformly random from a range defined below
        reset_mode = "reset_to_range"

        # * initial conditions for reset_to_range
        dof_pos_range = {
            "theta": [-torch.pi, torch.pi],
        }

    class control(FixedRobotCfg.control):
        actuated_joints_mask = [1]  # angle
        ctrl_frequency = 250
        desired_sim_frequency = 500

    class asset(FixedRobotCfg.asset):
        # * Things that differ
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/" + "pendulum/urdf/pendulum.urdf"
        disable_gravity = False  # False
        disable_motors = False  # all torques set to 0

    class reward_settings(FixedRobotCfg.reward_settings):
        pass

    class scaling(FixedRobotCfg.scaling):
        theta = 2.0 * torch.pi
        omega = 100.0
        # * Action scales
        tau_ff = 0.5


class PendulumRunnerCfg(FixedRobotCfgPPO):
    # We need random experiments to run
    seed = -1

    class policy(FixedRobotCfgPPO.policy):
        actor_obs = [
            "dof_pos",
            "dof_vel",
        ]  # ! check markov chain

        critic_obs = actor_obs

        actions = ["tau_ff"]

        class noise:
            dof_pos = 0.0
            dof_vel = 0.0

        class reward:
            class weights:
                theta = 10.0
                omega = 1.0
                equilibrium = 1.0
                energy = 1.0

            class termination_weight:
                termination = 0.0

    class algorithm(FixedRobotCfgPPO.algorithm):
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 6
        # * mini batch size = num_envs*nsteps / nminibatches
        num_mini_batches = 4
        learning_rate = 1.0e-3
        schedule = "adaptive"  # could be adaptive, fixed
        discount_horizon = 1.0  # [s]
        GAE_bootstrap_horizon = 1.0  # [s]
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner(FixedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "pendulum"
        max_iterations = 500  # number of policy updates
        algorithm_class_name = "PPO"
        num_steps_per_env = 32
