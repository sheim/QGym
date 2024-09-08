from gym.envs.pendulum.pendulum_SAC_config import PendulumSACCfg, PendulumSACRunnerCfg


class PendulumPSDCfg(PendulumSACCfg):
    class env(PendulumSACCfg.env):
        pass

    class init_state(PendulumSACCfg.init_state):
        pass

    class control(PendulumSACCfg.control):
        pass

    class asset(PendulumSACCfg.asset):
        pass

    class reward_settings(PendulumSACCfg.reward_settings):
        pass

    class scaling(PendulumSACCfg.scaling):
        pass


class PendulumPSDRunnerCfg(PendulumSACRunnerCfg):
    seed = -1
    runner_class_name = "PSACRunner"

    class actor(PendulumSACRunnerCfg.actor):
        pass

    class critic(PendulumSACRunnerCfg.critic):
        obs = ["dof_pos_obs", "dof_vel"]
        hidden_dims = [16, 16]
        # * can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        dropouts = [0.1, 0.0]
        layer_norm = [True, True]
        activation = "elu"
        # TODO[lm]: Current normalization uses torch.no_grad, this should be changed
        normalize_obs = False

        # critic_class_name = "CholeskyLatent"
        # critic_class_name = "Critic"
        critic_class_name = "DenseSpectralLatent"
        # * some class-specific params
        minimize = False
        relative_dim = 4  # 16
        latent_dim = 4  # 18,
        latent_hidden_dims = [16]
        latent_activation = "elu"

    class algorithm(PendulumSACRunnerCfg.algorithm):
        alpha_lr = 1e-4
        actor_lr = 1e-4
        critic_lr = 1e-4

    class runner(PendulumSACRunnerCfg.runner):
        run_name = ""
        experiment_name = "psd_pendulum"
        algorithm_class_name = "SAC"
        num_steps_per_env = 1
        save_interval = 500
        log_storage = True
