from gym.envs.go2.go2_config import Go2Cfg, Go2RunnerCfg


class Go2TrotCfg(Go2Cfg):
    class env(Go2Cfg.env):
        episode_length_s = 5

    class init_state(Go2Cfg.init_state):
        # The gait reference supplies the nominal posture; residual targets
        # therefore use zero joint offsets here.
        default_joint_angles = {
            "hip_joint": 0.0,
            "calf_joint": 0.0,
            "FL_thigh_joint": 0.0,
            "FR_thigh_joint": 0.0,
            "RL_thigh_joint": 0.0,
            "RR_thigh_joint": 0.0,
        }
        reset_mode = "reset_to_basic"

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

    class control(Go2Cfg.control):
        stiffness = {"hip": 20.0, "thigh": 20.0, "calf": 20.0}
        damping = {"hip": 0.5, "thigh": 0.5, "calf": 0.5}
        ctrl_frequency = 100
        desired_sim_frequency = 100
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

    class commands(Go2Cfg.commands):
        var = 1.0

        class ranges(Go2Cfg.commands.ranges):
            lin_vel_x = [-1.0, 0.0, 1.0, 3.0]

    class push_robots(Go2Cfg.push_robots):
        toggle = True
        interval_s = 5

    class domain_randomization(Go2Cfg.domain_randomization):
        class startup(Go2Cfg.domain_randomization.startup):
            link_mass_scale_range = [0.9, 1.2]

        class episode(Go2Cfg.domain_randomization.episode):
            scale_ranges = {
                "p_gains": [0.9, 1.1],
                "d_gains": [0.8, 1.2],
            }

    class asset(Go2Cfg.asset):
        penalize_contacts_on = ["calf", "hip"]
        terminate_after_contacts_on = ["base", "Head_upper", "Head_lower"]

    class reward_settings(Go2Cfg.reward_settings):
        base_height_target = 0.9 * Go2Cfg.reward_settings.base_height_target

    class scaling(Go2Cfg.scaling):
        # Canonical RobotLayout order is FL, FR, RL, RR, with
        # hip, thigh, calf inside each leg. Backends map native order to it.
        base_height = 0.3
        dof_pos = 4 * [1.0472, 2.53075, 0.94247]
        dof_pos_obs = dof_pos
        dof_pos_target = [0.5 * x for x in dof_pos]
        tau_ff = 4 * [23.7, 23.7, 45.43]


class Go2TrotRunnerCfg(Go2RunnerCfg):
    class actor(Go2RunnerCfg.actor):
        obs = Go2RunnerCfg.actor.obs + ["phase_obs", "phase_frequency"]
        add_noise = True

    class critic(Go2RunnerCfg.critic):
        obs = Go2RunnerCfg.critic.obs + ["phase_obs", "phase_frequency"]
        normalize_obs = False

        class reward(Go2RunnerCfg.critic.reward):
            class weights(Go2RunnerCfg.critic.reward.weights):
                min_base_height = 0.5
                action_rate = 0.25
                action_rate2 = 0.025
                # Preserve the old combined term's approximate +/-0.625 range,
                # while making both stance feet necessary for positive credit.
                trot_support = 0.625
                swing_contact = 1.25

    class algorithm(Go2RunnerCfg.algorithm):
        rollout_size = 2**16
        max_gradient_steps = 32

    class runner(Go2RunnerCfg.runner):
        experiment_name = "go2trot"
        max_iterations = 550
