"""Hardware-oriented VSim tuning configuration for ``mini_cheetah_ref``.

This is intentionally separate from the parity configuration.  Changes here
are selected using the native VSim hardware scorecard, not cross-backend reward
or trajectory agreement.
"""

from gym.envs.mini_cheetah.mini_cheetah_ref_config import (
    MiniCheetahRefCfg,
    MiniCheetahRefRunnerCfg,
)


class MiniCheetahRefVSimCfg(MiniCheetahRefCfg):
    class env(MiniCheetahRefCfg.env):
        # 2**15 samples / 256 envs = 128 policy steps (1.28 s), long enough
        # for three complete 2.5 Hz reference cycles in every PPO rollout.
        num_envs = 256

    class init_state(MiniCheetahRefCfg.init_state):
        root_vel_range = [
            [-0.5, 1.5],  # x [m/s]
            [-0.2, 0.2],  # y [m/s]
            [-0.05, 0.05],  # z [m/s]
            [-0.1, 0.1],  # roll [rad/s]
            [-0.1, 0.1],  # pitch [rad/s]
            [-0.2, 0.2],  # yaw [rad/s]
        ]

    class commands(MiniCheetahRefCfg.commands):
        # First tune for the intended deployment envelope. Faster running and
        # stronger pushes belong in a later robustness curriculum.
        resampling_time = 3.0
        # Independent x/y/yaw sampling almost never shows the policy a pure
        # strafe or yaw command. Explicitly make most training commands
        # axis-aligned while retaining combined commands in the remaining 40%.
        axis_aligned_fraction = 0.6

        class ranges:
            lin_vel_x = [-0.5, 1.5]
            lin_vel_y = 0.4
            yaw_vel = 0.75

    class push_robots(MiniCheetahRefCfg.push_robots):
        toggle = False

    class domain_rand(MiniCheetahRefCfg.domain_rand):
        # VSim currently creates one shared rigid material and articulation
        # definition, so the legacy per-environment randomizers are not
        # applied. Keep this explicit until VSim-native randomization exists.
        randomize_friction = False
        randomize_base_mass = False

    class reward_settings(MiniCheetahRefCfg.reward_settings):
        # Preserve 7.5% of each URDF joint range for commanded position
        # targets. Actual state limits remain the URDF limits.
        soft_dof_pos_limit = 0.85
        # The inherited 2.5 rad/s normalization made ±0.75 rad/s deployment
        # commands nearly indistinguishable from zero yaw.
        tracking_ang_vel_scale = 1.0


class MiniCheetahRefVSimRunnerCfg(MiniCheetahRefRunnerCfg):
    seed = 7

    class actor(MiniCheetahRefRunnerCfg.actor):
        # The baseline's learned exploration standard deviation grew above
        # 1.3. Start lower and reduce entropy pressure while keeping the same
        # deterministic actor architecture.
        init_noise_std = 0.5

    class critic(MiniCheetahRefRunnerCfg.critic):
        class reward:
            class weights:
                tracking_lin_vel = 6.0
                # A bounded continuation at 10.0 did not materially improve
                # turning and degraded gait quality, so retain the smoother
                # iteration-400 setting. Turning needs a targeted follow-up.
                tracking_ang_vel = 4.0

                # Target-independent body steadiness: MiniCheetah implements
                # these as bounded rewards for low vertical and roll/pitch
                # velocity, so no exact base-height target is imposed.
                lin_vel_z = 0.25
                ang_vel_xy = 0.10
                orientation = 1.5
                min_base_height = 1.0

                # Smooth, hardware-conscious actuation proxies. FFT metrics
                # remain evaluation-only because they require a time window.
                torques = 2.0e-5
                dof_vel = 0.02
                action_rate = 0.20
                action_rate2 = 0.10
                dof_pos_target_limits = 5.0
                dof_vel_limits = 0.10
                torque_limits = 0.25

                # Favor a four-leg alternating trot over the baseline's
                # RF/LH-only solution at 1 m/s.
                reference_traj = 4.0
                swing_grf = 3.0
                stance_grf = 3.0

                stand_still = 1.0
                collision = 0.25
                dof_pos_limits = 0.0
                feet_contact_forces = 0.0
                dof_near_home = 0.0

            class termination_weight:
                # Termination is integrated at the 2 ms physics rate. A weight
                # of 100 is a 0.2-return penalty, unlike the effectively-zero
                # 0.15 baseline weight.
                termination = 100.0

    class algorithm(MiniCheetahRefRunnerCfg.algorithm):
        batch_size = 2**15
        entropy_coef = 0.003

    class runner(MiniCheetahRefRunnerCfg.runner):
        experiment_name = "mini_cheetah_ref_vsim_tune"
        max_iterations = 500
