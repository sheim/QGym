from gym.envs.horse.horse_osc_config import (
    HorseOscCfg,
    HorseOscRunnerCfg,
)


class HorseOscBelayCfg(HorseOscCfg):
    class belay:
        body_name = "front_base"
        mass_kg = 20.0
        force_n = mass_kg * 9.81
        keyboard_toggle = True
        toggle_all_envs = True
        start_enabled = True
        debug_print = True
        anchor_height = 2.0

    class perturbations:
        enabled = False

        reduced_torque_enabled = False
        torque_scale = 1.0  # 1.0 normal, 0.5 = 50% strength

        latency_enabled = False
        latency_steps = 0  # number of control steps delay
        max_latency_steps = 10

        motor_noise_enabled = False
        motor_noise_std = 0.0


class HorseOscBelayRunnerCfg(HorseOscRunnerCfg):
    class runner(HorseOscRunnerCfg.runner):
        experiment_name = "horse_osc_belay"
