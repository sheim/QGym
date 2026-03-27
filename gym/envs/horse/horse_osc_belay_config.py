from gym.envs.horse.horse_osc_config import (
    HorseOscCfg,
    HorseOscRunnerCfg,
)


class HorseOscBelayCfg(HorseOscCfg):
    class belay:
        enabled = True
        body_name = "front_base"
        mass_kg = 20.0
        force_n = mass_kg * 9.81
        keyboard_toggle = True
        toggle_all_envs = True
        start_enabled = True
        debug_print = True


class HorseOscBelayRunnerCfg(HorseOscRunnerCfg):
    class runner(HorseOscRunnerCfg.runner):
        experiment_name = "horse_osc_belay"
