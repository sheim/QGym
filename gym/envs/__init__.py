import importlib

from gym.utils.task_registry import task_registry

# * To add a new env:
# * 1. add the base env and env class name and location to the class dict
# * 2. add the config name and location to the config dict
# * 3. add the runner confg name and location to the runner config dict
# * 3. register the task experiment name to the env/config/ppo classes

class_dict = {
    "LeggedRobot": ".base.legged_robot",
    "FixedRobot": ".base.fixed_robot",
    "Cartpole": ".cartpole.cartpole",
    "MiniCheetah": ".mini_cheetah.mini_cheetah",
    "MiniCheetahRef": ".mini_cheetah.mini_cheetah_ref",
    "MiniCheetahOsc": ".mini_cheetah.mini_cheetah_osc",
    "MIT_Humanoid": ".mit_humanoid.mit_humanoid",
    "HumanoidRunning": ".mit_humanoid.humanoid_running",
    "Pendulum": ".pendulum.pendulum",
    "Go2": ".go2.go2",
}

config_dict = {
    "CartpoleCfg": ".cartpole.cartpole_config",
    "MiniCheetahCfg": ".mini_cheetah.mini_cheetah_config",
    "MiniCheetahRefCfg": ".mini_cheetah.mini_cheetah_ref_config",
    "MiniCheetahRefVSimCfg": ".mini_cheetah.mini_cheetah_ref_vsim_config",
    "MiniCheetahOscCfg": ".mini_cheetah.mini_cheetah_osc_config",
    "MiniCheetahSACCfg": ".mini_cheetah.mini_cheetah_SAC_config",
    "MITHumanoidCfg": ".mit_humanoid.mit_humanoid_config",
    "HumanoidRunningCfg": ".mit_humanoid.humanoid_running_config",
    "PendulumCfg": ".pendulum.pendulum_config",
    "PendulumSACCfg": ".pendulum.pendulum_SAC_config",
    "PendulumPSDCfg": ".pendulum.pendulum_PSD_config",
    "Go2Cfg": ".go2.go2_config",
}

runner_config_dict = {
    "CartpoleRunnerCfg": ".cartpole.cartpole_config",
    "MiniCheetahRunnerCfg": ".mini_cheetah.mini_cheetah_config",
    "MiniCheetahRefRunnerCfg": ".mini_cheetah.mini_cheetah_ref_config",
    "MiniCheetahRefVSimRunnerCfg": ".mini_cheetah.mini_cheetah_ref_vsim_config",
    "MiniCheetahOscRunnerCfg": ".mini_cheetah.mini_cheetah_osc_config",
    "MiniCheetahSACRunnerCfg": ".mini_cheetah.mini_cheetah_SAC_config",
    "MITHumanoidRunnerCfg": ".mit_humanoid.mit_humanoid_config",
    "HumanoidRunningRunnerCfg": ".mit_humanoid.humanoid_running_config",
    "PendulumRunnerCfg": ".pendulum.pendulum_config",
    "PendulumSACRunnerCfg": ".pendulum.pendulum_SAC_config",
    "PendulumPSDRunnerCfg": ".pendulum.pendulum_PSD_config",
    "Go2RunnerCfg": ".go2.go2_config",
}

task_dict = {
    "cartpole": ["Cartpole", "CartpoleCfg", "CartpoleRunnerCfg"],
    "mini_cheetah": ["MiniCheetah", "MiniCheetahCfg", "MiniCheetahRunnerCfg"],
    "mini_cheetah_ref": [
        "MiniCheetahRef",
        "MiniCheetahRefCfg",
        "MiniCheetahRefRunnerCfg",
    ],
    "mini_cheetah_ref_vsim": [
        "MiniCheetahRef",
        "MiniCheetahRefVSimCfg",
        "MiniCheetahRefVSimRunnerCfg",
    ],
    "mini_cheetah_osc": [
        "MiniCheetahOsc",
        "MiniCheetahOscCfg",
        "MiniCheetahOscRunnerCfg",
    ],
    "sac_mini_cheetah": [
        "MiniCheetahRef",
        "MiniCheetahSACCfg",
        "MiniCheetahSACRunnerCfg",
    ],
    "humanoid": ["MIT_Humanoid", "MITHumanoidCfg", "MITHumanoidRunnerCfg"],
    "humanoid_running": [
        "HumanoidRunning",
        "HumanoidRunningCfg",
        "HumanoidRunningRunnerCfg",
    ],
    "pendulum": ["Pendulum", "PendulumCfg", "PendulumRunnerCfg"],
    "sac_pendulum": ["Pendulum", "PendulumSACCfg", "PendulumSACRunnerCfg"],
    "psd_pendulum": ["Pendulum", "PendulumPSDCfg", "PendulumPSDRunnerCfg"],
    "go2": ["Go2", "Go2Cfg", "Go2RunnerCfg"]
}


def _load_declared_symbols(symbol_locations):
    loaded = {}
    for symbol, module in symbol_locations.items():
        loaded[symbol] = getattr(importlib.import_module(module, __name__), symbol)
        globals()[symbol] = loaded[symbol]
    return loaded


# Every declaration is required. Import and attribute failures intentionally
# abort registration instead of making a broken task disappear from the CLI.
_loaded = {}
for declarations in (class_dict, config_dict, runner_config_dict):
    _loaded.update(_load_declared_symbols(declarations))

for task_name, (class_name, config_name, runner_config_name) in task_dict.items():
    task_registry.register(
        task_name,
        _loaded[class_name],
        _loaded[config_name](),
        _loaded[runner_config_name](),
    )
