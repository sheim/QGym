from gym.envs import __init__  # noqa: F401
from gym.utils import get_args, task_registry
from gym.utils import SliderInterface
from gym.utils import VisualizationRecorder
import numpy as np
# torch needs to be imported after isaacgym imports in local source
import torch


def setup(args):
    env_cfg, train_cfg = task_registry.create_cfgs(args)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 16)
    if hasattr(env_cfg, "push_robots"):
        env_cfg.push_robots.toggle = False
    if hasattr(env_cfg, "commands"):
        env_cfg.commands.resampling_time = 9999
    env_cfg.env.episode_length_s = 9999
    env_cfg.env.num_projectiles = 20
    env_cfg.asset.fix_base_link = True
    task_registry.make_gym_and_sim()
    env_cfg.init_state.pos = [0, 0, 1.0]
    env_cfg.init_state.reset_mode = "reset_to_basic"
    env = task_registry.make_env(args.task, env_cfg)
    train_cfg.runner.resume = True
    train_cfg.logging.enable_local_saving = False
    runner = task_registry.make_alg_runner(env, train_cfg)

    # * switch to evaluation mode (dropout for example)
    runner.switch_to_eval()
    return env, runner, train_cfg



def play(env, runner, train_cfg):
    if env.cfg.viewer.record:
        recorder = VisualizationRecorder(
            env, train_cfg.runner.experiment_name, train_cfg.runner.load_run
        )
    saveLogs = False
    log = {'dof_pos_obs': [], 
           'dof_vel': [], 
           'torques': [],
           'grf': [], 
           'oscillators': [],
           'base_lin_vel': [],
           'base_ang_vel': [],
           'commands': [],
           'dof_pos_error': [],
           'reward': [],
            'dof_names': [],
           }
    RECORD_FRAMES = False
    print(env.dof_names)
 
    # * set up interface: GamepadInterface(env) or KeyboardInterface(env)
    COMMANDS_INTERFACE = hasattr(env, "commands")
    if COMMANDS_INTERFACE:
        # interface = GamepadInterface(env)
        interface = SliderInterface(env)
    img_idx = 0

    for i in range(10*int(env.max_episode_length)):

        if env.cfg.viewer.record:
            recorder.update(i)
        if i ==1000 and saveLogs:
            log['dof_names'] = env.dof_names
            np.savez('new_logs', **log)

        if COMMANDS_INTERFACE:
            interface.update(env)
        env.step()
        env.check_exit()



if __name__ == '__main__':
    EXPORT_POLICY = True
    args = get_args()
    with torch.no_grad():
        env, runner, train_cfg = setup(args)
        play(env, runner, train_cfg)
