from gym.utils.task_registry import task_registry
from gym.utils.helpers import set_seed
import gym.envs  # noqa: F401

import random


# Helper function to load actor network from logs/go2/run_name (default: most recent)
def setup_actor(run_name=None):
    env_cfg, train_cfg = task_registry.get_cfgs(name="go2")

    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 99999
    if hasattr(env_cfg, "commands"):
        env_cfg.commands.resampling_time = 99999
    if hasattr(env_cfg, "push_robots"):
        env_cfg.push_robots.toggle = False
    if hasattr(env_cfg, "init_state") and hasattr(env_cfg.init_state, "reset_mode"):
        env_cfg.init_state.reset_mode = "reset_to_range"

    env_cfg.seed = random.randint(0, 10000)
    train_cfg.seed = random.randint(0, 10000)

    train_cfg.runner.device = "cpu"
    train_cfg.runner.resume = True
    if run_name is not None:
        train_cfg.runner.load_run = run_name
    train_cfg.runner.checkpoint = -1
    train_cfg.logging.enable_local_saving = False

    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    task_registry.set_log_dir_name(train_cfg)
    set_seed(env_cfg.seed)

    env = task_registry.make_env_mujoco(
        name="go2", env_cfg=env_cfg, device="cpu", headless=True
    )

    runner = task_registry.make_alg_runner(env, train_cfg)
    runner.switch_to_eval()
    return runner.alg.actor


class RLController:
    def __init__(self):
        self.actor = setup_actor()

    def act(self, obs):
        return self.actor.act_inference(obs)
