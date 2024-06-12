from gym.envs import __init__  # noqa: F401
from gym.utils import get_args, task_registry, randomize_episode_counters
from gym.utils.logging_and_saving import wandb_singleton
from gym.utils.logging_and_saving import local_code_save_helper
from torch.multiprocessing import Process
from torch.multiprocessing import set_start_method

def setup(size):
    args = get_args()
    args.wandb_entity = "biomimetics"
    args.wandb_project = "network_sweep"
    args.experiment_name = "network_sweep"
    args.run_name = "osc_" + str([size, size])#"all_" + str(size) #
    wandb_helper = wandb_singleton.WandbSingleton()
    # args.ctrl_frequency = size
    env_cfg, train_cfg = task_registry.create_cfgs(args)
    train_cfg.policy.actor_hidden_dims = [size, size]
    train_cfg.policy.critic_hidden_dims = [size, size]

    task_registry.make_gym_and_sim()
    wandb_helper.setup_wandb(env_cfg=env_cfg, train_cfg=train_cfg, args=args)
    env = task_registry.make_env(name=args.task, env_cfg=env_cfg)
    randomize_episode_counters(env)

    policy_runner = task_registry.make_alg_runner(env, train_cfg)

    local_code_save_helper.save_local_files_to_logs(train_cfg.log_dir)

    return train_cfg, policy_runner, env


def train(train_cfg, policy_runner):
    wandb_helper = wandb_singleton.WandbSingleton()

    policy_runner.learn()

    wandb_helper.close_wandb()

def worker(size):
    train_cfg, policy_runner, env = setup(size)
    train(train_cfg=train_cfg, policy_runner=policy_runner)


if __name__ == "__main__":
    networksizes = [12]#[100, 75, 50, 25, 15, 10, 5]#[60, 55, 50, 45, 40, 35] #[28, 24, 20] #
    set_start_method('spawn')
    for i,size in enumerate(networksizes):
        print(i)
        p = Process(target=worker, args=(size,))
        p.start()
        p.join()  # Wait for the process to finish

        # Free up any resources if needed
        p.terminate()
        p.close()
