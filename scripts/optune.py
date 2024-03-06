import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback


from gym.envs import __init__  # noqa: F401
from gym.utils import get_args, task_registry, randomize_episode_counters
from gym.utils.logging_and_saving import wandb_singleton
from gym.utils.logging_and_saving import local_code_save_helper
from torch.multiprocessing import Process, Queue

wandb_helper = wandb_singleton.WandbSingleton()


def update_hyperparams(env_cfg, train_cfg, hyperparams):
    for key, val in hyperparams["env_cfg"].items():
        setattr(env_cfg, key, val)
    for key, val in hyperparams["train_cfg"].items():
        setattr(train_cfg, key, val)


def setup(env_cfg, train_cfg):
    args = get_args()
    task_registry.make_gym_and_sim()
    env = task_registry.make_env(name=args.task, env_cfg=env_cfg)
    randomize_episode_counters(env)

    policy_runner = task_registry.make_alg_runner(env, train_cfg)
    task_registry.set_log_dir_name(train_cfg)
    local_code_save_helper.save_local_files_to_logs(train_cfg.log_dir)
    wandb_helper.setup_wandb(env_cfg=env_cfg, train_cfg=train_cfg, args=args)

    return train_cfg, policy_runner


def train(train_cfg, policy_runner):
    policy_runner.learn()
    return policy_runner.alg.mean_surrogate_loss


def objective(trial, env_cfg, train_cfg):
    train_cfg.algorithm.learning_rate = trial.suggest_float(
        "learning_rate", 1e-5, 1e-1, log=True
    )

    queue = Queue()
    p = Process(target=run_setup_and_train, args=(env_cfg, train_cfg, queue))
    p.start()
    p.join()
    result = queue.get()
    p.kill()
    return result


def run_setup_and_train(env_cfg, train_cfg, queue):
    train_cfg, policy_runner = setup(env_cfg, train_cfg)
    train_result = train(train_cfg, policy_runner)
    queue.put(train_result)


if __name__ == "__main__":
    args = get_args()
    env_cfg, train_cfg = task_registry.create_cfgs(args)
    study = optuna.create_study(direction="minimize")
    wandb_helper.setup_wandb(env_cfg=env_cfg, train_cfg=train_cfg, args=args)
    wandb_kwargs = {
        "project": wandb_helper.get_project_name(),
        "entity": wandb_helper.get_entity_name(),
    }
    wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs)
    study.optimize(
        lambda trial: objective(trial, env_cfg, train_cfg),
        n_trials=3,
        callbacks=[wandbc],
    )
    print("Best value: {} (params: {})\n".format(study.best_value, study.best_params))
