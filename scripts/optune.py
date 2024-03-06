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
    # wandb_helper = wandb_singleton.WandbSingleton()
    policy_runner.learn()
    # wandb_helper.close_wandb()
    return policy_runner.alg.mean_surrogate_loss


# Define the objective function for Optuna
def objective(trial, env_cfg, train_cfg):
    train_cfg.algorithm.learning_rate = trial.suggest_loguniform(
        "learning_rate", 1e-5, 1e-1
    )

    queue = Queue()
    p = Process(target=run_setup_and_train, args=(env_cfg, train_cfg, queue))
    p.start()
    p.join()
    result = queue.get()
    p.kill()
    return result


# The function to run in a separate process
def run_setup_and_train(env_cfg, train_cfg, queue):
    train_cfg, policy_runner = setup(env_cfg, train_cfg)
    train_result = train(train_cfg, policy_runner)  # Execute training
    queue.put(train_result)


if __name__ == "__main__":
    args = get_args()
    env_cfg, train_cfg = task_registry.create_cfgs(args)
    wandb_helper.setup_wandb(env_cfg=env_cfg, train_cfg=train_cfg, args=args)
    # if wandb_helper.is_wandb_enabled():
    #     if sweep_id is None:
    #         sweep_id = wandb.sweep(
    #             sweep_config,
    #             entity=wandb_helper.get_entity_name(),
    #             project=wandb_helper.get_project_name(),
    #         )
    #     wandb.agent(
    #         sweep_id,
    #         sweep_wandb_mp,
    #         entity=wandb_helper.get_entity_name(),
    #         project=wandb_helper.get_project_name(),
    #         count=sweep_config["run_cap"],
    #     )
    # else:
    #     print("ERROR: No WandB project and entity provided for sweeping")

    ###########################

    study = optuna.create_study(direction="minimize")
    # to integrate with our exisitng wandb stuff, we would need to initialize
    # the wandb_helper out here.
    wandb_kwargs = {"project": "my-project"}
    wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs)
    study.optimize(
        lambda trial: objective(trial, env_cfg, train_cfg),
        n_trials=3,
        callbacks=[wandbc],
    )
    print("Best value: {} (params: {})\n".format(study.best_value, study.best_params))
