import optuna
from gym.envs import __init__  # noqa: F401
from gym.utils import get_args, task_registry, randomize_episode_counters
from gym.utils.logging_and_saving import wandb_singleton
from gym.utils.logging_and_saving import local_code_save_helper
from torch.multiprocessing import Process, Queue


# Adjusted setup function to accept hyperparameters
def setup(hyperparams):
    args = get_args()
    wandb_helper = wandb_singleton.WandbSingleton()

    # Load configurations
    env_cfg, train_cfg = task_registry.create_cfgs(args)

    # Adjust configurations based on hyperparameters
    # Example adjustments:
    # env_cfg.some_parameter = hyperparams["env_some_parameter"]
    train_cfg.learning_rate = hyperparams["learning_rate"]

    task_registry.make_gym_and_sim()
    wandb_helper.setup_wandb(env_cfg=env_cfg, train_cfg=train_cfg, args=args)
    env = task_registry.make_env(name=args.task, env_cfg=env_cfg)
    randomize_episode_counters(env)

    policy_runner = task_registry.make_alg_runner(env, train_cfg)

    local_code_save_helper.save_local_files_to_logs(train_cfg.log_dir)

    return train_cfg, policy_runner


def train(train_cfg, policy_runner):
    # wandb_helper = wandb_singleton.WandbSingleton()
    policy_runner.learn()
    # wandb_helper.close_wandb()
    return policy_runner.alg.mean_surrogate_loss


# Define the objective function for Optuna
def objective(trial):
    # Hyperparameter suggestions
    hyperparams = {
        # "env_some_parameter": trial.suggest_float("env_some_parameter", 0.0, 1.0),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-1),
    }

    queue = Queue()
    p = Process(target=run_setup_and_train, args=(queue, hyperparams))
    p.start()
    p.join()
    result = queue.get()
    p.kill()
    return result


# The function to run in a separate process
def run_setup_and_train(queue, hyperparams):
    train_cfg, policy_runner = setup(hyperparams)  # Pass hyperparams to setup
    train_result = train(train_cfg, policy_runner)  # Execute training
    queue.put(train_result)


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=3)
    print("Best value: {} (params: {})\n".format(study.best_value, study.best_params))
