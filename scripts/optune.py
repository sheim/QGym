import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback


from gym.envs import __init__  # noqa: F401
from gym.utils import get_args, task_registry, randomize_episode_counters
from gym.utils.logging_and_saving import wandb_singleton
from gym.utils.logging_and_saving import local_code_save_helper
from torch.multiprocessing import Process, Queue
import torch

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


def protocol(runner):
    # reset the rewards
    runner.policy_cfg["reward"]["weights"] = {"tracking_lin_vel": 1.0}
    # allow to run a long time
    runner.env.max_episode_length = 1e5
    runner.env.cfg.commands.resampling_time = 1e2
    # reset environments
    # runner.env.reset()
    # simulate and track for 5 seconds, perturb, then track another 5 seconds.
    five_seconds_of_steps = int(5 / runner.env.dt)
    # set up reward
    rewards_dict = {}
    avg_tracking_reward = 0.0

    # push settings
    runner.env.cfg.push_robots.max_push_vel_xy = 3.0

    with torch.inference_mode():
        for i in range(five_seconds_of_steps):
            runner.set_actions(
                runner.policy_cfg["actions"], runner.get_inference_actions()
            )
            runner.env.step()

            terminated = runner.get_terminated()
            rewards_dict.update(
                runner.get_rewards(
                    runner.policy_cfg["reward"]["weights"], mask=terminated
                )
            )
            avg_tracking_reward += rewards_dict["tracking_lin_vel"].mean()

        avg_tracking_reward /= five_seconds_of_steps
        # push robots
        runner.env._push_robots()
        dones = terminated  # just need to initialize it
        for i in range(five_seconds_of_steps):
            runner.set_actions(
                runner.policy_cfg["actions"], runner.get_inference_actions()
            )
            runner.env.step()

            terminated = runner.get_terminated()
            dones = terminated | dones
        fail_ratio = dones.sum() / runner.env.num_envs

    return (fail_ratio * avg_tracking_reward).item()


def run_setup_and_train(env_cfg, train_cfg, queue):
    train_cfg, policy_runner = setup(env_cfg, train_cfg)
    policy_runner.learn()
    result = protocol(policy_runner)
    queue.put(result)


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
