import time
import optuna
import csv

from learning.modules.critic import Critic  # noqa F401
from learning.modules.lqrc import *  # noqa F401
from learning.modules.lqrc import *  # noqa F401
from learning.utils import (
    compute_generalized_advantages,
    compute_MC_returns,
    create_uniform_generator,
)
from learning.modules.lqrc.utils import train, train_sequentially, train_interleaved
from gym import LEGGED_GYM_ROOT_DIR
import os
import torch
from torch import nn  # noqa F401
from critic_params import critic_params

DEVICE = "cuda:0"
for critic_param in critic_params.values():
    critic_param["device"] = DEVICE

# handle some bookkeeping
run_name = "May22_11-11-03_standard_critic"
log_dir = os.path.join(
    LEGGED_GYM_ROOT_DIR, "logs", "pendulum_standard_critic", run_name
)

critic_names = [
    "Critic",
    # "CholeskyInput",
    # "CholeskyLatent",
    # "PDCholeskyInput",
    # "PDCholeskyLatent",
    "SpectralLatent",
    # ]
    # "Cholesky",
    # "CholeskyPlusConst",
    # "CholeskyOffset1",
    # "CholeskyOffset2",
    # "NN_wQR",
    "NN_wLinearLatent",
    # "NN_wRiccati", # ! WIP
]

tot_iter = 200


def objective(trial, name):
    gamma = 0.95
    lam = trial.suggest_float("lam", 0.1, 1.0)
    max_gradient_steps = 100
    # trial.suggest_categorical("max_grad_steps", [10**x for x in range(2, 4)])
    batch_size = 512
    # batch_size = trial.suggest_categorical("batch_size", [2**x for x in range(7, 10)])

    # critic set up
    params = critic_params[name]
    if "critic_name" in params.keys():
        params.update(critic_params[params["critic_name"]])

    if ("NN_" in name) or ("Latent" in name):
        relative_dim = trial.suggest_int("relative_dim", 5, 16)
        latent_dim = relative_dim + trial.suggest_int("latent_dim_diff", 0, 16)
        params["latent_dim"] = latent_dim
        params["relative_dim"] = relative_dim
    print("params", params)
    critic_class = eval(name)

    # if "relative_dim" in params.keys():
    #     params["relative_dim"] = trial.suggest_int("relative_dim", 1, 5)
    test_critic = critic_class(**params).to(DEVICE)

    if hasattr(test_critic, "value_offset"):
        initial_offset = trial.suggest_float("initial_offset", -1.0, 1.75)
        with torch.no_grad():
            test_critic.value_offset.copy_(initial_offset)

    if name == "NN_wQR":
        critic_optimizer = {
            "value": torch.optim.Adam(
                test_critic.critic.parameters(),
                lr=trial.suggest_float("lr_critic", 1.0e-7, 1.0e-5, log=True),
            ),
            "regularization": torch.optim.Adam(
                test_critic.QR_network.parameters(),
                lr=5e-7,  # trial.suggest_float("lr_QR", 1.0e-7, 1.0e-2, log=True),
            ),
        }
    elif name == "NN_wLinearLatent":
        critic_optimizer = {
            "value": torch.optim.Adam(
                test_critic.critic.parameters(),
                lr=trial.suggest_float("lr_critic", 1.0e-7, 1.0e-4, log=True),
            ),
            "regularization": torch.optim.Adam(
                test_critic.critic.latent_NN.parameters(),
                lr=5e-7,  # trial.suggest_float("lr_latent", 1.0e-7, 1.0e-2, log=True),
            ),
        }
    else:
        critic_optimizer = torch.optim.Adam(
            test_critic.parameters(),
            lr=trial.suggest_float("lr_critic", 1.0e-7, 1.0e-4, log=True),
        )

    # train critic
    for iteration in range(199, tot_iter, 1):
        # load data and empty log
        base_data = torch.load(
            os.path.join(log_dir, "data_{}.pt".format(iteration))
        ).to(DEVICE)

        # load data and set hyperparameters
        data = base_data.detach().clone()
        data["values"] = test_critic.evaluate(data["critic_obs"])
        data["advantages"] = compute_generalized_advantages(
            data, gamma, lam, test_critic
        )
        data["returns"] = data["advantages"] + data["values"]

        num_steps = 1
        n_trajs = trial.suggest_int("n_trajs", 201, data.shape[1], step=23)
        traj_idx = torch.randperm(data.shape[1])[0:n_trajs]
        generator = create_uniform_generator(
            data[:num_steps, traj_idx],
            # data[:num_steps, 0:-1:200],
            batch_size,
            max_gradient_steps=max_gradient_steps,
        )

        # perform backprop
        regularization = params.get("regularization")
        if regularization is not None:
            train_func = (
                train_sequentially
                if regularization == "sequential"
                else train_interleaved
            )
            val_generator = create_uniform_generator(
                data,
                batch_size,
                max_gradient_steps=max_gradient_steps,
            )
            reg_generator = create_uniform_generator(
                data, batch_size=1000, max_gradient_steps=100
            )
            (
                mean_val_loss,
                regularization_loss,
            ) = train_func(
                test_critic,
                critic_optimizer["value"],
                critic_optimizer["regularization"],
                val_generator,
                reg_generator,
            )
        else:
            mean_val_loss = train(test_critic, critic_optimizer, generator)  # noqa F405

    episode_rollouts = compute_MC_returns(data, gamma)
    actual_error = (
        test_critic.evaluate(data["critic_obs"][0]) - episode_rollouts[0]
    ).pow(2)
    return actual_error.mean().item(), actual_error.max().item(), n_trajs
    # return actual_error.mean().item(), actual_error.max().item()


time_str = time.strftime("%Y%m%d_%H%M%S")
for name in critic_names:
    save_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "optuna", time_str)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    study = optuna.create_study(
        storage=f"sqlite:///{save_path}/{name}_db.sqlite3",
        directions=["minimize", "minimize", "minimize"],
    )
    if name == "critic":
        n_trials = 1000
    else:
        n_trials = 10000
    study.optimize(lambda trial: objective(trial, name), n_trials=n_trials)
    study.trials_dataframe().to_csv(save_path + f"/{name}.csv")
    best_trials = [trial.params for trial in study.best_trials]
    with open(save_path + f"/{name}_best_trials.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=best_trials[0].keys())
        writer.writeheader()
        writer.writerows(best_trials)
