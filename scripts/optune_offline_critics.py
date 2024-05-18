import time
import wandb
import optuna

from learning.modules.critic import Critic  # noqa F401
from learning.modules.lqrc import *  # noqa F401
from learning.modules.lqrc import *  # noqa F401
from learning.utils import (
    compute_generalized_advantages,
    compute_MC_returns,
    create_uniform_generator,
)
from learning.modules.lqrc.utils import train, train_sequentially, train_interleaved
from learning.modules.lqrc.plotting import plot_pendulum_multiple_critics
from gym import LEGGED_GYM_ROOT_DIR
import os
import torch
from torch import nn  # noqa F401

DEVICE = "cuda:0"
# handle some bookkeeping
run_name = "May15_16-20-21_standard_critic"  # "May13_10-52-30_standard_critic"  # "May13_10-52-30_standard_critic"
log_dir = os.path.join(
    LEGGED_GYM_ROOT_DIR, "logs", "pendulum_standard_critic", run_name
)

# Parameters for different critics
critic_params = {
    "CholeskyInput": {
        "num_obs": 2,
        "hidden_dims": [128, 64, 32],
        "activation": ["elu", "elu", "tanh"],
        "normalize_obs": True,
        "latent_dim": None,  # 16,
        "minimize": False,
        "device": DEVICE,
    },
    "PDCholeskyInput": {
        "num_obs": 2,
        "hidden_dims": [128, 64, 32],
        "activation": ["elu", "elu", "elu"],
        "normalize_obs": True,
        "latent_dim": None,  # 16,
        "minimize": False,
        "device": DEVICE,
    },
    "CholeskyLatent": {
        "num_obs": 2,
        "hidden_dims": [128, 64, 32],
        "activation": ["elu", "elu", "elu"],
        "normalize_obs": False,
        "minimize": False,
        "latent_dim": 16,
        "latent_hidden_dims": [4, 8],
        "latent_activation": ["elu", "elu"],
        "device": DEVICE,
    },
    "PDCholeskyLatent": {
        "num_obs": 2,
        "hidden_dims": [128, 64, 32],
        "activation": ["elu", "elu", "elu"],
        "normalize_obs": False,
        "minimize": False,
        "latent_dim": 16,
        "latent_hidden_dims": [4, 8],
        "latent_activation": ["elu", "elu"],
        "device": DEVICE,
    },
    "SpectralLatent": {
        "num_obs": 2,
        "hidden_dims": [128, 64, 32],
        "activation": ["elu", "elu", "elu"],
        "normalize_obs": False,
        "minimize": False,
        "relative_dim": 4,
        "latent_dim": 16,
        "latent_hidden_dims": [4, 8],
        "latent_activation": ["elu", "elu"],
        "device": DEVICE,
    },
    "Critic": {
        "num_obs": 2,
        "hidden_dims": [128, 64, 32],
        "activation": "elu",
        "normalize_obs": False,
        "output_size": 1,
        "device": DEVICE,
    },
    "Cholesky": {
        "num_obs": 2,
        "hidden_dims": None,
        "activation": "elu",
        "normalize_obs": False,
        "output_size": 1,
        "device": DEVICE,
    },
    "CholeskyPlusConst": {
        "num_obs": 2,
        "hidden_dims": None,
        "activation": "elu",
        "normalize_obs": False,
        "output_size": 1,
        "device": DEVICE,
    },
    "CholeskyOffset1": {
        "num_obs": 2,
        "hidden_dims": None,
        "activation": "elu",
        "normalize_obs": False,
        "output_size": 1,
        "device": DEVICE,
    },
    "CholeskyOffset2": {
        "num_obs": 2,
        "hidden_dims": None,
        "activation": "elu",
        "normalize_obs": False,
        "output_size": 1,
        "device": DEVICE,
    },
    "NN_wRiccati": {
        "critic_name": "CholeskyInput",
        "action_dim": 2,
        "regularization": "sequential",  # alternative is "interleaved"
    },
    "NN_wLinearLatent": {
        "critic_name": "SpectralLatent",
        "action_dim": 2,
        "regularization": "sequential",  # alternative is "interleaved"
    },
}

learning_rate = 0.001
critic_names = [
    # "Critic",
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
    # "NN_wRiccati",
    "NN_wLinearLatent",
]


tot_iter = 2


def objective(trial):
    gamma = 0.99
    lam = trial.suggest_float("lam", 0.5, 1.0)
    max_gradient_steps = 100
    batch_size = trial.suggest_categorical("batch_size", [10**x for x in range(4, 6)])

    # critic set up
    if hasattr(test_critic, "value_offset"):
        with torch.no_grad():
            test_critic.value_offset.copy_(3.3 / 100.0)
    if name == "NN_wRiccati":
        critic_optimizer = {
            "value": torch.optim.Adam(
                test_critic.critic.parameters(),
                lr=trial.suggest_float("lr_critic", 1.0e-7, 1.0e-2, log=True),
            ),
            "regularization": torch.optim.Adam(
                test_critic.QR_network.parameters(),
                lr=trial.suggest_float("lr_QR", 1.0e-7, 1.0e-2, log=True),
            ),
        }
    elif name == "NN_wLinearLatent":
        critic_optimizer = {
            "value": torch.optim.Adam(
                test_critic.critic.parameters(),
                lr=trial.suggest_float("lr_critic", 1.0e-7, 1.0e-2, log=True),
            ),
            "regularization": torch.optim.Adam(
                test_critic.critic.latent_NN.parameters(),
                lr=trial.suggest_float("lr_latent", 1.0e-7, 1.0e-2, log=True),
            ),
        }
    else:
        critic_optimizer = torch.optim.Adam(
            test_critic.parameters(),
            lr=trial.suggest_float("lr_critic", 1.0e-7, 1.0e-2, log=True),
        )

    latest_loss = 0
    # train critic
    for iteration in range(1, tot_iter, 1):
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
        generator = create_uniform_generator(
            data,
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
                mean_value_loss,
                regularization_loss,
            ) = train_func(
                test_critic,
                critic_optimizer["value"],
                critic_optimizer["regularization"],
                val_generator,
                reg_generator,
            )
            trial.report(mean_value_loss + regularization_loss, iteration)
            latest_loss = mean_value_loss + regularization_loss
        else:
            mean_value_loss = train(test_critic, critic_optimizer, generator)
            trial.report(mean_value_loss, iteration)
            latest_loss = mean_value_loss
    return latest_loss


time_str = time.strftime("%Y%m%d_%H%M%S")
for name in critic_names:
    params = critic_params[name]
    if "critic_name" in params.keys():
        params.update(critic_params[params["critic_name"]])
    print("params", params)
    critic_class = eval(name)
    test_critic = critic_class(**params).to(DEVICE)
    study = optuna.create_study(storage="sqlite:///db.sqlite3", direction="minimize")
    study.optimize(objective, n_trials=1)
    save_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "optuna", time_str)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    study.trials_dataframe().to_csv(save_path + f"/{name}.csv")
