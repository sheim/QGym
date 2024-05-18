import time
import wandb
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
time_str = time.strftime("%Y%m%d_%H%M%S")

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
# Instantiate the critics and add them to test_critics
test_critics = {}
critic_optimizers = {}
for name in critic_names:
    params = critic_params[name]
    if "critic_name" in params.keys():
        params.update(critic_params[params["critic_name"]])
    critic_class = globals()[name]
    test_critics[name] = critic_class(**params).to(DEVICE)
    if hasattr(test_critics[name], "value_offset"):
        with torch.no_grad():
            test_critics[name].value_offset.copy_(3.3 / 100.0)
    if name == "NN_wRiccati":
        critic_optimizers[name] = {
            "value": torch.optim.Adam(
                test_critics[name].critic.parameters(), lr=learning_rate
            ),
            "regularization": torch.optim.Adam(
                test_critics[name].QR_network.parameters(), lr=learning_rate
            ),
        }
    elif name == "NN_wLinearLatent":
        critic_optimizers[name] = {
            "value": torch.optim.Adam(
                test_critics[name].critic.parameters(), lr=learning_rate
            ),
            "regularization": torch.optim.Adam(
                test_critics[name].critic.latent_NN.parameters(), lr=learning_rate
            ),
        }
    else:
        critic_optimizers[name] = torch.optim.Adam(
            test_critics[name].parameters(), lr=learning_rate
        )

gamma = 0.99
lam = 0.95
tot_iter = 200
wandb_run = wandb.init(
    project="lqrc", entity="biomimetics", name="_".join(critic_names)
)

for iteration in range(1, tot_iter, 1):
    # load data and empty log
    base_data = torch.load(os.path.join(log_dir, "data_{}.pt".format(iteration))).to(
        DEVICE
    )
    logging_dict = {
        name: {} for name in critic_names
    }  # TODO: account for using the same critic wrapper with different custom critics

    # compute ground-truth
    graphing_data = {data_name: {} for data_name in ["critic_obs", "values", "returns"]}

    episode_rollouts = compute_MC_returns(base_data, gamma)
    graphing_data["critic_obs"]["Ground Truth MC Returns"] = (
        base_data[0, :]["critic_obs"].detach().clone()
    )
    graphing_data["values"]["Ground Truth MC Returns"] = episode_rollouts[0, :]
    graphing_data["returns"]["Ground Truth MC Returns"] = episode_rollouts[0, :]

    for name, test_critic in test_critics.items():
        # load data and set hyperparameters
        critic_optimizer = critic_optimizers[name]
        data = base_data.detach().clone()
        data["values"] = test_critic.evaluate(data["critic_obs"])
        data["advantages"] = compute_generalized_advantages(
            data, gamma, lam, test_critic
        )
        data["returns"] = data["advantages"] + data["values"]

        max_gradient_steps = 100
        # max_grad_norm = 1.0
        batch_size = 10 * 4096
        generator = create_uniform_generator(
            data,
            batch_size,
            max_gradient_steps=max_gradient_steps,
        )

        # perform backprop
        regularization = critic_params[name].get("regularization")
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
                logging_dict[name]["mean_value_loss"],
                logging_dict[name]["mean_regularization_loss"],
            ) = train_func(
                test_critic,
                critic_optimizer["value"],
                critic_optimizer["regularization"],
                val_generator,
                reg_generator,
            )
        else:
            logging_dict[name]["mean value loss"] = train(
                test_critic, critic_optimizer, generator
            )
        # prepare data for graphing
        graphing_data["critic_obs"][name] = data[0, :]["critic_obs"]
        graphing_data["values"][name] = data[0, :]["values"]
        graphing_data["returns"][name] = data[0, :]["returns"]

    wandb_run.log(logging_dict)
    # plot
    save_path = os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", "offline_critics_graph", time_str
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plot_pendulum_multiple_critics(
        graphing_data["critic_obs"],
        graphing_data["values"],
        graphing_data["returns"],
        title=f"iteration{iteration}",
        fn=save_path + f"/{len(critic_names)}_critics_it{iteration}",
    )
