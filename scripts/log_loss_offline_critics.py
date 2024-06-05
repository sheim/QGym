import time
from learning.modules.critic import Critic  # noqa F401
from learning.modules.lqrc import *  # noqa F401
from learning.modules.lqrc import *  # noqa F401
from learning.utils import (
    compute_generalized_advantages,
    compute_MC_returns,
    create_uniform_generator,
)
from learning.modules.lqrc.utils import train, train_sequentially, train_interleaved
from learning.modules.lqrc.plotting import (
    plot_pendulum_multiple_critics,
    plot_state_data_dist,
    plot_pendulum_multiple_critics_w_data,
)
from gym import LEGGED_GYM_ROOT_DIR
import os
import torch
from torch import nn  # noqa F401
from critic_params import critic_params
import wandb # noqa F401

DEVICE = "cuda:0"
for critic_param in critic_params.values():
    critic_param["device"] = DEVICE

# handle some bookkeeping
run_name = "May15_16-20-21_standard_critic"
log_dir = os.path.join(
    LEGGED_GYM_ROOT_DIR, "logs", "pendulum_standard_critic", run_name
)
time_str = time.strftime("%Y%m%d_%H%M%S")

save_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "offline_critics_graph", time_str)
if not os.path.exists(save_path):
    os.makedirs(save_path)

learning_rate = 0.001
critic_names = [
    "Critic",
    # "CholeskyInput",
    # "CholeskyLatent",
    # "PDCholeskyInput",
    # "PDCholeskyLatent",
    # "SpectralLatent",
    # ]
    # "Cholesky",
    # "CholeskyPlusConst",
    # "CholeskyOffset1",
    # "CholeskyOffset2",
    "QPNet"
    # "NN_wQR",
    # "NN_wLinearLatent",
    # "NN_wRiccati", # ! WIP
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
            test_critics[name].value_offset.copy_(3.25)
    if name == "NN_wQR" or name == "NN_wRiccati":
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
                test_critics[name].critic.parameters(), lr=0.0035155648350750773  # learning_rate
            ),
            "regularization": torch.optim.Adam(
                test_critics[name].critic.latent_NN.parameters(), lr=0.00619911552327558 #learning_rate
            ),
        }
    else:
        critic_optimizers[name] = torch.optim.Adam(
            test_critics[name].parameters(), lr=learning_rate
        )

gamma = 0.95
lam = 0.5080185484279778  # 0.95
tot_iter = 200
# wandb_run = wandb.init(
#     project="lqrc", entity="biomimetics", name="_".join(critic_names)
# )

for iteration in range(100, tot_iter, 10):
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

    for name, critic in test_critics.items():
        critic_optimizer = critic_optimizers[name]
        data = base_data.detach().clone()
        # train new critic
        data["values"] = critic.evaluate(data["critic_obs"])
        data["advantages"] = compute_generalized_advantages(data, gamma, lam, critic)
        data["returns"] = data["advantages"] + data["values"]

        max_gradient_steps = 1000 # 100
        # max_grad_norm = 1.0
        batch_size = 256
        num_steps = 50
        n_trajs = 50
        traj_idx = torch.randperm(data.shape[1])[0:n_trajs]
        generator = create_uniform_generator(
            data[:num_steps, traj_idx],
            # data[:num_steps, 0:-1:200],
            batch_size,
            max_gradient_steps=max_gradient_steps,
        )
        # plot_state_data_dist(
        #     data[:num_steps, traj_idx]["critic_obs"], save_path + "/data_dist"
        #     # data[:num_steps, 0:-1:200]["critic_obs"], save_path + "/data_dist"
        # )

        # perform backprop
        regularization = critic_params[name].get("regularization")
        if regularization is not None:
            train_func = (
                train_sequentially
                if regularization == "sequential"
                else train_interleaved
            )
            reg_generator = create_uniform_generator(
                data[:num_steps, traj_idx], batch_size=64,  # 1000,
                max_gradient_steps=100
            )
            (
                logging_dict[name]["mean_value_loss"],
                logging_dict[name]["mean_regularization_loss"],
            ) = train_func(
                critic,
                critic_optimizer["value"],
                critic_optimizer["regularization"],
                generator,
                reg_generator,
            )
        else:
            logging_dict[name]["mean value loss"] = train(
                critic, critic_optimizer, generator
            )
        # prepare data for graphing
        graphing_data["critic_obs"][name] = data[0, :]["critic_obs"]
        graphing_data["values"][name] = critic.evaluate(data[0, :]["critic_obs"])
        graphing_data["returns"][name] = data[0, :]["returns"]

    # print("logging dict", logging_dict)
    # wandb_run.log(logging_dict)
    # plot
    # plot_pendulum_multiple_critics(
    #     graphing_data["critic_obs"],
    #     graphing_data["values"],
    #     graphing_data["returns"],
    #     title=f"iteration{iteration}",
    #     fn=save_path + f"/{len(critic_names)}_CRITIC_it{iteration}",
    # )
    plot_pendulum_multiple_critics_w_data(
        graphing_data["critic_obs"],
        graphing_data["values"],
        graphing_data["returns"],
        title=f"iteration{iteration}",
        fn=save_path + f"/{len(critic_names)}_CRITIC_it{iteration}",
        data=data[:num_steps, traj_idx]["critic_obs"]
    )
