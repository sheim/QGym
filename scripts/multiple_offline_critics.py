import time
from learning.modules.critic import Critic  # noqa F401
from learning.modules.lqrc import *  # noqa F401
from learning.modules.lqrc import *  # noqa F401
from learning.utils import (
    compute_generalized_advantages,
    compute_MC_returns,
    create_uniform_generator,
)
from learning.modules.lqrc.plotting import plot_pendulum_multiple_critics2
from gym import LEGGED_GYM_ROOT_DIR
import os
import torch
from torch import nn  # noqa F401
from critic_params import critic_params

DEVICE = "cuda:0"
for critic_param in critic_params.values():
    critic_param["device"] = DEVICE


# handle some bookkeeping
run_name = "May15_16-20-21_standard_critic"  # May22_11-11-03_standard_critic
log_dir = os.path.join(
    LEGGED_GYM_ROOT_DIR, "logs", "pendulum_standard_critic", run_name
)
time_str = time.strftime("%Y%m%d_%H%M%S")


learning_rate = 0.001
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
    "NN_wQR",
    "NN_wLinearLatent",
]
# Instantiate the critics and add them to test_critics
test_critics = {}
for name in critic_names:
    params = critic_params[name]
    if "critic_name" in params.keys():
        params.update(critic_params[params["critic_name"]])
    critic_class = globals()[name]
    test_critics[name] = critic_class(**params).to(DEVICE)
    if hasattr(test_critics[name], "value_offset"):
        with torch.no_grad():
            test_critics[name].value_offset.copy_(3.25)
critic_optimizers = {
    name: torch.optim.Adam(critic.parameters(), lr=learning_rate)
    for name, critic in test_critics.items()
}
gamma = 0.95
lam = 0.95
tot_iter = 200

for iteration in range(100, tot_iter, 10):
    # load data
    base_data = torch.load(os.path.join(log_dir, "data_{}.pt".format(iteration))).to(
        DEVICE
    )

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

        mean_value_loss = 0
        counter = 0
        max_gradient_steps = 100
        # max_grad_norm = 1.0
        batch_size = 256
        num_steps = 1  # ! want this at 1
        n_trajs = 50
        traj_idx = torch.randperm(data.shape[1])[0:n_trajs]
        generator = create_uniform_generator(
            data[:num_steps, traj_idx],
            batch_size,
            max_gradient_steps=max_gradient_steps,
        )
        for batch in generator:
            value_loss = critic.loss_fn(
                batch["critic_obs"], batch["returns"], actions=batch["actions"]
            )
            critic_optimizer.zero_grad()
            value_loss.backward()
            # # noqa F401.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_optimizer.step()
            mean_value_loss += value_loss.item()
            counter += 1
        ground_truth_loss = (
            (episode_rollouts[0] - critic.evaluate(data["critic_obs"][0])).pow(2).mean()
        ).to("cpu")
        print(f"{name} ground truth loss: ", ground_truth_loss.item())
        # print(f"{name} value loss: ", value_loss.item())
        mean_value_loss /= counter

        graphing_data["critic_obs"][name] = data[0, :]["critic_obs"]
        graphing_data["values"][name] = critic.evaluate(data[0, :]["critic_obs"])
        graphing_data["returns"][name] = data[0, :]["returns"]

    # compare new and old critics
    save_path = os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", "offline_critics_graph", time_str
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plot_pendulum_multiple_critics2(
        graphing_data["critic_obs"],
        graphing_data["values"],
        graphing_data["returns"],
        title=f"iteration{iteration}",
        fn=save_path + f"/{len(critic_names)}_CRITIC_it{iteration}",
    )
