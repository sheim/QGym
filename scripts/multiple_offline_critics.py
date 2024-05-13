import time
from learning.modules.critic import Critic
from learning.modules.lqrc import (
    CustomCriticBaseline,
    Cholesky,
    CholeskyPlusConst,
    CholeskyOffset1,
    CholeskyOffset2,
)  # noqa F401
from learning.utils import (
    compute_generalized_advantages,
    compute_MC_returns,
    create_uniform_generator,
)
from learning.modules.lqrc.plotting import (
    plot_pendulum_multiple_critics,
)
from gym import LEGGED_GYM_ROOT_DIR
import os
import torch
from torch import nn

DEVICE = "cuda:0"
# handle some bookkeeping
run_name = "May12_13-27-05_standard_critic"  # "May03_20-49-23_standard_critic" "May02_08-59-46_standard_critic"
log_dir = os.path.join(
    LEGGED_GYM_ROOT_DIR, "logs", "pendulum_standard_critic", run_name
)
time_str = time.strftime("%Y%m%d_%H%M%S")

# create fresh critic
custom_critic_params = {
    "num_obs": 2,
    "hidden_dims": None,
    "activation": "elu",
    "normalize_obs": False,
    "output_size": 1,
    "device": DEVICE,
}
vanilla_critic_params = {
    "num_obs": 2,
    "hidden_dims": [128, 64, 32],
    "activation": "elu",
    "normalize_obs": False,
    "output_size": 1,
    "device": DEVICE,
}
learning_rate = 1.0e-4
critic_names = [
    "Critic",
    "Cholesky",
    "CholeskyPlusConst",
    "CholeskyOffset1",
    "CholeskyOffset2",
]
test_critics = {
    name: eval(f"{name}(**custom_critic_params).to(DEVICE)")
    if not name == "Critic"
    else eval(f"{name}(**vanilla_critic_params).to(DEVICE)")
    for name in critic_names
}

critic_optimizers = {
    name: torch.optim.Adam(critic.parameters(), lr=learning_rate)
    for name, critic in test_critics.items()
}
gamma = 0.99
lam = 1.0
tot_iter = 200

for iteration in range(tot_iter):
    # load data
    data = torch.load(os.path.join(log_dir, "data_{}.pt".format(iteration))).to(DEVICE)

    # compute ground-truth
    graphing_data = {data_name: {} for data_name in ["critic_obs", "values", "returns"]}

    episode_rollouts = compute_MC_returns(data, gamma)
    graphing_data["critic_obs"]["Ground Truth MC Returns"] = data[0, :]["critic_obs"]
    graphing_data["values"]["Ground Truth MC Returns"] = episode_rollouts[0, :]
    graphing_data["returns"]["Ground Truth MC Returns"] = episode_rollouts[0, :]

    for name, test_critic in test_critics.items():
        critic_optimizer = critic_optimizers[name]
        # train new critic
        data["values"] = test_critic.evaluate(data["critic_obs"])
        data["advantages"] = compute_generalized_advantages(
            data, gamma, lam, test_critic
        )
        data["returns"] = data["advantages"] + data["values"]

        mean_value_loss = 0
        counter = 0
        max_gradient_steps = 24
        max_grad_norm = 1.0
        batch_size = 2**16
        generator = create_uniform_generator(
            data,
            batch_size,
            max_gradient_steps=max_gradient_steps,
        )
        for batch in generator:
            value_loss = test_critic.loss_fn(batch["critic_obs"], batch["returns"])
            critic_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(test_critic.parameters(), max_grad_norm)
            critic_optimizer.step()
            mean_value_loss += value_loss.item()
            print(f"{name} value loss: ", value_loss.item())
            counter += 1
        mean_value_loss /= counter

        graphing_data["critic_obs"][name] = data[0, :]["critic_obs"]
        graphing_data["values"][name] = data[0, :]["values"]
        graphing_data["returns"][name] = data[0, :]["returns"]

    # compare new and old critics
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
