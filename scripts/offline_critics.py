import time

# from learning.modules.lqrc import Cholesky  # noqa F401
from learning.modules.lqrc import *  # noqa F401
from learning.utils import (
    compute_generalized_advantages,
    compute_MC_returns,
    create_uniform_generator,
)
from learning.modules.lqrc.plotting import plot_pendulum_single_critic
from gym import LEGGED_GYM_ROOT_DIR
import os
import torch


DEVICE = "cuda:0"
# handle some bookkeeping
run_name = "May22_10-54-14_standard_critic"
# "May13_10-52-30_standard_critic"  # "May13_10-52-30_standard_critic"
log_dir = os.path.join(
    LEGGED_GYM_ROOT_DIR, "logs", "pendulum_standard_critic", run_name
)
time_str = time.strftime("%Y%m%d_%H%M%S")

test_critic_params = {
    "num_obs": 2,
    "hidden_dims": [512, 128, 64],
    "activation": ["elu", "elu", "tanh"],
    "relative_dim": 3,
    "normalize_obs": True,
    "latent_dim": 16,  # 16,
    "minimize": False,
    "latent_hidden_dims": [256, 64],
    "latent_activation": ["elu", "tanh"],
    "device": DEVICE,
}

learning_rate = 0.005415828580992768
critic_name = "SpectralLatent"
test_critic = SpectralLatent(**test_critic_params).to(DEVICE)  # noqa F405
critic_optimizer = torch.optim.Adam(test_critic.parameters(), lr=learning_rate)
gamma = 0.99
lam = 0.867
tot_iter = 200

for iteration in range(1, tot_iter, 10):
    # load data
    data = torch.load(os.path.join(log_dir, "data_{}.pt".format(iteration))).to(DEVICE)

    # compute ground-truth
    episode_rollouts = compute_MC_returns(data, gamma)

    # train new critic
    data["values"] = test_critic.evaluate(data["critic_obs"])
    data["advantages"] = compute_generalized_advantages(data, gamma, lam, test_critic)
    data["returns"] = data["advantages"] + data["values"]

    mean_value_loss = 0
    counter = 0
    max_gradient_steps = 10
    # max_grad_norm = 1.0
    batch_size = 64

    with torch.no_grad():
        test_critic.value_offset.copy_(1.5653120779642644)

    generator = create_uniform_generator(
        data, batch_size, max_gradient_steps=max_gradient_steps
    )

    # first train the latent representation for a bit
    lat_generator = create_uniform_generator(
        data, batch_size=256, max_gradient_steps=100
    )

    latent_optimizer = torch.optim.Adam(test_critic.latent_NN.parameters(), lr=0.001)

    for batch in lat_generator:
        latent_loss = linear_latent_loss_fn(batch, test_critic.latent_NN)  # noqa F405
        latent_optimizer.zero_grad()
        latent_loss.backward()
        latent_optimizer.step()

    for batch in generator:
        value_loss = test_critic.loss_fn(
            batch["critic_obs"], batch["returns"], actions=batch["actions"]
        )
        critic_optimizer.zero_grad()
        value_loss.backward()
        # nn.utils.clip_grad_norm_(test_critic.parameters(), max_grad_norm)
        critic_optimizer.step()
        mean_value_loss += value_loss.item()
        # print("Value loss: ", value_loss.item())
        # print("Value Offset: ", test_critic.value_offset.item())
        counter += 1
    mean_value_loss /= counter

    # compare new and old critics
    save_path = os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", "offline_critics_graph", time_str
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plot_pendulum_single_critic(
        x=data["critic_obs"][0],
        predictions=data["values"][0],
        targets=episode_rollouts[0],
        title=f"{critic_name}_iteration{iteration}",
        fn=save_path + f"/{critic_name}_it{iteration}",
    )

    print("Value Offset: ", test_critic.value_offset.item())
