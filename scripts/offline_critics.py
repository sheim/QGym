import time
from learning.modules.lqrc import Cholesky
from learning.utils import (
    compute_generalized_advantages,
    compute_MC_returns,
    create_uniform_generator,
)
from learning.modules.lqrc.plotting import plot_pendulum_single_critic_predictions
from gym import LEGGED_GYM_ROOT_DIR
import os
import torch
from torch import nn

DEVICE = "cuda:0"
# handle some bookkeeping
#
run_name = "May03_17-11-21_standard_critic"  # "May02_08-59-46_standard_critic"
log_dir = os.path.join(
    LEGGED_GYM_ROOT_DIR, "logs", "pendulum_standard_critic", run_name
)

# create fresh critic

test_critic_params = {
    "num_obs": 2,
    "hidden_dims": None,
    "activation": "elu",
    "normalize_obs": False,
    "output_size": 1,
    "device": DEVICE,
}
learning_rate = 1.0e-4
critic_name = "Cholesky"
test_critic = eval(f"{critic_name}(**test_critic_params).to(DEVICE)")
critic_optimizer = torch.optim.Adam(test_critic.parameters(), lr=learning_rate)
# load data
iteration = 1
data = torch.load(os.path.join(log_dir, "data_{}.pt".format(iteration))).to(DEVICE)
gamma = 0.99
lam = 0.99

# compute ground-truth

episode_rollouts = compute_MC_returns(data, gamma)

# train new critic

data["values"] = test_critic.evaluate(data["critic_obs"])
data["advantages"] = compute_generalized_advantages(data, gamma, lam, test_critic)
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
    print("Value loss: ", value_loss.item())
    counter += 1
mean_value_loss /= counter

# compare new and old critics
time_str = time.strftime("%Y%m%d_%H%M%S")
save_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "offline_critics_graph", time_str)
if not os.path.exists(save_path):
    os.makedirs(save_path)
# TODO: revisit this, TwoSlopeNorm was causing discoloration
# vmin = min(torch.min(data["returns"]).item(), torch.min(data["values"]).item())
# vmax = max(torch.max(data["returns"]).item(), torch.max(data["values"]).item())
plot_pendulum_single_critic_predictions(
    x=data["critic_obs"][-1],
    predictions=data["values"][-1],
    targets=data["returns"][-1],
    title=f"{critic_name}_iteration{iteration}",
    fn=save_path + f"/{critic_name}_it{iteration}",
)
