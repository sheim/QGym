from learning.modules.lqrc import Cholesky
from learning.utils import (
    compute_generalized_advantages,
)

from gym import LEGGED_GYM_ROOT_DIR
import os
import torch

DEVICE = "cuda:0"
# handle some bookkeeping
#
run_name = "May02_08-59-46_standard_critic"
log_dir = os.path.join(
    LEGGED_GYM_ROOT_DIR, "logs", "pendulum_standard_critic", run_name
)

# create critic based on saved policy
# iteration = 1
# loaded_dict = torch.load(os.path.join(log_dir, "model_{}.pt".format(iteration)))
# trained_actor = Actor(num_obs=2, num_actions=1, **train_cfg["actor"])
# trained_actor.load_state_dict(loaded_dict["actor_state_dict"])
# trained_critic = Critic(num_obs=2, **train_cfg["critic"])
# trained_critic.load_state_dict(loaded_dict["critic_state_dict"])

# create fresh critic

test_critic_params = {
    "num_obs": 2,
    "hidden_dims": None,
    "activation": "elu",
    "normalize_obs": False,
    "output_size": 1,
    "device": DEVICE,
}
test_critic = Cholesky(**test_critic_params).to(DEVICE)

# load data
iteration = 1
data = torch.load(os.path.join(log_dir, "data_{}.pt".format(iteration))).to(DEVICE)
gamma = 0.99
lam = 0.99

# compute ground-truth

# train new critic

data["values"] = test_critic.evaluate(data["critic_obs"])
data["advantages"] = compute_generalized_advantages(data, gamma, lam, test_critic)
data["returns"] = data["advantages"] + data["values"]

# plug in supervised learning here, based off of PPO2.update

# compare new and old critics

end = True
