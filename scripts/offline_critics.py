from gym.envs import __init__  # noqa: F401
from gym.utils import get_args, task_registry
from learning.modules import Actor, Critic
from learning.modules.lqrc import Cholesky
from gym.utils.helpers import class_to_dict
from learning.utils import (
    compute_generalized_advantages,
)

from gym import LEGGED_GYM_ROOT_DIR
import os
import torch

# handle some bookkeeping
args = get_args()
env_cfg, train_cfg = task_registry.create_cfgs(args)
env_cfg = class_to_dict(env_cfg)
train_cfg = class_to_dict(train_cfg)
#
run_name = "May01_10-47-07_lam09standard_critic"
log_dir = os.path.join(
    LEGGED_GYM_ROOT_DIR, "logs", train_cfg["runner"]["experiment_name"], run_name
)

# create critic based on saved policy
iteration = 1
loaded_dict = torch.load(os.path.join(log_dir, "model_{}.pt".format(iteration)))
trained_actor = Actor(num_obs=2, num_actions=1, **train_cfg["actor"])
trained_actor.load_state_dict(loaded_dict["actor_state_dict"])
trained_critic = Critic(num_obs=2, **train_cfg["critic"])
trained_critic.load_state_dict(loaded_dict["critic_state_dict"])

# create fresh critic

test_critic = Cholesky(num_obs=2, **train_cfg["critic"])

# load data

data = torch.load(os.path.join(log_dir, "data_{}.pt".format(iteration)))
gamma = train_cfg["algorithm"]["gamma"]
lam = train_cfg["algorithm"]["lam"]

# compute ground-truth

# train new critic

data["values"] = test_critic.evaluate(data["critic_obs"])
data["advantages"] = compute_generalized_advantages(data, gamma, lam, test_critic)
data["returns"] = data["advantages"] + data["values"]

# plug in supervised learning here, based off of PPO2.update

# compare new and old critics

end = True
