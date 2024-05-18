import time
import optuna

# from learning.modules.lqrc import Cholesky  # noqa F401
from learning.modules.lqrc import CholeskyInput, CholeskyLatent  # noqa F401
from learning.utils import (
    compute_generalized_advantages,
    compute_MC_returns,
    create_uniform_generator,
)
from gym import LEGGED_GYM_ROOT_DIR
import os
import torch

DEVICE = "cuda:0"
# handle some bookkeeping
run_name = "May15_16-20-21_standard_critic"  # "May13_10-52-30_standard_critic"
log_dir = os.path.join(
    LEGGED_GYM_ROOT_DIR, "logs", "pendulum_standard_critic", run_name
)
time_str = time.strftime("%Y%m%d_%H%M%S")


def objective(trial):
    # learning_rate = trial.suggest_float("learning_rate", 1.0e-6, 1.0e-3, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1.0e-7, 1.0e-2, log=True)

    # hidden_dims = [trial.suggest_int(f'latent_dim_{i}', 32, 256) for i in range(3)]
    # latent_dim = [trial.suggest_categorical(f'latent_hidden_dim_{i}'
    # , [2**x for x in range(5, 9)]) for i in range(list_length)]
    latent_dim = trial.suggest_categorical("latent_dim", [2**x for x in range(1, 5)])

    test_critic_params = {
        "num_obs": 2,
        "hidden_dims": [512, 256, 64],
        "activation": ["elu", "elu", "tanh"],
        "normalize_obs": True,
        "latent_dim": latent_dim,
        "latent_hidden_dims": [4, 8],
        "latent_activation": ["elu", "elu"],
        "device": DEVICE,
    }
    gamma = 0.99
    lam = trial.suggest_float("lam", 0.5, 1.0)
    max_gradient_steps = 1000
    batch_size = trial.suggest_categorical("batch_size", [10**x for x in range(4, 6)])
    # value_offset_0 = data["returns"].max()
    value_offset_0 = trial.suggest_float("value_offset_0", 0.0, 5.0)

    test_critic = CholeskyLatent(**test_critic_params).to(DEVICE)
    iteration = 100
    critic_optimizer = torch.optim.Adam(test_critic.parameters(), lr=learning_rate)
    data = torch.load(os.path.join(log_dir, "data_{}.pt".format(iteration))).to(DEVICE)
    # compute ground-truth
    episode_rollouts = compute_MC_returns(data, gamma)

    # train new critic
    with torch.no_grad():
        data["advantages"] = compute_generalized_advantages(
            data, gamma, lam, test_critic
        )
        data["returns"] = data["advantages"] + data["values"]

    mean_value_loss = 0
    counter = 0
    with torch.no_grad():
        test_critic.value_offset.copy_(value_offset_0)
    generator = create_uniform_generator(
        data,
        batch_size,
        max_gradient_steps=max_gradient_steps,
    )

    for batch in generator:
        value_loss = test_critic.loss_fn(batch["critic_obs"], batch["returns"])
        critic_optimizer.zero_grad()
        value_loss.backward()
        critic_optimizer.step()
        mean_value_loss += value_loss.item()
        counter += 1
    mean_value_loss /= counter
    with torch.no_grad():
        final_loss = (
            (episode_rollouts[0] - test_critic.evaluate(data["critic_obs"][0]))
            .pow(2)
            .mean()
        ).to("cpu")
    return final_loss.item()


study = optuna.create_study(storage="sqlite:///db.sqlite3", direction="minimize")
study.optimize(objective, n_trials=1000)

print("Number of finished trials: ", len(study.trials))
print(study.best_trial.value)  # Show the best value.
# print best trial parameters
print(study.best_trial.params)
