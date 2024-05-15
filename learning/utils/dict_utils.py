import numpy as np
import torch
from tensordict import TensorDict


@torch.no_grad
def compute_MC_returns(data: TensorDict, gamma, critic=None):
    # todo not as accurate as taking
    if critic is None:
        last_values = torch.zeros_like(data["rewards"][0])
    else:
        last_values = critic.evaluate(data["critic_obs"][-1])

    returns = torch.zeros_like(data["rewards"])
    returns[-1] = data["rewards"][-1] + gamma * last_values * ~data["terminated"][-1]
    for k in reversed(range(data["rewards"].shape[0] - 1)):
        not_done = ~data["dones"][k]
        returns[k] = data["rewards"][k] + gamma * returns[k + 1] * not_done
        if critic is not None:
            returns[k] += (
                gamma
                * critic.evaluate(data["critic_obs"][k])
                * data["timed_out"][k]
                * ~data["terminated"][k]
            )
    return returns


@torch.no_grad
def normalize(input, eps=1e-8):
    return (input - input.mean()) / (input.std() + eps)


@torch.no_grad
def compute_generalized_advantages(data, gamma, lam, critic):
    data["values"] = critic.evaluate(data["critic_obs"])
    last_values = critic.evaluate(data["next_critic_obs"][-1])
    advantages = torch.zeros_like(data["values"])
    if last_values is not None:
        # todo check this
        not_done = ~data["dones"][-1]
        advantages[-1] = (
            data["rewards"][-1]
            + gamma * data["values"][-1] * data["timed_out"][-1]
            + gamma * last_values * not_done
            - data["values"][-1]
        )

    for k in reversed(range(data["values"].shape[0] - 1)):
        not_done = ~data["dones"][k]
        td_error = (
            data["rewards"][k]
            + gamma * data["values"][k] * data["timed_out"][k]
            + gamma * data["values"][k + 1] * not_done
            - data["values"][k]
        )
        advantages[k] = td_error + gamma * lam * not_done * advantages[k + 1]

    return advantages


# todo change num_epochs to num_batches
@torch.no_grad
def create_uniform_generator(
    data, batch_size, num_epochs=1, max_gradient_steps=None, keys=None
):
    n, m = data.shape
    total_data = n * m

    if batch_size > total_data:
        batch_size = total_data

    num_batches_per_epoch = total_data // batch_size
    if max_gradient_steps:
        num_epochs = max_gradient_steps // num_batches_per_epoch
        num_epochs = max(num_epochs, 1)
    for epoch in range(num_epochs):
        indices = torch.randperm(total_data, device=data.device)
        for i in range(num_batches_per_epoch):
            batched_data = data.flatten(0, 1)[
                indices[i * batch_size : (i + 1) * batch_size]
            ]
            yield batched_data


@torch.no_grad
def export_to_numpy(data, path):
    # check if path ends iwth ".npz", and if not append it.
    if not path.endswith(".npz"):
        path += ".npz"
    np.savez_compressed(path, **{key: val.cpu().numpy() for key, val in data.items()})
    return
