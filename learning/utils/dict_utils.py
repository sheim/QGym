import numpy as np
import torch
from tensordict import TensorDict


@torch.no_grad
def compute_MC_returns(data: TensorDict, gamma, critic=None):
    if critic is None:
        last_values = torch.zeros_like(data["rewards"][0])
    else:
        last_values = critic.evaluate(data["critic_obs"][-1])

    returns = torch.zeros_like(data["rewards"])
    returns[-1] = last_values * ~data["dones"][-1]
    for k in reversed(range(data["rewards"].shape[0] - 1)):
        not_done = ~data["dones"][k]
        returns[k] = data["rewards"][k] + gamma * returns[k + 1] * not_done

    return normalize(returns)


@torch.no_grad
def normalize(input, eps=1e-8):
    return (input - input.mean()) / (input.std() + eps)


@torch.no_grad
def compute_generalized_advantages(data, gamma, lam, critic, last_values=None):
    advantages = torch.zeros_like(data["values"])
    if last_values is not None:
        # todo check this
        # since we don't have observations for the last step, need last value plugged in
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

    return normalize(advantages)


# todo change num_epochs to num_batches
@torch.no_grad
def create_uniform_generator(data, batch_size, num_epochs, keys=None):
    n, m = data.shape
    total_data = n * m
    num_batches_per_epoch = total_data // batch_size
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
