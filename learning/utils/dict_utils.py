import torch
from tensordict import TensorDict


@torch.no_grad
def compute_MC_returns(data: TensorDict, gamma, critic=None):
    if critic is None:
        last_values = torch.zeros_like(data["rewards"][0])
    else:
        last_values = critic.evaluate(data["critic_obs"][-1])

    data.update({"returns": torch.zeros_like(data["rewards"])})
    data["returns"][-1] = last_values * ~data["dones"][-1]
    for k in reversed(range(data["rewards"].shape[0] - 1)):
        not_done = ~data["dones"][k]
        data["returns"][k] = (
            data["rewards"][k] + gamma * data["returns"][k + 1] * not_done
        )
    data["returns"] = (data["returns"] - data["returns"].mean()) / (
        data["returns"].std() + 1e-8
    )
    return


@torch.no_grad
def compute_generalized_advantages(data, gamma, lam, critic, last_values=None):
    data.update({"values": critic.evaluate(data["critic_obs"])})

    data.update({"advantages": torch.zeros_like(data["values"])})

    if last_values is not None:
        # todo check this
        # since we don't have observations for the last step, need last value plugged in
        not_done = ~data["dones"][-1]
        data["advantages"][-1] = (
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
        data["advantages"][k] = (
            td_error + gamma * lam * not_done * data["advantages"][k + 1]
        )

    data["returns"] = data["advantages"] + data["values"]

    data["advantages"] = (data["advantages"] - data["advantages"].mean()) / (
        data["advantages"].std() + 1e-8
    )


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
