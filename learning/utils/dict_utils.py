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
def compute_generalized_advantages(data, gamma, lam, critic):
    last_values = critic.evaluate(data["next_critic_obs"][-1])
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

    return advantages


# Implementation based on GePPO repo: https://github.com/jqueeney/geppo
@torch.no_grad
def compute_gae_vtrace(data, gamma, lam, is_trunc, actor, critic):
    if actor.store_pik is False:
        raise NotImplementedError("Need to store pik for V-trace")

    log_prob = actor.get_actions_log_prob(data["actions"])
    log_prob_pik = actor.get_pik_log_prob(data["actor_obs"], data["actions"])

    # n: rollout length, e: num envs
    # TODO: Double check GePPO code and paper (they diverge imo)
    ratio = torch.exp(log_prob - log_prob_pik)  # shape [n, e]

    n, e = ratio.shape
    ones_U = torch.triu(torch.ones((n, n)), 0).to(data.device)

    ratio_trunc = torch.clamp_max(ratio, is_trunc)  # [n, e]
    ratio_trunc_T = ratio_trunc.transpose(0, 1)  # [e, n]
    ratio_trunc_repeat = ratio_trunc_T.unsqueeze(-1).repeat(1, 1, n)  # [e, n, n]
    ratio_trunc_L = torch.tril(ratio_trunc_repeat, -1)
    # cumprod along axis 1, keep shape [e, n, n]
    ratio_trunc_prods = torch.tril(torch.cumprod(ratio_trunc_L + ones_U, axis=1), 0)

    # everything in data dict is [n, e]
    values = critic.evaluate(data["critic_obs"])
    values_next = critic.evaluate(data["next_critic_obs"])
    not_done = ~data["dones"]

    delta = data["rewards"] + gamma * values_next * not_done - values  # [n, e]
    delta_T = delta.transpose(0, 1)  # [e, n]
    delta_repeat = delta_T.unsqueeze(-1).repeat(1, 1, n)  # [e, n, n]

    rate_L = torch.tril(torch.ones((n, n)) * gamma * lam, -1).to(data.device)  # [n, n]
    rates = torch.tril(torch.cumprod(rate_L + ones_U, axis=0), 0)
    rates_repeat = rates.unsqueeze(0).repeat(e, 1, 1)  # [e, n, n]
    batch_prod = torch.bmm(rates_repeat, ratio_trunc_prods)  # [e, n, n]

    # element-wise multiplication:
    intermediate = batch_prod * delta_repeat  # [e, n, n]
    advantages = torch.sum(intermediate, axis=1)  # [e, n]

    advantages = advantages.transpose(0, 1)  # [n, e]
    returns = advantages * ratio_trunc + values  # [n, e]

    return advantages, returns


# todo change num_epochs to num_batches
@torch.no_grad
def create_uniform_generator(
    data, batch_size, num_epochs=1, max_gradient_steps=None, keys=None
):
    n, m = data.shape
    total_data = n * m

    if batch_size > total_data:
        Warning("Batch size is larger than total data, using available data only.")
        batch_size = total_data

    num_batches_per_epoch = total_data // batch_size
    if max_gradient_steps:
        if max_gradient_steps < num_batches_per_epoch:
            num_batches_per_epoch = max_gradient_steps
        num_epochs = max_gradient_steps // num_batches_per_epoch
        num_epochs = max(num_epochs, 1)

    for epoch in range(num_epochs):
        indices = torch.randperm(total_data, device=data.device)
        for i in range(num_batches_per_epoch):
            batched_data = data.flatten(0, 1)[
                indices[i * batch_size : (i + 1) * batch_size]
            ]
            yield batched_data
