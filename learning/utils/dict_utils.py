import torch
from tensordict import TensorDict


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
    return


def compute_generalized_advantages(data, gamma, lam, critic, last_values=None):
    if "values" not in data.keys():
        data.update({"values": critic.evaluate(data["critic_obs"])})

    if "advantages" not in data.keys():
        data.update({"advantages": torch.zeros_like(data["values"])})
    else:
        data["advantages"].zero_()

    if last_values is not None:
        # todo check this
        # since we don't have observations for the last step, need last value plugged in
        data["advantages"][-1] = (
            data["rewards"][-1]
            + gamma * last_values * ~data["dones"][-1]
            - data["values"][-1]
        )

    for k in reversed(range(data["values"].shape[0] - 1)):
        not_done = ~data["dones"][k]
        td_error = (
            data["rewards"][k]
            + gamma * data["values"][k + 1] * not_done
            - data["values"][k]
        )
        data["advantages"][k] = (
            td_error + gamma * lam * not_done * data["advantages"][k + 1]
        )
    # data["returns"] = data["advantages"] + data["values"]
    data["advantages"] = (data["advantages"] - data["advantages"].mean()) / (
        data["advantages"].std() + 1e-8
    )
