import os

from learning import LEGGED_GYM_ROOT_DIR

from utils import critic_eval_args, get_load_path
from plotting import (
    plot_custom_critic,
    plot_critic_prediction_only,
)
from learning.modules import Critic
from learning.modules.lqrc.custom_critics import (
    CustomCriticBaseline, Cholesky, CholeskyPlusConst, CholeskyOffset1,
    CholeskyOffset2
)
import torch


DEVICE = "cuda:0"


def generate_data(lb, ub, steps=100):
    all_linspaces = [
        torch.linspace(lb[i], ub[i], steps, device=DEVICE) for i in range(2)
    ]
    X = torch.cartesian_prod(*all_linspaces)
    return X


def filter_state_dict(state_dict):
    critic_state_dict = {}
    for key, val in state_dict.items():
        if "critic." in key:
            critic_state_dict[key.replace("critic.", "")] = val.to(DEVICE)
    return critic_state_dict


def model_switch(args):
    if args["model_type"] == "StandardMLP":
        return Critic(2, [128, 64, 32], "tanh").to(DEVICE)
    elif args["model_type"] == "CholeskyPlusConst":
        return CholeskyPlusConst(2).to(DEVICE)
    elif args["model_type"] == "CholeskyOffset1":
        return CholeskyOffset1(2).to(DEVICE)
    elif args["model_type"] == "CholeskyOffset2":
        return CholeskyOffset2(2).to(DEVICE)
    else:
        raise KeyError("Specified model type is not supported for critic evaluation.")


if __name__ == "__main__":
    args = vars(critic_eval_args())
    model_type = args["model_type"]
    path = get_load_path(args["experiment_name"], args["load_run"], args["checkpoint"])
    model = model_switch(args)

    critic_state_dict = torch.load(path)["critic_state_dict"]
    model.load_state_dict(critic_state_dict)

    lb = [-torch.pi, -8.0]
    ub = [torch.pi, 8.0]
    x = generate_data(lb, ub)
    # x_norm = model.normalize(x)
    y_pred = []

    model.eval()
    for X_batch in x:
        y_hat = model.evaluate(X_batch.unsqueeze(0))
        y_pred.append(y_hat)

    save_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "lqrc")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fn = args["fn"]

    plot_critic_prediction_only(
            x, torch.vstack(y_pred), save_path + f"/{fn}.png", contour=args["contour"]
        )
