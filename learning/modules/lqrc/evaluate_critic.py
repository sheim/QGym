import os

from learning import LEGGED_GYM_LQRC_DIR

from utils import critic_eval_args, get_load_path
from plotting import plot_custom_critic, plot_critic_prediction_only
from learning.modules import Critic

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
    if args["model_type"] == "CholeskyPlusConst":
        return Critic(2, standard_nn=False).to(DEVICE)
    elif args["model_type"] == "StandardMLP":
        return Critic(2, [512, 256, 128], "elu", standard_nn=True).to(DEVICE)
    else:
        raise KeyError("Specified model type is not supported for critic evaluation.")


if __name__ == "__main__":
    args = vars(critic_eval_args())
    path = get_load_path(args["experiment_name"], args["load_run"], args["checkpoint"])
    model = model_switch(args)

    loaded_dict = torch.load(path)
    critic_state_dict = filter_state_dict(loaded_dict["model_state_dict"])
    model.load_state_dict(critic_state_dict)

    lb = [0.0, -100.0]
    ub = [2.0 * torch.pi, 100.0]
    x = generate_data(lb, ub)
    x_norm = model.normalize(x)
    y_pred = []
    A_pred = []
    c_pred = []

    model.eval()
    t_c = 0
    p_c = 0
    for X_batch in x:
        y_hat = model.evaluate(X_batch.unsqueeze(0))
        y_pred.append(y_hat)
        if args["model_type"] == "CholeskyPlusConst":
            A_pred.append(model.NN.intermediates["A"])
            c_pred.append(model.NN.intermediates["c"])

    save_path = os.path.join(LEGGED_GYM_LQRC_DIR, "logs")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fn = args["fn"]

    if args["model_type"] == "CholeskyPlusConst":
        plot_custom_critic(
            x,
            torch.vstack(y_pred),
            (
                x.unsqueeze(2)
                .transpose(1, 2)
                .bmm(torch.vstack(A_pred))
                .bmm(x.unsqueeze(2))
            ).squeeze(2),
            torch.vstack(c_pred),
            save_path + f"/{fn}.png",
            contour=args["contour"],
        )
        plot_critic_prediction_only(
            x,
            torch.vstack(y_pred),
            save_path + f"/{fn}_prediction_only.png",
            contour=args["contour"],
        )
        plot_custom_critic(
            x_norm,
            torch.vstack(y_pred),
            (
                x.unsqueeze(2)
                .transpose(1, 2)
                .bmm(torch.vstack(A_pred))
                .bmm(x.unsqueeze(2))
            ).squeeze(2),
            torch.vstack(c_pred),
            save_path + f"/{fn}_normalized.png",
            contour=args["contour"],
        )
        plot_critic_prediction_only(
            x_norm,
            torch.vstack(y_pred),
            save_path + f"/{fn}_prediction_only_normalized.png",
            contour=args["contour"],
        )
    else:
        plot_critic_prediction_only(
            x, torch.vstack(y_pred), save_path + f"/{fn}.png", contour=args["contour"]
        )
        plot_critic_prediction_only(
            x_norm,
            torch.vstack(y_pred),
            save_path + f"/{fn}_normalized.png",
            contour=args["contour"],
        )
