import argparse
import os
from gym import LEGGED_GYM_ROOT_DIR
import torch


def benchmark_args():
    parser = argparse.ArgumentParser(
        description="Set hyperparameters for benchmarking custom NNs."
    )
    parser.add_argument(
        "--model_type",
        action="store",
        type=str,
        nargs="?",
        default="BaselineMLP",
        help="Name of the model type to train",
    )
    parser.add_argument(
        "--test_case",
        action="store",
        type=str,
        nargs="?",
        default="quadratic",
        help="Test case function",
    )
    parser.add_argument(
        "--input_dim",
        action="store",
        type=int,
        nargs="?",
        default=1,
        help="Dimenison of the input variables",
    )
    parser.add_argument(
        "--epochs",
        action="store",
        type=int,
        nargs="?",
        default=1000,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--split_chunk", action="store_true", help="Contiguous chunk test-train split"
    )
    parser.add_argument("--save_model", action="store_true", help="Save the model")
    parser.add_argument(
        "--colormap_diff", action="store_true", help="Save a colormap of the diff in 3D"
    )
    parser.add_argument(
        "--colormap_values",
        action="store_true",
        help="Save a colormap of the pointwise values in 3D",
    )
    try:
        args = parser.parse_args()
    except SystemExit:
        print("Exception occurred! You've likely mispelled a key.")
    return args


def critic_eval_args():
    parser = argparse.ArgumentParser(
        description="Toggle between critic evaluation settings."
    )
    parser.add_argument(
        "--model_type",
        action="store",
        type=str,
        nargs="?",
        default="StandardMLP",
        help="Name of the model type to evaluate",
    )
    parser.add_argument(
        "--experiment_name",
        action="store",
        type=str,
        nargs="?",
        help="The experiment name used when training the policy to load",
    )
    parser.add_argument(
        "--fn",
        action="store",
        type=str,
        nargs="?",
        help="Filename to use in saving graphs",
    )
    parser.add_argument(
        "--load_run",
        action="store",
        type=str,
        default=-1,
        nargs="?",
        help="Name of the run to load.",
    )
    parser.add_argument(
        "--checkpoint",
        action="store",
        type=int,
        default=-1,
        nargs="?",
        help="Save model checkpoint number.",
    )
    parser.add_argument(
        "--contour",
        action="store_true",
        help="Flag to graph contours instead of colormesh.",
    )
    try:
        args = parser.parse_args()
    except SystemExit:
        print("Exception occurred! You've likely mispelled a key.")
    return args


def autoencoder_args():
    parser = argparse.ArgumentParser(
        description="Toggle between autoencoder training settings."
    )
    parser.add_argument(
        "--input_dim",
        action="store",
        type=int,
        nargs="?",
        default=1,
        help="Dimenison of the input variables",
    )
    parser.add_argument(
        "--latent_dim",
        action="store",
        type=int,
        nargs="?",
        default=1,
        help="Dimenison of the input variables",
    )
    parser.add_argument(
        "--epochs",
        action="store",
        type=int,
        nargs="?",
        default=1000,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batches",
        action="store",
        type=int,
        nargs="?",
        default=4,
        help="Number of randomly generated batches per epoch.",
    )
    parser.add_argument(
        "--n",
        action="store",
        type=int,
        nargs="?",
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument("--save_model", action="store_true", help="Save the model")
    try:
        args = parser.parse_args()
    except SystemExit:
        print("Exception occurred! You've likely mispelled a key.")
    return args


def get_load_path(name, load_run=-1, checkpoint=-1):
    root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", name)
    run_path = select_run(root, load_run)
    model_name = select_model(run_path, checkpoint)
    load_path = os.path.join(run_path, model_name)
    return load_path


def select_run(root, load_run):
    try:
        runs = sorted(
            os.listdir(root),
            key=lambda x: os.path.getctime(os.path.join(root, x)),
        )
        if "exported" in runs:
            runs.remove("exported")
        if "videos" in runs:
            runs.remove("videos")
        last_run = os.path.join(root, runs[-1])
    except:  # ! no bare excepts!!!
        raise ValueError("No runs in this directory: " + root)

    if load_run == -1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)
    return load_run


def select_model(load_run, checkpoint):
    if checkpoint == -1:
        models = [file for file in os.listdir(load_run) if "model" in file]
        models.sort(key=lambda m: "{0:0>15}".format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)
    return model


def train(critic, optimizer, generator):
    mean_value_loss = 0
    counter = 0
    for batch in generator:
        value_loss = critic.loss_fn(
            batch["critic_obs"], batch["returns"], actions=batch["actions"]
        )
        optimizer.zero_grad()
        value_loss.backward()
        # # noqa F401.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
        optimizer.step()
        mean_value_loss += value_loss.item()
        counter += 1
    mean_value_loss /= counter

    return mean_value_loss


def train_sequentially(critic, value_opt, reg_opt, value_generator, reg_generator):
    mean_reg_loss = 0
    reg_counter = 0
    # regularize
    for batch in reg_generator:
        reg_loss = critic.reg_loss(
            batch["critic_obs"], batch["returns"], batch["actions"]
        )
        reg_opt.zero_grad()
        reg_loss.backward()
        reg_opt.step()
        mean_reg_loss += reg_loss.item()
        reg_counter += 1
    mean_reg_loss /= reg_counter
    # train value output
    mean_value_loss = train(critic.critic, value_opt, value_generator)
    return mean_value_loss, mean_reg_loss


def train_interleaved(critic, value_opt, reg_opt, value_generator, **kwargs):
    mean_value_loss = 0
    val_counter = 0
    mean_reg_loss = 0
    reg_counter = 0

    for batch in value_generator:
        # value loss
        value_loss = critic.critic.loss_fn(
            batch["critic_obs"], batch["returns"], actions=batch["actions"]
        )
        value_opt.zero_grad()
        value_loss.backward()
        # # noqa F401.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
        value_opt.step()
        mean_value_loss += value_loss.item()
        val_counter += 1

        # regularization loss
        reg_loss = critic.reg_loss(
            batch["critic_obs"], batch["returns"], batch["actions"]
        )
        reg_opt.zero_grad()
        reg_loss.backward()
        reg_opt.step()
        mean_reg_loss += reg_loss.item()
        reg_counter += 1
    mean_value_loss /= val_counter
    mean_reg_loss /= reg_counter
    return mean_value_loss, mean_reg_loss


def get_latent_matrix(input_size, model, device):
    dummy_input = torch.randn(*input_size, device=device)
    # Use torch.jit.trace to create a traced version of the model
    traced_model = torch.jit.trace(model, dummy_input)
    final_weight = traced_model.get_parameter("0.weight")
    for i in range(1, len(list(model.children()))):
        final_weight = traced_model.get_parameter(f"{i}.weight") @ final_weight

    try:
        final_bias = traced_model.get_parameter("0.bias")
        for i in range(1, len(list(model.children()))):
            final_bias = traced_model.get_parameter(
                f"{i}.weight"
            ) @ final_bias + traced_model.get_parameter(f"{i}.bias")
        return final_weight, final_bias
    except:
        print("No bias found.")

    return final_weight
