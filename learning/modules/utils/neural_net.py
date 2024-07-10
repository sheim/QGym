import torch
import os
import copy
import numpy as np


def create_MLP(num_inputs, num_outputs, hidden_dims, activation, dropouts=None):
    activation = get_activation(activation)

    if dropouts is None:
        dropouts = [0] * len(hidden_dims)

    layers = []
    if not hidden_dims:  # handle no hidden layers
        add_layer(layers, num_inputs, num_outputs)
    else:
        add_layer(layers, num_inputs, hidden_dims[0], activation, dropouts[0])
        for i in range(len(hidden_dims)):
            if i == len(hidden_dims) - 1:
                add_layer(layers, hidden_dims[i], num_outputs)
            else:
                add_layer(
                    layers,
                    hidden_dims[i],
                    hidden_dims[i + 1],
                    activation,
                    dropouts[i + 1],
                )
    return torch.nn.Sequential(*layers)


def get_activation(act_name):
    if act_name == "elu":
        return torch.nn.ELU()
    elif act_name == "selu":
        return torch.nn.SELU()
    elif act_name == "relu":
        return torch.nn.ReLU()
    elif act_name == "crelu":
        return torch.nn.ReLU()
    elif act_name == "lrelu":
        return torch.nn.LeakyReLU()
    elif act_name == "tanh":
        return torch.nn.Tanh()
    elif act_name == "sigmoid":
        return torch.nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


def add_layer(layer_list, num_inputs, num_outputs, activation=None, dropout=0):
    layer_list.append(torch.nn.Linear(num_inputs, num_outputs))
    if dropout > 0:
        layer_list.append(torch.nn.Dropout(p=dropout))
    if activation is not None:
        layer_list.append(activation)


def export_network(network, network_name, path, num_inputs, latent=False):
    """
    Thsi function traces and exports the given network module in .pt and
    .onnx file formats. These can be used for evaluation on other systems
    without needing a Pytorch environment.

    :param network:         PyTorch neural network module
    :param network_name:    (string) Network will be saved with this name
    :path:                  (string) Network will be saved to this location
    :param num_inputs:      (int) Number of inputs to the network module
    """

    os.makedirs(path, exist_ok=True)
    path_TS = os.path.join(path, network_name + ".pt")  # TorchScript path
    path_onnx = os.path.join(path, network_name + ".onnx")  # ONNX path
    model = copy.deepcopy(network).to("cpu")
    # To trace model, must be evaluated once with arbitrary input
    model.eval()
    dummy_input = torch.rand((num_inputs))
    model_traced = torch.jit.trace(model, dummy_input)
    torch.jit.save(model_traced, path_TS)
    torch.onnx.export(model_traced, dummy_input, path_onnx)

    if latent:
        # Export latent model
        path_latent = os.path.join(path, network_name + "_latent.onnx")
        model_latent = torch.nn.Sequential(model.obs_rms, model.latent_net)
        model_latent.eval()
        dummy_input = torch.rand((num_inputs))
        model_traced = torch.jit.trace(model_latent, dummy_input)
        torch.onnx.export(model_traced, dummy_input, path_latent)

        # Save actor std of shape (num_actions, latent_dim)
        # It is important that the shape is the same as the exploration matrix
        path_std = os.path.join(path, network_name + "_std.txt")
        std_transposed = model.get_std.numpy().T
        np.savetxt(path_std, std_transposed)
