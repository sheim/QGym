import torch
import os
import copy


def create_MLP(
    num_inputs,
    num_outputs,
    hidden_dims,
    activations,
    dropouts=None,
    just_list=False,
    bias_in_linear_layers=True,
):
    if not isinstance(activations, list):
        activations = [activations] * len(hidden_dims)
    if dropouts is None:
        dropouts = [0] * len(hidden_dims)
    elif not isinstance(dropouts, list):
        dropouts = [dropouts] * len(hidden_dims)
    layers = []
    # first layer
    if len(hidden_dims) > 0:
        add_layer(
            layers,
            num_inputs,
            hidden_dims[0],
            activations[0],
            dropouts[0],
            bias_in_linear_layers=bias_in_linear_layers,
        )
        for i in range(len(hidden_dims) - 1):
            add_layer(
                layers,
                hidden_dims[i],
                hidden_dims[i + 1],
                activations[i + 1],
                dropouts[i + 1],
                bias_in_linear_layers=bias_in_linear_layers,
            )
        else:
            add_layer(
                layers,
                hidden_dims[-1],
                num_outputs,
                bias_in_linear_layers=bias_in_linear_layers,
            )
    else:  # handle no hidden dims, just linear layer
        add_layer(
            layers, num_inputs, num_outputs, bias_in_linear_layers=bias_in_linear_layers
        )

    if just_list:
        return layers
    else:
        return torch.nn.Sequential(*layers)


def add_layer(
    layer_list,
    num_inputs,
    num_outputs,
    activation=None,
    dropout=0,
    bias_in_linear_layers=True,
):
    layer_list.append(
        torch.nn.Linear(num_inputs, num_outputs, bias=bias_in_linear_layers)
    )
    if dropout > 0:
        layer_list.append(torch.nn.Dropout(p=dropout))
    if activation is not None:
        layer_list.append(get_activation(activation))


def get_activation(act_name):
    if act_name == "elu":
        return torch.nn.ELU()
    elif act_name == "selu":
        return torch.nn.SELU()
    elif act_name == "relu":
        return torch.nn.ReLU()
    elif act_name == "crelu":
        return torch.nn.CELU()
    elif act_name == "lrelu":
        return torch.nn.LeakyReLU()
    elif act_name == "tanh":
        return torch.nn.Tanh()
    elif act_name == "sigmoid":
        return torch.nn.Sigmoid()
    elif act_name == "softplus":
        return torch.nn.Softplus()
    elif act_name == "softmax":
        return torch.nn.Softmax(dim=-1)
    elif act_name == "randrelu":
        return torch.nn.RReLU()
    elif act_name == "softsign":
        return torch.nn.Softsign()
    elif act_name == "mish":
        return torch.nn.Mish()
    else:
        print("invalid activation function!")
        return None


def export_network(network, network_name, path, num_inputs):
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
    model.device = "cpu"  # force all tensors created as intermediate steps to CPU
    # To trace model, must be evaluated once with arbitrary input
    model.eval()
    dummy_input = torch.rand((2, num_inputs))
    model_traced = torch.jit.trace(model, dummy_input)
    torch.jit.save(model_traced, path_TS)
    # torch.onnx.export(model_traced, dummy_input, path_onnx)
