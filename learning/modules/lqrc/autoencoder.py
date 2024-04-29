import torch.nn as nn

from learning.modules.utils.neural_net import create_MLP


class Autoencoder(nn.Module):
    def __init__(self, num_inputs, num_latent, hidden_dims, activation="elu"):
        super().__init__()
        self.encoder = create_MLP(num_inputs, num_latent, hidden_dims, activation)
        self.decoder = create_MLP(num_latent, num_inputs, hidden_dims, activation)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
