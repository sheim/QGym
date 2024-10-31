# Parameters for different critics
critic_params = {
    "CholeskyInput": {
        "num_obs": 10,
        "hidden_dims": [256, 128],
        "activation": "tanh",
        "normalize_obs": True,
        # "latent_dim": 16,  # 16,
        "minimize": True,
        "offset_hidden_dims": [64],
        "loss_fn": "mse_loss",  # [mse_loss, l1_loss]
        "loss_type": "sobol",  # [standard, sobol, shape]
    },
    "PDCholeskyInput": {
        "num_obs": 10,
        "hidden_dims": [256, 128],
        "activation": "tanh",
        "normalize_obs": False,
        "latent_dim": None,
        "minimize": True,
        "offset_hidden_dims": [64],
        "loss_fn": "mse_loss",  # [mse_loss, l1_loss]
        "loss_type": "sobol",  # [standard, sobol, shape]
    },
    "CholeskyLatent": {
        "num_obs": 10,
        "hidden_dims": [256, 128],
        "activation": "tanh",
        "normalize_obs": False,
        "minimize": True,
        "latent_dim": 16,
        "latent_hidden_dims": [64, 64],
        "latent_activation": None,
        "offset_hidden_dims": [64],
        "loss_fn": "mse_loss",  # [mse_loss, l1_loss]
        "loss_type": "sobol",  # [standard, sobol, shape]
    },
    "PDCholeskyLatent": {
        "num_obs": 10,
        "hidden_dims": [256, 128],
        "activation": "tanh",
        "normalize_obs": False,
        "minimize": True,
        "latent_dim": 16,
        "latent_hidden_dims": [64, 64],
        "latent_activation": None,
        "offset_hidden_dims": [64],
        "loss_fn": "mse_loss",  # [mse_loss, l1_loss]
        "loss_type": "sobol",  # [standard, sobol, shape]
    },
    "SpectralLatent": {
        "num_obs": 10,
        "hidden_dims": [256, 128],
        "activation": "tanh",
        "normalize_obs": False,
        "minimize": True,
        "relative_dim": 8,
        "latent_dim": 16,
        "latent_hidden_dims": [64, 64],
        "latent_activation": None,
        "offset_hidden_dims": [64],
        "loss_fn": "mse_loss",  # [mse_loss, l1_loss]
        "loss_type": "sobol",  # [standard, sobol, shape]
    },
    "DenseSpectralLatent": {
        "num_obs": 10,
        "hidden_dims": [256, 128],
        "activation": "tanh",
        "normalize_obs": False,
        "minimize": True,
        "relative_dim": 4,  # 1,
        "latent_dim": 16,  # 18,
        "latent_hidden_dims": [32, 32],
        "latent_activation": None,
        "offset_hidden_dims": [64],
        "loss_fn": "mse_loss",  # [mse_loss, l1_loss]
        "loss_type": "sobol",  # [standard, sobol, shape]
    },
    "OuterProduct": {
        "num_obs": 10,
        "hidden_dims": [256, 128],
        "minimize": True,
        "activation": "tanh",
        "normalize_obs": False,
        "offset_hidden_dims": [64],
        "loss_fn": "mse_loss",  # [mse_loss, l1_loss]
        "loss_type": "sobol",  # [standard, sobol, shape]
    },
    "OuterProductLatent": {
        "num_obs": 10,
        "hidden_dims": [256, 128],
        "activation": "tanh",
        "normalize_obs": False,
        "minimize": True,
        "latent_dim": 16,
        "latent_hidden_dims": [32, 32],
        "latent_activation": None,
        "offset_hidden_dims": [64],
        "loss_fn": "mse_loss",  # [mse_loss, l1_loss]
        "loss_type": "sobol",  # [standard, sobol, shape]
    },
    "Critic": {
        "num_obs": 10,
        "hidden_dims": [256, 128],
        "activation": "tanh",
        "normalize_obs": False,
    },
}
