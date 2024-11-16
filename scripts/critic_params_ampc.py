# Parameters for different critics
critic_params = {
    "CholeskyInput": {
        "num_obs": 10,
        "hidden_dims": [64, 128, 128, 128, 64],
        "activation": "lrelu",
        "normalize_obs": False,
        # "latent_dim": 8,  # 16,
        "minimize": True,
        "offset_hidden_dims": [64, 128, 128, 128, 64],
        "loss": "l1_loss",  # [mse_loss, l1_loss]
        "loss_type": "shape",  # [standard, sobol, shape]
        "c_offset": False,
    },
    "PDCholeskyInput": {
        "num_obs": 10,
        "hidden_dims": [64, 128, 128, 128, 64],
        "activation": "lrelu",
        "normalize_obs": False,
        "latent_dim": None,
        "minimize": True,
        "offset_hidden_dims": [64, 128, 128, 128, 64],
        "loss": "l1_loss",  # [mse_loss, l1_loss]
        "loss_type": "shape",  # [standard, sobol, shape]
        "c_offset": False,
    },
    "CholeskyLatent": {
        "num_obs": 10,
        "hidden_dims": [64, 128, 128, 128, 64],
        "activation": "lrelu",
        "normalize_obs": False,
        "minimize": True,
        "latent_dim": 8,
        "latent_hidden_dims": [64, 64],
        "latent_activation": None,
        "offset_hidden_dims": [64, 128, 128, 128, 64],
        "loss": "l1_loss",  # [mse_loss, l1_loss]
        "loss_type": "shape",  # [standard, sobol, shape]
        "c_offset": False,
    },
    "PDCholeskyLatent": {
        "num_obs": 10,
        "hidden_dims": [64, 128, 128, 128, 64],
        "activation": "lrelu",
        "normalize_obs": False,
        "minimize": True,
        "latent_dim": 8,
        "latent_hidden_dims": [64, 64],
        "latent_activation": None,
        "offset_hidden_dims": [64, 128, 128, 128, 64],
        "loss": "l1_loss",  # [mse_loss, l1_loss]
        "loss_type": "shape",  # [standard, sobol, shape]
        "c_offset": False,
    },
    "SpectralLatent": {
        "num_obs": 10,
        "hidden_dims": [64, 128, 128, 128, 64],
        "activation": "lrelu",
        "normalize_obs": False,
        "minimize": True,
        "relative_dim": 8,
        "latent_dim": 8,
        "latent_hidden_dims": [64, 64],
        "latent_activation": None,
        "offset_hidden_dims": [64, 128, 128, 128, 64],
        "loss": "l1_loss",  # [mse_loss, l1_loss]
        "loss_type": "shape",  # [standard, sobol, shape]
        "c_offset": False,
    },
    "DenseSpectralLatent": {
        "num_obs": 10,
        "hidden_dims": [64, 128, 128, 128, 64],
        "activation": "lrelu",
        "normalize_obs": False,
        "minimize": True,
        "relative_dim": 4,  # 1,
        "latent_dim": 8,  # 18,
        "latent_hidden_dims": [32, 32],
        "latent_activation": None,
        "offset_hidden_dims": [64, 128, 128, 128, 64],
        "loss": "l1_loss",  # [mse_loss, l1_loss]
        "loss_type": "shape",  # [standard, sobol, shape]
        "c_offset": False,
    },
    "Diagonal": {
        "num_obs": 10,
        "hidden_dims": [64, 128, 128, 128, 64],
        "activation": "lrelu",
        "normalize_obs": False,
        # "latent_dim": 8,  # 16,
        "minimize": True,
        "offset_hidden_dims": [64, 128, 128, 128, 64],
        "loss": "l1_loss",  # [mse_loss, l1_loss]
        "loss_type": "shape",  # [standard, sobol, shape]
        "c_offset": False,
    },
    "OuterProduct": {
        "num_obs": 10,
        "hidden_dims": [64, 128, 128, 128, 64],
        "minimize": True,
        "activation": "lrelu",
        "normalize_obs": False,
        "offset_hidden_dims": [64, 128, 128, 128, 64],
        "loss": "l1_loss",  # [mse_loss, l1_loss]
        "loss_type": "shape",  # [standard, sobol, shape]
        "c_offset": False,
    },
    "OuterProductLatent": {
        "num_obs": 10,
        "hidden_dims": [64, 128, 128, 128, 64],
        "activation": "lrelu",
        "normalize_obs": False,
        "minimize": True,
        "latent_dim": 8,
        "latent_hidden_dims": [32, 32],
        "latent_activation": None,
        "offset_hidden_dims": [64, 128, 128, 128, 64],
        "loss": "l1_loss",  # [mse_loss, l1_loss]
        "loss_type": "shape",  # [standard, sobol, shape]
        "c_offset": False,
    },
    "Critic": {
        "num_obs": 10,
        "hidden_dims": [64, 128, 128, 128, 64],
        "activation": "lrelu",
        "normalize_obs": False,
        "c_offset": False,
    },
}
