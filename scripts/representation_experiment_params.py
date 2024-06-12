# Parameters for different critics
experiment_params = {
    "Baseline": {
        "critic_name": "Critic",
        "critic_params": {
            "num_obs": 2,
            "hidden_dims": [32, 16],
            "activation": "relu",
            "normalize_obs": False,
        },
    },
    "Identity": {
        "critic_name": "PDCholeskyInput",
        "critic_params": {
            "num_obs": 2,
            "hidden_dims": [32, 16],
            "activation": "relu",
            "normalize_obs": True,
            "latent_dim": None,  # 16,
            "minimize": False,
        },
    },
    "Affine": {
        "critic_name": "DenseSpectralLatent",
        "critic_params": {
            "num_obs": 2,
            "hidden_dims": [32, 16],
            "activation": "relu",
            "normalize_obs": False,
            "minimize": False,
            "relative_dim": 2,  # 16,
            "latent_dim": 2,  # 18,
            "latent_hidden_dims": [64, 64],
            "latent_activation": None,
        },
    },
    "Nonlinear": {
        "critic_name": "DenseSpectralLatent",
        "critic_params": {
            "num_obs": 2,
            "hidden_dims": [32, 16],
            "activation": "relu",
            "normalize_obs": False,
            "minimize": False,
            "relative_dim": 2,  # 16,
            "latent_dim": 2,  # 18,
            "latent_hidden_dims": [64, 64],
            "latent_activation": "relu",
        },
    },
}
