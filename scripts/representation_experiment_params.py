# Parameters for different critics
experiment_params = {
    "Baseline": {
        "critic_name": "Critic",
        "critic_params": {
            "num_obs": 2,
            "hidden_dims": [256, 128],
            "activation": "relu",
            "normalize_obs": False,
        },
    },
    "Identity": {
        "critic_name": "PDCholeskyInput",
        "critic_params": {
            "num_obs": 2,
            "hidden_dims": [256, 128],
            "activation": "relu",
            "normalize_obs": True,
            "latent_dim": None,
            "minimize": True,
        },
    },
    "Affine": {
        "critic_name": "DenseSpectralLatent",
        "critic_params": {
            "num_obs": 2,
            "hidden_dims": [256, 128],
            "activation": "relu",
            "normalize_obs": False,
            "minimize": True,
            "relative_dim": 4,  # 1,
            "latent_dim": 16,  # 18,
            "latent_hidden_dims": [32, 32],
            "latent_activation": None,
        },
    },
    "Nonlinear": {
        "critic_name": "DenseSpectralLatent",
        "critic_params": {
            "num_obs": 2,
            "hidden_dims": [256, 128],
            "activation": "relu",
            "normalize_obs": False,
            "minimize": True,
            "relative_dim": 4,  # 1,
            "latent_dim": 16,  # 18,
            "latent_hidden_dims": [32, 32],
            "latent_activation": None,
        },
    },
}
