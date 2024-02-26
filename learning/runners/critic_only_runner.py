import torch
from .data_logging_runner import DataLoggingRunner


class CriticOnlyRunner(DataLoggingRunner):
    def load(self, path):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.actor.load_state_dict(
            loaded_dict["actor_state_dict"])
