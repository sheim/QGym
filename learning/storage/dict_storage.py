import torch
from tensordict import TensorDict


class DictStorage:
    def __init__(self):
        self.initialized = False

    def initialize(
        self,
        dummy_dict,
        num_envs=2**12,
        max_storage=2**17,
        device="cpu",
    ):
        self.device = device
        self.num_envs = num_envs
        max_length = max_storage // num_envs
        self.max_length = max_length
        self.data = TensorDict({}, batch_size=(max_length, num_envs), device=device)
        self.fill_count = 0

        for key in dummy_dict.keys():
            if dummy_dict[key].dim() == 1:  # if scalar
                self.data[key] = torch.zeros(
                    (max_length, num_envs),
                    device=self.device,
                )
            else:
                self.data[key] = torch.zeros(
                    (max_length, num_envs, dummy_dict[key].shape[1]),
                    device=self.device,
                )

    def add_transitions(self, transition: TensorDict):
        if self.fill_count >= self.max_length:
            raise AssertionError("Rollout buffer overflow")
        self.data[self.fill_count] = transition
        self.fill_count += 1

    def clear(self):
        self.fill_count = 0

    def mini_batch_generator(self, keys, mini_batch_size, num_epochs):
        pass
