import torch
from tensordict import TensorDict


class TrajectoryStorage:
    def __init__(self):
        self.initialized = False

    def initialize(
        self,
        dummy_dict,
        num_envs,
        max_traj_length,
        max_num_trajs,
        device="cpu",
    ):
        assert self.initialized is False, "Storage already initialized"

        self.device = device
        self.num_envs = num_envs
        self.max_traj_length = max_traj_length
        self.max_num_trajs = max_num_trajs

        self.data = TensorDict(
            {}, batch_size=(max_traj_length, max_num_trajs), device=device
        )

        for key in dummy_dict.keys():
            if dummy_dict[key].dim() == 1:  # if scalar
                self.data[key] = torch.zeros(
                    (max_traj_length, max_num_trajs),
                    dtype=dummy_dict[key].dtype,
                    device=self.device,
                )
            else:
                self.data[key] = torch.zeros(
                    (max_traj_length, max_num_trajs, dummy_dict[key].shape[1]),
                    dtype=dummy_dict[key].dtype,
                    device=self.device,
                )

        self.time_add_index = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.traj_index = torch.tensor(
            list(range(num_envs)), dtype=torch.long, device=device
        )
        # TODO not using age now, relying on traj_list being FIFO
        self.traj_age = torch.zeros(max_num_trajs, dtype=torch.long, device=device)
        self.current_age = 1
        self.traj_length = torch.zeros(max_num_trajs, dtype=torch.long, device=device)
        self.finished_trajs = []
        self.next_available = num_envs
        self.in_overflow = False

        self.initialized = True

    @torch.inference_mode
    def add_transitions(self, transition: TensorDict):
        assert "dones" in transition.keys(), "Transition must contain `dones` key"
        # add unfinished episodes
        tdx = (~transition["dones"]).nonzero().squeeze()
        self.data[self.time_add_index[tdx], self.traj_index[tdx]] = transition[tdx]
        self.time_add_index[tdx] += 1
        # handle finished episodes
        tdx = transition["dones"].nonzero().squeeze()
        if tdx.numel() == 0:
            return
        self.data[self.time_add_index[tdx], self.traj_index[tdx]] = transition[tdx]
        self.time_add_index[tdx] += 1

        self.traj_age[self.traj_index[tdx]] = self.current_age
        self.current_age += 1

        self.traj_length[self.traj_index[tdx]] = self.time_add_index[tdx]

        self.finished_trajs.extend(self.traj_index[tdx].ravel().tolist())

        # handle indexes for new trajs
        for idx in tdx.ravel():
            self.traj_index[idx] = self.next_available
            if self.in_overflow:
                self.finished_trajs.remove(self.next_available)
                self.next_available = self.finished_trajs[0]
            else:
                self.next_available += 1
                if self.next_available >= self.max_num_trajs:
                    self.next_available = self.finished_trajs[0]
                    self.in_overflow = True
        self.time_add_index[tdx] = 0

    def get_data(self):
        # concatenate all the data into one really long tensor?
        # Easiest way to still use prior implementations (e.g. GAE)
        return torch.cat(
            [self.data[: self.traj_length[idx], idx] for idx in self.finished_trajs]
        )

    # @torch.inference_mode
    # def clear(self):
    #     self.fill_count = 0
    #     self.add_index = 0
    #     for tensor in self.data:
    #         tensor.zero_()
