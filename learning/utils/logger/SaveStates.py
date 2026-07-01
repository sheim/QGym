import torch
import matplotlib.pyplot as plt


def init_env_log_buffers(env, state_list, num_timesteps):
    """Initialize logging buffers for states in state_list to log num_timesteps times"""
    # set log buffers of size [num_envs * num_timesteps, num_observations]
    setattr(env, "_buffer_size_dict", {})
    for state in state_list:
        state_buffer_shape = getattr(env, state).shape
        if getattr(env, state).ndim == 1:
            state_buffer_shape = torch.Tensor([1, state_buffer_shape[0]])

        env._buffer_size_dict[state] = state_buffer_shape
        setattr(
            env,
            state + "_log_buffer",
            torch.zeros([state_buffer_shape[0] * num_timesteps, state_buffer_shape[1]]),
        )
    # keep track of how many times states are logged
    setattr(env, "_num_log_timesteps", torch.tensor([0], dtype=torch.int32))
    setattr(
        env, "_max_num_log_timesteps", torch.tensor([num_timesteps], dtype=torch.int32)
    )


def save_to_log_buffers(env, state_list):
    """Log states in state_list to the logging buffers"""
    t = env._num_log_timesteps
    if t >= env._max_num_log_timesteps:
        print(
            "No longer logging states, max timestep in logging buffers reached: "
            + str(t.item())
        )
        return
    for state in state_list:
        buffer_size = env._buffer_size_dict[state]
        log_buffer = getattr(env, state + "_log_buffer")
        tensor = getattr(env, state)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        log_buffer[t * buffer_size[0] : (t + 1) * buffer_size[0], :] = tensor
    env._num_log_timesteps += 1


def save_histogram_from_dict(state_dict):
    """Saves the states listed in state_dict and plots 1 histogram per column"""
    for state in state_dict:
        data = state_dict[state].cpu().numpy()
        if data.ndim == 1:
            plt.figure()
            plt.hist(data, bins=20)
            plt.ylabel("Occurrences")
            plt.title(state[:-11] + " distribution")
            plt.savefig(state[:-11] + ".png")
            return
        else:
            for col in range(data.shape[1]):
                col_data = data[:, col].reshape(-1, 1)
                plt.figure()
                plt.hist(col_data, bins=20)
                plt.ylabel("Occurrences")
                plt.title(
                    state
                    + " col "
                    + str(col)
                    + " of "
                    + str(data.shape[1] - 1)
                    + " distribution"
                )
                plt.savefig(state + "_col_" + str(col) + ".png")


def save_histogram_from_env(env, state_list):
    """Saves the states listed in state_list of environment env,
    plots 1 histogram per column"""
    state_dict = {}
    for state in state_list:
        log_buffer = getattr(env, state + "_log_buffer")
        num_rows = env._num_log_timesteps * env._buffer_size_dict[state][0]
        log_buffer = log_buffer[:num_rows]
        state_dict[state] = log_buffer
    save_histogram_from_dict(state_dict)
    torch.save(state_dict, "saved_states.pt")
    print("Saved states to saved_states.pt")
