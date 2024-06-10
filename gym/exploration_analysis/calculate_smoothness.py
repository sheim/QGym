import numpy as np

# Change signal to 500 steps
smooth_name = "mini_cheetah_ref_smooth_16"
baseline_name = "mini_cheetah_ref"
colored_name = "mini_cheetah_ref_colored_1"

colored_data_dir = "./data_train/" + colored_name
smooth_data_dir = "./data_train/" + smooth_name
baseline_data_dir = "./data_train/" + baseline_name

# load data
smooth_pos_target = np.load(smooth_data_dir + "/dof_pos_target.npy")[0]
baseline_pos_target = np.load(baseline_data_dir + "/dof_pos_target.npy")[0]
smooth_terminated = np.load(smooth_data_dir + "/terminated.npy")[0]
baseline_terminated = np.load(baseline_data_dir + "/terminated.npy")[0]
colored_pos_target = np.load(colored_data_dir + "/dof_pos_target.npy")[0]
colored_terminated = np.load(colored_data_dir + "/terminated.npy")[0]

# compute FFT averages
smooth_squared_deltas = [[], [], []]
colored_squared_deltas = [[], [], []]
baseline_squared_deltas = [[], [], []]
for it in range(0, baseline_pos_target.shape[0], 50):
    # only use data that didn't terminate
    if not np.any(smooth_terminated[it, :, 0]):
        for idx in range(3):
            squared_deltas = (
                smooth_pos_target[it, 1:, idx] - smooth_pos_target[it, :-1, idx]
            ) ** 2
            smooth_squared_deltas[idx].append(squared_deltas)

    if not np.any(baseline_terminated[it, :, 0]):
        for idx in range(3):
            squared_deltas = (
                baseline_pos_target[it, 1:, idx] - baseline_pos_target[it, :-1, idx]
            ) ** 2
            baseline_squared_deltas[idx].append(squared_deltas)

    if not np.any(colored_terminated[it, :, 0]):
        for idx in range(3):
            squared_deltas = (
                colored_pos_target[it, 1:, idx] - colored_pos_target[it, :-1, idx]
            ) ** 2
            colored_squared_deltas[idx].append(squared_deltas)

smooth_squared_deltas_array = np.array(smooth_squared_deltas)
baseline_squared_deltas_array = np.array(baseline_squared_deltas)
colored_squared_deltas_array = np.array(colored_squared_deltas)

# Find the maximum value of each array
max_smooth = np.max(smooth_squared_deltas_array)
max_baseline = np.max(baseline_squared_deltas_array)
max_colored = np.max(colored_squared_deltas_array)

# Find the maximum value among the three arrays
max_squared_value = max(max_smooth, max_baseline, max_colored)

smooth_squared_deltas_scaled = np.divide(
    smooth_squared_deltas_array[:, 0, :], max_squared_value
)
baseline_squared_deltas_scaled = np.divide(
    baseline_squared_deltas_array[:, 0, :], max_squared_value
)
colored_squared_deltas_scaled = np.divide(
    colored_squared_deltas_array[:, 0, :], max_squared_value
)

# Calculate the mean of each scaled array
mean_smooth = np.mean(smooth_squared_deltas_scaled)
mean_baseline = np.mean(baseline_squared_deltas_scaled)
mean_colored = np.mean(colored_squared_deltas_scaled)

# Print the mean of each scaled array
print(f"The mean of the scaled smooth_squared_deltas array is {mean_smooth*100}")
print(f"The mean of the scaled baseline_squared_deltas array is {mean_baseline*100}")
print(f"The mean of the scaled colored_squared_deltas array is {mean_colored*100}")

smooth_squared_deltas_scaled = np.divide(
    smooth_squared_deltas_array[:, -1, :], max_squared_value
)
baseline_squared_deltas_scaled = np.divide(
    baseline_squared_deltas_array[:, -1, :], max_squared_value
)
colored_squared_deltas_scaled = np.divide(
    colored_squared_deltas_array[:, -1, :], max_squared_value
)

# Calculate the mean of each scaled array
mean_smooth = np.mean(smooth_squared_deltas_scaled)
mean_baseline = np.mean(baseline_squared_deltas_scaled)
mean_colored = np.mean(colored_squared_deltas_scaled)

# Print the mean of each scaled array
print(f"The mean of the scaled smooth_squared_deltas array is {mean_smooth*100}")
print(f"The mean of the scaled baseline_squared_deltas array is {mean_baseline*100}")
print(f"The mean of the scaled colored_squared_deltas array is {mean_colored*100}")
