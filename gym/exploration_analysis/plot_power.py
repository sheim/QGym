import numpy as np
import os

smooth_name = "mini_cheetah_ref_smooth_16"
baseline_name = "mini_cheetah_ref"
colored_name = "mini_cheetah_ref_colored_1"

colored_data_dir = "./data_train/" + colored_name
smooth_data_dir = "./data_train/" + smooth_name
baseline_data_dir = "./data_train/" + baseline_name
fig_dir = "./figures_train/"

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

# load data
smooth_dof_vel = np.load(smooth_data_dir + "/dof_vel.npy")[0]
baseline_dof_vel = np.load(baseline_data_dir + "/dof_vel.npy")[0]
smooth_terminated = np.load(smooth_data_dir + "/terminated.npy")[0]
baseline_terminated = np.load(baseline_data_dir + "/terminated.npy")[0]
colored_dof_vel = np.load(colored_data_dir + "/dof_vel.npy")[0]
colored_terminated = np.load(colored_data_dir + "/terminated.npy")[0]
smooth_torques = np.load(smooth_data_dir + "/torques.npy")[0]
baseline_torques = np.load(baseline_data_dir + "/torques.npy")[0]
colored_torques = np.load(colored_data_dir + "/torques.npy")[0]

smooth_power = [[], [], [], [], [], [], [], [], [], [], [], []]
colored_power = [[], [], [], [], [], [], [], [], [], [], [], []]
baseline_power = [[], [], [], [], [], [], [], [], [], [], [], []]
for it in range(0, smooth_dof_vel.shape[0], 50):
    # only use data that didn't terminate
    if not np.any(smooth_terminated[it, :, 0]):
        for idx in range(12):
            smooth_power[idx].append(
                np.abs(
                    np.multiply(smooth_dof_vel[it, :, idx], smooth_torques[it, :, idx])
                )
            )

    if not np.any(baseline_terminated[it, :, 0]):
        for idx in range(12):
            baseline_power[idx].append(
                np.abs(
                    np.multiply(
                        baseline_dof_vel[it, :, idx], baseline_torques[it, :, idx]
                    )
                )
            )

    if not np.any(colored_terminated[it, :, 0]):
        for idx in range(12):
            colored_power[idx].append(
                np.abs(
                    np.multiply(
                        colored_dof_vel[it, :, idx], colored_torques[it, :, idx]
                    )
                )
            )

print(f"Total smooth: {len(smooth_power[0])}")
print(f"Total baseline: {len(baseline_power[0])}")
print(f"Total colored: {len(colored_power[0])}")

power_values = [
    np.array(smooth_power),
    np.array(baseline_power),
    np.array(colored_power),
]

# Calculate mean power at the beginning and end of training
# Calculate mean power at the beginning and end of training

mean_power_beginning = [power[:, 0, :].mean() for power in power_values]
mean_power_end = [power[:, -1, :].mean() for power in power_values]

print(mean_power_beginning)
print(mean_power_end)
