import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

WINDOW_SIZE = 20

smooth_name = "mini_cheetah_ref_smooth_16"
baseline_name = "mini_cheetah_ref"
colored_name = "mini_cheetah_ref_colored_1"

colored_data_dir = "./data_train/" + colored_name
smooth_data_dir = "./data_train/" + smooth_name
baseline_data_dir = "./data_train/" + baseline_name
fig_dir = "./figures_train/"

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)


def rolling_variance(signal, window_size):
    window = np.ones(window_size) / window_size
    signal_mean = np.convolve(signal, window, "valid")
    signal_sqr = np.convolve(signal**2, window, "valid")
    return signal_sqr - signal_mean**2


# load data
smooth_pos_obs = np.load(smooth_data_dir + "/dof_pos_obs.npy")[0]
baseline_pos_obs = np.load(baseline_data_dir + "/dof_pos_obs.npy")[0]
smooth_terminated = np.load(smooth_data_dir + "/terminated.npy")[0]
baseline_terminated = np.load(baseline_data_dir + "/terminated.npy")[0]
colored_pos_obs = np.load(colored_data_dir + "/dof_pos_obs.npy")[0]
colored_terminated = np.load(colored_data_dir + "/terminated.npy")[0]
smooth_dof_vel = np.load(smooth_data_dir + "/dof_vel.npy")[0]
baseline_dof_vel = np.load(baseline_data_dir + "/dof_vel.npy")[0]
colored_dof_vel = np.load(colored_data_dir + "/dof_vel.npy")[0]

# compute variance averages
smooth_vars = [[], [], [], [], [], [], [], [], [], [], [], []]
colored_vars = [[], [], [], [], [], [], [], [], [], [], [], []]
baseline_vars = [[], [], [], [], [], [], [], [], [], [], [], []]
total_smooth_var = [[], [], [], [], [], [], [], [], [], []]
total_baseline_var = [[], [], [], [], [], [], [], [], [], []]
total_colored_var = [[], [], [], [], [], [], [], [], [], []]

i = 0
for it in range(0, smooth_pos_obs.shape[0], 50):
    # only use data that didn't terminate
    if not np.any(smooth_terminated[it, :, 0]):
        for idx in range(12):
            # var = rolling_variance(smooth_pos_obs[it, :, idx], WINDOW_SIZE)
            total_var_obs = np.var(smooth_pos_obs[it, :, idx])
            total_var_vel = np.var(smooth_dof_vel[it, :, idx])
            total_smooth_var[i].append(total_var_obs)
            total_smooth_var[i].append(total_var_vel)
            # smooth_vars[idx].append(var)

    if not np.any(baseline_terminated[it, :, 0]):
        for idx in range(12):
            # var = rolling_variance(baseline_pos_obs[it, :, idx], WINDOW_SIZE)
            total_var_obs = np.var(baseline_pos_obs[it, :, idx])
            total_var_vel = np.var(baseline_dof_vel[it, :, idx])
            total_baseline_var[i].append(total_var_obs)
            total_baseline_var[i].append(total_var_vel)
            # baseline_vars[idx].append(var)

    if not np.any(colored_terminated[it, :, 0]):
        for idx in range(12):
            # var = rolling_variance(colored_pos_obs[it, :, idx], WINDOW_SIZE)
            total_var_obs = np.var(colored_pos_obs[it, :, idx])
            total_var_vel = np.var(colored_dof_vel[it, :, idx])
            total_colored_var[i].append(total_var_obs)
            total_colored_var[i].append(total_var_vel)
            # colored_vars[idx].append(var)
    i += 1

# print(f"Total smooth variance: {len(smooth_vars[0])}")
# print(f"Total baseline variance: {len(baseline_vars[0])}")
# print(f"Total colored variance: {len(colored_vars[0])}")

# smooth_var_means = [np.array(smooth_vars[idx]).mean(axis=0) for idx in range(3)]
# baseline_var_means = [np.array(baseline_vars[idx]).mean(axis=0) for idx in range(3)]
# colored_var_means = [np.array(colored_vars[idx]).mean(axis=0) for idx in range(3)]

# plot Variance
# fig, axs = plt.subplots(3, 1, figsize=(10, 10))
# for idx in range(3):
#    axs[idx].plot(smooth_var_means[idx], label="smooth")
#    axs[idx].plot(baseline_var_means[idx], label="baseline")
#    axs[idx].plot(colored_var_means[idx], label="colored")
#    axs[idx].set_title(f"Variance Amplitude idx {idx}")
#    axs[idx].set_xlabel("Step")
#    axs[idx].set_ylabel("Amplitude")
#    axs[idx].legend()

# fig.tight_layout()
# fig.savefig(fig_dir + "/" + "variance.png")

# Generate x-axis labels for each array
array_labels = ["Axis {}".format(i) for i in range(1, 25)]

# Plotting
plt.figure(figsize=(10, 6))

# Plot variances for baseline
plt.plot(
    array_labels,
    total_baseline_var[9],
    marker="o",
    linestyle="-",
    color="green",
    label="Baseline",
)

# Plot variances for method 1
plt.plot(
    array_labels,
    total_smooth_var[9],
    marker="o",
    linestyle="-",
    color="red",
    label="Smooth",
)

# Plot variances for method 2
plt.plot(
    array_labels,
    total_colored_var[9],
    marker="o",
    linestyle="-",
    color="blue",
    label="Colored",
)

# Add labels and legend
plt.xlabel("Axis")
plt.ylabel("Variance")
plt.title("Variances of Axis Across Different Implementations")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(fig_dir + "/" + "variance_2.png")

# Combine variances into a single dataset
variances_combined = [total_baseline_var[9], total_smooth_var[9], total_colored_var[9]]

# Create a boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=variances_combined)
plt.xlabel("Method")
plt.ylabel("Variance")
plt.title("Distribution of Variances Across Methods (Boxplot)")
plt.xticks(ticks=[0, 1, 2], labels=["Baseline", "Smooth", "Colored"])
plt.grid(True)
plt.show()

# Create a violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(data=variances_combined)
plt.xlabel("Method")
plt.ylabel("Variance")
plt.title("Distribution of Variances Across Methods (Violin Plot)")
plt.xticks(ticks=[0, 1, 2], labels=["Baseline", "Smooth", "Colored"])
plt.grid(True)
plt.show()
