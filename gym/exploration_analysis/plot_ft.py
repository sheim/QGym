import numpy as np
import matplotlib.pyplot as plt
import os

# Change signal to 500 steps
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
smooth_pos_target = np.load(smooth_data_dir + "/dof_pos_target.npy")[0]
baseline_pos_target = np.load(baseline_data_dir + "/dof_pos_target.npy")[0]
smooth_terminated = np.load(smooth_data_dir + "/terminated.npy")[0]
baseline_terminated = np.load(baseline_data_dir + "/terminated.npy")[0]
colored_pos_target = np.load(colored_data_dir + "/dof_pos_target.npy")[0]
colored_terminated = np.load(colored_data_dir + "/terminated.npy")[0]

# compute FFT averages
smooth_ffts = [[], [], [], [], [], [], [], [], [], [], [], []]
colored_ffts = [[], [], [], [], [], [], [], [], [], [], [], []]
baseline_ffts = [[], [], [], [], [], [], [], [], [], [], [], []]
for it in range(0, baseline_pos_target.shape[0], 50):
    # only use data that didn't terminate
    if not np.any(smooth_terminated[it, :, 0]):
        for idx in range(12):
            fft = np.fft.fft(smooth_pos_target[it, :, idx])
            smooth_ffts[idx].append(fft[: len(fft) // 2])

    if not np.any(baseline_terminated[it, :, 0]):
        for idx in range(12):
            fft = np.fft.fft(baseline_pos_target[it, :, idx])
            baseline_ffts[idx].append(fft[: len(fft) // 2])

    if not np.any(colored_terminated[it, :, 0]):
        for idx in range(12):
            fft = np.fft.fft(colored_pos_target[it, :, idx])
            colored_ffts[idx].append(fft[: len(fft) // 2])

print(f"Total smooth FFTS: {len(smooth_ffts[0])}")
print(f"Total baseline FFTS: {len(baseline_ffts[0])}")
print(f"Total colored FFTS: {len(colored_ffts[0])}")

smooth_fft_means = [np.array(smooth_ffts[idx]).mean(axis=0) for idx in range(12)]
baseline_fft_means = [np.array(baseline_ffts[idx]).mean(axis=0) for idx in range(12)]
colored_fft_means = [np.array(colored_ffts[idx]).mean(axis=0) for idx in range(12)]


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


x_values = np.linspace(0, 50, 498)
# plot FFTs
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for idx in range(2):
    colored_smooth_start = moving_average(
        np.array(np.abs(colored_ffts))[:, 0, :].mean(axis=0)
    )
    baseline_smooth_start = moving_average(
        np.array(np.abs(baseline_ffts))[:, 0, :].mean(axis=0)
    )
    sde_smooth_start = moving_average(
        np.array(np.abs(smooth_ffts))[:, 0, :].mean(axis=0)
    )
    colored_smooth_end = moving_average(
        np.array(np.abs(colored_ffts))[:, -1, :].mean(axis=0)
    )
    baseline_smooth_end = moving_average(
        np.array(np.abs(baseline_ffts))[:, -1, :].mean(axis=0)
    )
    sde_smooth_end = moving_average(
        np.array(np.abs(smooth_ffts))[:, -1, :].mean(axis=0)
    )

    if idx == 0:
        axs[idx].plot(x_values, colored_smooth_start, label="Pink", color="blue")
        axs[idx].plot(x_values, baseline_smooth_start, label="Baseline", color="green")
        axs[idx].plot(x_values, sde_smooth_start, label="gSDE-16", color="red")
        axs[idx].set_title("Fourier Transform at the Beginning of Training")
        axs[idx].set_xlabel("Frequency [Hz]")
        axs[idx].set_ylabel("Amplitude")
        axs[idx].legend()
        axs[idx].set_ylim([-1, 40])

    else:
        axs[idx].plot(x_values, colored_smooth_end, label="Pink", color="blue")
        axs[idx].plot(x_values, baseline_smooth_end, label="Baseline", color="green")
        axs[idx].plot(x_values, sde_smooth_end, label="gSDE-16", color="red")
        axs[idx].set_title("Fourier Transform at the End of Training")
        axs[idx].set_xlabel("Frequency [Hz]")
        axs[idx].set_ylabel("Amplitude")
        axs[idx].legend()
        axs[idx].set_ylim([-1, 40])

fig.tight_layout()
fig.savefig(fig_dir + "/" + "fourier.png")
