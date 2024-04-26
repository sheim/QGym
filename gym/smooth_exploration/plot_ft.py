import numpy as np
import matplotlib.pyplot as plt
import os

SAMPLE_FREQ = 8
STEPS = 500

smooth_name = "ref_sample_8_len_500"
baseline_name = "ref_baseline_len_500"

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

# compute FFT averages
smooth_ffts = [[], [], []]
baseline_ffts = [[], [], []]
for it in range(0, smooth_pos_target.shape[0], 10):
    # only use data that didn't terminate
    if not np.any(smooth_terminated[it, :, 0]):
        for idx in range(3):
            fft = np.fft.fft(smooth_pos_target[it, :, idx])
            smooth_ffts[idx].append(fft[: len(fft) // 2])

    if not np.any(baseline_terminated[it, :, 0]):
        for idx in range(3):
            fft = np.fft.fft(baseline_pos_target[it, :, idx])
            baseline_ffts[idx].append(fft[: len(fft) // 2])

print(f"Total smooth FFTS: {len(smooth_ffts[0])}")
print(f"Total baseline FFTS: {len(baseline_ffts[0])}")

smooth_fft_means = [np.array(smooth_ffts[idx]).mean(axis=0) for idx in range(3)]
baseline_fft_means = [np.array(baseline_ffts[idx]).mean(axis=0) for idx in range(3)]

# plot FFTs
fig, axs = plt.subplots(3, 1, figsize=(10, 10))
for idx in range(3):
    axs[idx].plot(np.abs(smooth_fft_means[idx]))
    axs[idx].plot(np.abs(baseline_fft_means[idx]))
    axs[idx].set_title(f"FT Amplitude idx {idx}")
    axs[idx].set_xlabel("Frequency")
    axs[idx].set_ylabel("Amplitude")
    axs[idx].legend(["smooth", "baseline"])

fig.tight_layout()
fig.savefig(fig_dir + "/" + smooth_name + ".png")
