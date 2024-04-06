import numpy as np
import matplotlib.pyplot as plt
import os

FOURIER = True
SMOOTH = True
SAMPLE_FREQ = 16
STEPS = 1000

name = "ref_sample_16_len_1000"
data_dir = "./data_train/" + name
fig_dir = "./figures_train/" + name

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

# load data
dof_pos_obs = np.load(data_dir + "/dof_pos_obs.npy")[0]
dof_pos_target = np.load(data_dir + "/dof_pos_target.npy")[0]
dof_vel = np.load(data_dir + "/dof_vel.npy")[0]
torques = np.load(data_dir + "/torques.npy")[0]
terminated = np.load(data_dir + "/terminated.npy")[0]


# plot fourier trainsform
def plot_fourier(data, it):
    fig_ft, axs_ft = plt.subplots(2, figsize=(10, 10))
    for i in range(3):
        ft = np.fft.fft(data[:, i])
        ft_half = ft[: len(ft) // 2]
        axs_ft[0].plot(np.abs(ft_half))
        axs_ft[1].plot(np.angle(ft_half))

    axs_ft[0].set_title("FT Amplitude")
    axs_ft[0].set_xlabel("Frequency")
    axs_ft[0].set_ylabel("Amplitude")
    axs_ft[0].legend(["idx 0", "idx 1", "idx 2"])
    axs_ft[1].set_title("FT Phase")
    axs_ft[1].set_xlabel("Frequency")
    axs_ft[1].set_ylabel("Phase")
    axs_ft[1].legend(["idx 0", "idx 1", "idx 2"])

    fig_ft.savefig(fig_dir + "/dof_pos_target_FT_it_" + str(it) + ".png")


# plot data for each iteration
for it in range(0, dof_pos_obs.shape[0], 10):
    # check if iteration terminated
    terminate_idx = np.where(terminated[it, :, 0] == 1)[0]
    if terminate_idx.size > 0:
        n_steps = terminate_idx[0]
    else:
        n_steps = dof_pos_obs.shape[1]
    print(n_steps)

    # generate figure
    fig, axs = plt.subplots(4, figsize=(10, 10))
    plt.suptitle(name + " iteration " + str(it))

    axs[0].set_title("dof_pos_obs")
    for i in range(3):
        axs[0].plot(dof_pos_obs[it, :n_steps, i])

    axs[1].set_title("dof_pos_target")
    for i in range(3):
        axs[1].plot(dof_pos_target[it, :n_steps, i])
    if FOURIER and n_steps == STEPS:
        plot_fourier(dof_pos_target[it, :n_steps, :], it)

    axs[2].set_title("dof_vel")
    for i in range(3):
        axs[2].plot(dof_vel[it, :n_steps, i])

    axs[3].set_title("torques")
    for i in range(3):
        axs[3].plot(torques[it, :n_steps, i])

    # format plots
    for idx in range(4):
        axs[idx].legend(["idx 0", "idx 1", "idx 2"])
        axs[idx].set_xlabel("time steps")
        axs[idx].set_xlim([0, n_steps])

    if SMOOTH:
        # plot vertical lines where noise is resampled
        for x in range(0, dof_pos_obs.shape[1], SAMPLE_FREQ):
            axs[0].axvline(x, color="r", linestyle="--")
            axs[1].axvline(x, color="r", linestyle="--")
            axs[2].axvline(x, color="r", linestyle="--")
            axs[3].axvline(x, color="r", linestyle="--")

    fig.tight_layout()
    fig.savefig(fig_dir + "/" + name + "_it_" + str(it) + ".png")
