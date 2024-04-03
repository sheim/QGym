import numpy as np
import matplotlib.pyplot as plt

SMOOTH = True
name = "ref_sample_8"
data_dir = "./data/train/" + name
fig_dir = "./figures/train"

# load data
dof_pos_obs = np.load(data_dir + "/dof_pos_obs.npy")[0]
dof_pos_target = np.load(data_dir + "/dof_pos_target.npy")[0]
dof_vel = np.load(data_dir + "/dof_vel.npy")[0]
torques = np.load(data_dir + "/torques.npy")[0]

iterations = range(0, dof_pos_obs.shape[0], 10)

# plot data for each iteration
for it in iterations:
    fig, axs = plt.subplots(4, figsize=(10, 10))
    plt.suptitle(name + " iteration " + str(it))

    for i in range(3):
        axs[0].plot(dof_pos_obs[it, :, i])
        axs[0].set_title("dof_pos_obs")
        axs[0].legend(["idx 0", "idx 1", "idx 2"])
        axs[0].set_xlabel("time steps")

    for i in range(3):
        axs[1].plot(dof_pos_target[it, :, i])
        axs[1].set_title("dof_pos_target")
        axs[1].legend(["idx 0", "idx 1", "idx 2"])
        axs[1].set_xlabel("time steps")

    for i in range(3):
        axs[2].plot(dof_vel[it, :, i])
        axs[2].set_title("dof_vel")
        axs[2].legend(["idx 0", "idx 1", "idx 2"])
        axs[2].set_xlabel("time steps")

    for i in range(3):
        axs[3].plot(torques[it, :, i])
        axs[3].set_title("torques")
        axs[3].legend(["idx 0", "idx 1", "idx 2"])
        axs[3].set_xlabel("time steps")

    if SMOOTH:
        # plot vertical lines where noise is resampled
        for x in range(0, dof_pos_obs.shape[1], 8):
            axs[0].axvline(x, color="r", linestyle="--")
            axs[1].axvline(x, color="r", linestyle="--")
            axs[2].axvline(x, color="r", linestyle="--")
            axs[3].axvline(x, color="r", linestyle="--")

    plt.tight_layout()
    plt.savefig(fig_dir + "/" + name + "_it_" + str(it) + ".png")
