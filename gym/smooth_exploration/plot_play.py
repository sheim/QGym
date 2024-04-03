import numpy as np
import matplotlib.pyplot as plt

name = "mini_cheetah_ref"
data_dir = "./data/play/" + name
fig_dir = "./figures/play"

# load data
dof_pos_obs = np.load(data_dir + "/dof_pos_obs.npy")[0]
dof_pos_target = np.load(data_dir + "/dof_pos_target.npy")[0]
dof_vel = np.load(data_dir + "/dof_vel.npy")[0]

# plot data
n_steps = 200
fig, axs = plt.subplots(3, figsize=(10, 10))
plt.suptitle(name)

for i in range(3):
    axs[0].plot(dof_pos_obs[:n_steps, i])
axs[0].set_title("dof_pos_obs")
axs[0].legend(["idx 0", "idx 1", "idx 2"])
axs[0].set_xlabel("time steps")

for i in range(3):
    axs[1].plot(dof_pos_target[:n_steps, i])
axs[1].set_title("dof_pos_target")
axs[1].legend(["idx 0", "idx 1", "idx 2"])
axs[1].set_xlabel("time steps")

for i in range(3):
    axs[2].plot(dof_vel[:n_steps, i])
axs[2].set_title("dof_vel")
axs[2].legend(["idx 0", "idx 1", "idx 2"])
axs[2].set_xlabel("time steps")

plt.tight_layout()
plt.savefig(fig_dir + "/" + name + ".png")
plt.show()
