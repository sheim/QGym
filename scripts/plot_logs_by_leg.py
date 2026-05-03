import numpy as np
import matplotlib.pyplot as plt
import os

# config
LOG_FILE = "joint_logs.npz"
ENV_ID = 0
SAVE_DIR = "plots_by_leg"

# output folder
os.makedirs(SAVE_DIR, exist_ok=True)

data = np.load(LOG_FILE, allow_pickle=True)
steps = data["step"]
target_pos = data["target_pos"]
actual_pos = data["actual_pos"]
torques = data["torque"]
joint_names = data["joint_name"]

num_envs, num_steps, num_joints = actual_pos.shape
print(f"Loaded log for {num_joints} joints, {num_steps} steps.")

# define grouping
groups = {
    "base": [0],
    "right_hind": list(range(1, 6)),
    "left_hind": list(range(6, 11)),
    "right_front": list(range(11, 16)),
    "left_front": list(range(16, 21)),
}


def plot_joint_group(group_name, joint_indices):
    num_joints_group = len(joint_indices)
    fig, axs = plt.subplots(
        nrows=num_joints_group * 2,
        ncols=1,
        figsize=(10, 3 * num_joints_group * 2),
        constrained_layout=True,
    )

    if num_joints_group == 1:
        axs = np.array([axs[0], axs[1]])  # ensure consistent 2D array shape

    for idx, j in enumerate(joint_indices):
        pos_ax = axs[2 * idx]
        torque_ax = axs[2 * idx + 1]

        # find last index with nonzero torque (or position)
        nonzero_idx = np.where(np.abs(actual_pos[ENV_ID, :, j]) > 1e-6)[0]
        if len(nonzero_idx) > 0:
            last_valid = nonzero_idx[-1] + 1
        else:
            last_valid = len(steps)

        # position
        pos_ax.plot(
            steps[:last_valid],
            target_pos[ENV_ID, :last_valid, j],
            label="Target Pos",
            linestyle="--",
            linewidth=1.5,
        )
        pos_ax.plot(
            steps[:last_valid],
            actual_pos[ENV_ID, :last_valid, j],
            label="Actual Pos",
            linewidth=1.5,
        )
        pos_ax.set_ylabel(f"{joint_names[j]} Pos (rad)")
        pos_ax.legend(loc="upper right", fontsize=8)
        pos_ax.grid(True, linestyle="--", alpha=0.4)

        # torque
        torque_ax.plot(
            steps[:last_valid],
            torques[ENV_ID, :last_valid, j],
            color="tab:red",
            linewidth=1.2,
        )
        torque_ax.set_ylabel(f"{joint_names[j]} Torque (Nm)")
        torque_ax.grid(True, linestyle="--", alpha=0.4)

    axs[-1].set_xlabel("Step")
    plt.suptitle(f"{group_name.capitalize()} Joint Logs", fontsize=14)
    save_path = os.path.join(SAVE_DIR, f"{group_name}.png")
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved {group_name}.png")


# generate plots
for group_name, joint_indices in groups.items():
    if max(joint_indices) < num_joints:
        plot_joint_group(group_name, joint_indices)
    else:
        print(f"Skipping {group_name}: exceeds joint index range.")
