import numpy as np
import matplotlib.pyplot as plt
import os

log_path = os.path.join(os.getcwd(), "joint_logs.npz")
if not os.path.exists(log_path):
    raise FileNotFoundError(f"Could not find log file at {log_path}")

data = np.load(log_path, allow_pickle=True)
steps = data["step"]
target_pos = data["target_pos"]
actual_pos = data["actual_pos"]
torques = data["torque"]
joint_names = data["joint_name"]


num_joints = len(joint_names)
num_steps = len(steps)
env_id = 0  # only visualize env 0

print(f"Loaded {num_joints} joints, {num_steps} steps from environment {env_id}.")

fig, axs = plt.subplots(num_joints * 2, 1, figsize=(10, 4 * num_joints), sharex=True)
fig.suptitle("Joint Target/Actual Positions and Torques Over Time", fontsize=16)

for j in range(num_joints):
    pos_ax = axs[2 * j]
    torque_ax = axs[2 * j + 1]

    # Find last index with nonzero torque (or position)
    nonzero_idx = np.where(np.abs(actual_pos[env_id, :, j]) > 1e-6)[0]
    if len(nonzero_idx) > 0:
        last_valid = nonzero_idx[-1] + 1
    else:
        last_valid = len(steps)

    # --- Position subplot ---
    pos_ax.plot(
        steps[:last_valid],
        target_pos[env_id, :last_valid, j],
        label="Target Pos",
        linestyle="--",
        linewidth=1.5,
    )
    pos_ax.plot(
        steps[:last_valid],
        actual_pos[env_id, :last_valid, j],
        label="Actual Pos",
        linewidth=1.5,
    )
    pos_ax.set_ylabel(f"{joint_names[j]} Pos (rad)")
    pos_ax.legend(loc="upper right", fontsize=8)
    pos_ax.grid(True, linestyle="--", alpha=0.4)

    # --- Torque subplot ---
    torque_ax.plot(
        steps[:last_valid],
        torques[env_id, :last_valid, j],
        color="tab:red",
        linewidth=1.2,
    )
    torque_ax.set_ylabel(f"{joint_names[j]} Torque (Nm)")
    torque_ax.grid(True, linestyle="--", alpha=0.4)

axs[-1].set_xlabel("Simulation Step")
plt.tight_layout(rect=[0, 0, 1, 0.96])

save_path = os.path.join(os.getcwd(), "joint_data_all_joints_separated.png")
plt.savefig(save_path, dpi=300)
print(f"Saved separated plot to {save_path}")
