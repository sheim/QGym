import numpy as np
import matplotlib.pyplot as plt
import os
import re

# config
LOG_FILE = "joint_logs.npz"
ENV_ID = 0
SAVE_DIR = "plots_by_joint"

# output folder
os.makedirs(SAVE_DIR, exist_ok=True)

# joint limits
JOINT_LIMITS = {
    r".*haa": [-0.2, 0.2],
    r".*f_hfe": [-1.0, 0.6],
    r".*h_hfe": [-1.5, 0.5],
    r".*f_kfe": [-1.5, 0.1],
    r".*h_kfe": [-0.2, 1.0],
    r".*f_pfe": [-0.3, 3.0],
    r".*h_pfe": [-1.2, 2.5],
    r".*f_pastern_to_foot": [-0.3, 1.8],
    r".*h_pastern_to_foot": [-0.3, 1.8],
    r".*base_joint": [-0.2, 0.2],
}


def find_joint_limits(joint_name: str):
    for pattern, limits in JOINT_LIMITS.items():
        if re.match(pattern, joint_name):
            return limits
    return None


data = np.load(LOG_FILE, allow_pickle=True)
steps = data["step"]
target_pos = data["target_pos"]
actual_pos = data["actual_pos"]
torques = data["torque"]
joint_names = data["joint_name"]

num_envs, num_steps, num_joints = actual_pos.shape
print(f"Loaded log for {num_joints} joints, {num_steps} steps.")

pattern = re.compile(r".*(rh|lh|rf|lf)_(haa|hfe|kfe|pfe|pastern_to_foot)$")

joint_map = {}
base_joints = {}

for idx, name in enumerate(joint_names):
    match = pattern.match(name)
    if match:
        leg, joint_type = match.groups()
        leg, joint_type = leg.lower(), joint_type.lower()
        joint_map.setdefault(joint_type, {})[leg] = idx
    else:
        # any joint name not matching the joint pattern will get its own plot
        base_joints[name] = idx


def plot_joint_type(joint_type, leg_to_idx):
    # 4 rows (LF/RF pos+torque, LH/RH pos+torque) × 2 columns
    fig, axs = plt.subplots(4, 2, figsize=(12, 12), constrained_layout=True)
    axs = np.array(axs)

    leg_pos = {
        "lf": (0, 0),
        "rf": (0, 1),
        "lh": (2, 0),
        "rh": (2, 1),
    }

    for leg, (pos_row, col) in leg_pos.items():
        torque_row = pos_row + 1
        if leg not in leg_to_idx:
            axs[pos_row, col].axis("off")
            axs[torque_row, col].axis("off")
            continue

        j = leg_to_idx[leg]

        # find last index with nonzero torque (or position)
        nonzero_idx = np.where(np.abs(actual_pos[ENV_ID, :, j]) > 1e-6)[0]
        last_valid = nonzero_idx[-1] + 1 if len(nonzero_idx) > 0 else len(steps)

        # position plot
        pos_ax = axs[pos_row, col]
        pos_ax.plot(
            steps[:last_valid],
            target_pos[ENV_ID, :last_valid, j],
            linestyle="--",
            linewidth=1.3,
            label="Target Pos",
        )
        pos_ax.plot(
            steps[:last_valid],
            actual_pos[ENV_ID, :last_valid, j],
            linewidth=1.3,
            label="Actual Pos",
        )

        # add joint limit lines
        limits = find_joint_limits(joint_names[j])
        if limits is not None:
            lo, hi = limits
            pos_ax.axhline(
                lo,
                color="blue",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
                label=f"Lower Limit ({lo:.2f})",
            )
            pos_ax.axhline(
                hi,
                color="red",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
                label=f"Upper Limit ({hi:.2f})",
            )

            ymin, ymax = pos_ax.get_ylim()
            ymin = min(ymin, lo - 0.05)
            ymax = max(ymax, hi + 0.05)
            pos_ax.set_ylim([ymin, ymax])

        pos_ax.set_title(f"{leg.upper()} - {joint_type.upper()} Position")
        pos_ax.set_ylabel("Position (rad)")
        pos_ax.grid(True, linestyle="--", alpha=0.4)
        pos_ax.legend(fontsize=7)

        # torque plot
        torque_ax = axs[torque_row, col]
        torque_ax.plot(
            steps[:last_valid],
            torques[ENV_ID, :last_valid, j],
            color="tab:red",
            linewidth=1.2,
        )
        torque_ax.set_title(f"{leg.upper()} - {joint_type.upper()} Torque")
        torque_ax.set_ylabel("Torque (Nm)")
        torque_ax.set_xlabel("Step")
        torque_ax.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle(f"{joint_type.upper()} Joints (All Legs)", fontsize=16)
    save_path = os.path.join(SAVE_DIR, f"{joint_type}.png")
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved {save_path}")


# plot function for base
def plot_base_joints(base_joints):
    fig, axs = plt.subplots(
        len(base_joints) * 2,
        1,
        figsize=(10, 4 * len(base_joints)),
        constrained_layout=True,
    )

    if len(base_joints) == 1:
        axs = np.array([axs[0], axs[1]])

    for i, (name, j) in enumerate(base_joints.items()):
        pos_ax = axs[2 * i]
        torque_ax = axs[2 * i + 1]

        # find last index with nonzero torque (or position)
        nonzero_idx = np.where(np.abs(actual_pos[ENV_ID, :, j]) > 1e-6)[0]
        last_valid = nonzero_idx[-1] + 1 if len(nonzero_idx) > 0 else len(steps)

        # position
        pos_ax.plot(
            steps[:last_valid],
            target_pos[ENV_ID, :last_valid, j],
            linestyle="--",
            linewidth=1.3,
            label="Target Pos",
        )
        pos_ax.plot(
            steps[:last_valid],
            actual_pos[ENV_ID, :last_valid, j],
            linewidth=1.3,
            label="Actual Pos",
        )

        # add joint limits
        limits = find_joint_limits(name)
        if limits is not None:
            lo, hi = limits
            pos_ax.axhline(
                lo,
                color="blue",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
                label=f"Lower Limit ({lo:.2f})",
            )
            pos_ax.axhline(
                hi,
                color="red",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
                label=f"Upper Limit ({hi:.2f})",
            )

            ymin, ymax = pos_ax.get_ylim()
            ymin = min(ymin, lo - 0.05)
            ymax = max(ymax, hi + 0.05)
            pos_ax.set_ylim([ymin, ymax])

        pos_ax.set_title(f"{name} Position")
        pos_ax.set_ylabel("Position (rad)")
        pos_ax.legend(fontsize=7)
        pos_ax.grid(True, linestyle="--", alpha=0.4)

        # torque
        torque_ax.plot(
            steps[:last_valid],
            torques[ENV_ID, :last_valid, j],
            color="tab:red",
            linewidth=1.2,
        )
        torque_ax.set_title(f"{name} Torque")
        torque_ax.set_ylabel("Torque (Nm)")
        torque_ax.set_xlabel("Step")
        torque_ax.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Base Joints", fontsize=16)
    save_path = os.path.join(SAVE_DIR, "base_joints.png")
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved {save_path}")


# generate plots
for joint_type, leg_to_idx in joint_map.items():
    plot_joint_type(joint_type, leg_to_idx)

if base_joints:
    plot_base_joints(base_joints)
