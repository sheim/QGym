#!/usr/bin/env python3

import numpy as np
import os
import matplotlib.pyplot as plt
import csv

OUT_DIR = "scaling_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

LOG_FILE = "joint_logs.npz"

# load data
d = np.load(LOG_FILE, allow_pickle=True)

actual = d["actual_pos"]  # [envs, steps, joints]
target = d["target_pos"]
joint_names = d["joint_name"]

num_envs, total_steps, num_joints = actual.shape

# episode length
if "step" in d:
    step_arr = d["step"]
    valid = np.where(step_arr != 0)[0]
    actual_steps = valid[-1] + 1 if len(valid) > 0 else 1
else:
    tmp = target[0]
    nz = np.where(np.any(tmp != 0, axis=1))[0]
    actual_steps = nz[-1] + 1

print(f"Trimming timestep dimension from {total_steps} -> {actual_steps}")

# trim arrays
actual = actual[:, :actual_steps, :]
target = target[:, :actual_steps, :]

# raw error
error = target - actual  # [envs, steps, joints]
abs_error = np.abs(error)

# shape: (num_envs * steps, num_joints)
flat_err = abs_error.reshape(-1, num_joints)

# mean absolute error for each joint
mae = np.mean(flat_err, axis=0)

# standard deviation of absolute error for each joint
std_err = np.std(flat_err, axis=0)

# csv stats
csv_path = os.path.join(OUT_DIR, "error_summary.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["joint_idx", "joint_name", "MAE_all_envs", "STD_all_envs"])
    for j in range(num_joints):
        w.writerow(
            [
                j,
                joint_names[j],
                float(mae[j]),
                float(std_err[j]),
            ]
        )

print(f"Saved raw error summary (all envs): {csv_path}")

# histogram with all env
all_errors = error.reshape(-1, num_joints)

for j in range(num_joints):
    name = joint_names[j]
    plt.figure(figsize=(10, 4))
    plt.hist(all_errors[:, j], bins=80)
    plt.title(f"{name} – Raw Error Histogram (All Envs Combined)")
    plt.xlabel("target_pos - actual_pos")
    plt.ylabel("count")
    plt.grid(True, alpha=0.3)

    out = os.path.join(OUT_DIR, f"{j:02d}_{name}_hist.png")
    plt.savefig(out, dpi=200)
    plt.close()

print(f"Saved histograms in: {OUT_DIR}")
