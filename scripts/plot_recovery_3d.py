# plot_recovery_3d_surface.py

import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = "recovery_sweep"
OUTPUT_PNG = "recovery_success_3d_surface.png"

DT = 0.004  # seconds per step

csv_files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {RESULTS_DIR}/")

data = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

summary = (
    data.groupby(["torque_scale", "latency_steps"])["success"].mean().reset_index()
)

summary["success_percent"] = summary["success"] * 100.0

# convert torque to percentage
summary["torque_percent"] = summary["torque_scale"] * 100.0

# convert latency to seconds
summary["latency_seconds"] = summary["latency_steps"] * DT

pivot = summary.pivot(
    index="latency_seconds",
    columns="torque_percent",
    values="success_percent",
)

# preserve exact values (no interpolation)
torque_vals = pivot.columns.values
latency_vals = pivot.index.values

X, Y = np.meshgrid(torque_vals, latency_vals)
Z = pivot.values

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(
    X,
    Y,
    Z,
    cmap="viridis",
    vmin=0,
    vmax=100,
    edgecolor="k",
    linewidth=0.5,
)

ax.set_xlabel("Torque (%)")
ax.set_ylabel("Latency (seconds)")
ax.set_zlabel("Success rate (%)")
ax.set_title("Recovery Success Surface")

ax.set_zlim(0, 100)
ax.invert_xaxis()

ax.set_xticks(torque_vals)
ax.set_xticklabels([f"{int(v)}" for v in torque_vals])

ax.set_yticks(latency_vals)
ax.set_yticklabels([f"{v:.3f}" for v in latency_vals])

cbar = fig.colorbar(surf, ax=ax, pad=0.1)
cbar.set_label("Success rate (%)")

plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")

print(f"Saved plot to {OUTPUT_PNG}")
