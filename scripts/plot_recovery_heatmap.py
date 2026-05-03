# plot_recovery_heatmap.py

import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = "recovery_sweep"
OUTPUT_PNG = "recovery_success_heatmap.png"

DT = 0.004  # seconds per step

csv_files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {RESULTS_DIR}/")

data = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

summary = (
    data.groupby(["torque_scale", "latency_steps"])["success"].mean().reset_index()
)

summary["success_percent"] = summary["success"] * 100.0
summary["torque_percent"] = summary["torque_scale"] * 100.0
summary["latency_seconds"] = summary["latency_steps"] * DT

pivot = summary.pivot(
    index="latency_seconds",
    columns="torque_percent",
    values="success_percent",
)

pivot = pivot.sort_index(ascending=True)
pivot = pivot.reindex(sorted(pivot.columns, reverse=True), axis=1)

plt.figure(figsize=(9, 6))

im = plt.imshow(
    pivot.values,
    aspect="auto",
    origin="lower",
    vmin=0,
    vmax=100,
)

plt.colorbar(im, label="Success rate (%)")

plt.xticks(
    ticks=np.arange(len(pivot.columns)),
    labels=[f"{int(v)}" for v in pivot.columns],
)

plt.yticks(
    ticks=np.arange(len(pivot.index)),
    labels=[f"{v:.3f}" for v in pivot.index],
)

plt.xlabel("Torque (%)")
plt.ylabel("Latency (seconds)")
plt.title("Recovery Success Rate Heatmap")

plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")

print(f"Saved heatmap to {OUTPUT_PNG}")
