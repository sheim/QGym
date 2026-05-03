# plot_recovery_latency.py

import glob
import os
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = "recovery_latency"  # folder with latency sweep CSVs
OUTPUT_PNG = "success_vs_latency.png"

csv_files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {RESULTS_DIR}/")

dfs = []
for path in csv_files:
    df = pd.read_csv(path)
    df["source_file"] = os.path.basename(path)
    dfs.append(df)

all_data = pd.concat(dfs, ignore_index=True)

required_cols = {"latency_steps", "success"}
missing = required_cols - set(all_data.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# success rate per latency value
summary = (
    all_data.groupby("latency_steps")["success"]
    .agg(["mean", "count"])
    .reset_index()
    .sort_values("latency_steps")
)

summary["success_percent"] = summary["mean"] * 100.0

print(summary[["latency_steps", "success_percent", "count"]])

plt.figure(figsize=(8, 5))
plt.plot(
    summary["latency_steps"],
    summary["success_percent"],
    marker="o",
)

plt.xlabel("Latency steps")
plt.ylabel("% of envs that successfully stand")
plt.title("Recovery Success vs Control Latency")
plt.ylim(-5, 105)
plt.grid(True)

plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")

print(f"Saved plot to {OUTPUT_PNG}")
