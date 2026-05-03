# plot_recovery_torque.py

import glob
import os
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = "recovery_torque"
OUTPUT_PNG = "success_vs_torque.png"


def main():
    csv_files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {RESULTS_DIR}/")

    dfs = []
    for path in csv_files:
        df = pd.read_csv(path)
        df["source_file"] = os.path.basename(path)
        dfs.append(df)

    all_data = pd.concat(dfs, ignore_index=True)

    required_cols = {"torque_scale", "success"}
    missing = required_cols - set(all_data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    summary = (
        all_data.groupby("torque_scale")["success"]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values("torque_scale", ascending=False)
    )

    summary["success_percent"] = summary["mean"] * 100.0

    print(summary[["torque_scale", "success_percent", "count"]])

    plt.figure(figsize=(8, 5))
    plt.plot(
        summary["torque_scale"],
        summary["success_percent"],
        marker="o",
    )

    plt.xlabel("Torque scale")
    plt.ylabel("% of envs that successfully stand")
    plt.title("Recovery Success vs Reduced Torque")
    plt.ylim(-5, 105)
    plt.grid(True)

    # reverse x-axis so impairment increases left-to-right
    plt.gca().invert_xaxis()

    plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")

    print(f"Saved plot to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
