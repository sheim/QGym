# plot_recovery_belay_comparison.py

import glob
import os
import re
import pandas as pd
import matplotlib.pyplot as plt


TORQUE_DIR = "recovery_torque"
LATENCY_DIR = "recovery_latency"

TORQUE_OUTPUT = "success_vs_torque_by_belay.png"
LATENCY_OUTPUT = "success_vs_latency_by_belay.png"


def parse_belay_strength(path):
    """
    Expects folder names like:
    recovery_torque/belay_0/recovery_results_torque_0.8_latency_0.csv
    recovery_torque/belay_20/recovery_results_torque_0.8_latency_0.csv
    """
    parts = os.path.normpath(path).split(os.sep)

    for part in parts:
        match = re.match(r"belay_(.+)", part)
        if match:
            value = match.group(1)
            try:
                return float(value)
            except ValueError:
                return value

    return "unknown"


def load_nested_results(base_dir):
    csv_files = glob.glob(os.path.join(base_dir, "belay_*", "*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {base_dir}/belay_*/")

    dfs = []

    for path in csv_files:
        df = pd.read_csv(path)
        df["source_file"] = os.path.basename(path)
        df["belay_strength"] = parse_belay_strength(path)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def format_belay_label(value):
    if isinstance(value, str):
        return f"belay={value}"

    if float(value).is_integer():
        return f"belay={int(value)}"

    return f"belay={value}"


def plot_success_vs_torque():
    df = load_nested_results(TORQUE_DIR)

    required_cols = {"torque_scale", "success", "belay_strength"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in torque data: {missing}")

    summary = (
        df.groupby(["belay_strength", "torque_scale"])["success"]
        .agg(["mean", "count"])
        .reset_index()
    )

    summary["success_percent"] = summary["mean"] * 100.0

    print("\nTorque summary:")
    print(
        summary[
            ["belay_strength", "torque_scale", "success_percent", "count"]
        ].sort_values(["belay_strength", "torque_scale"])
    )

    plt.figure(figsize=(9, 6))

    for belay_strength, group in summary.groupby("belay_strength"):
        group = group.sort_values("torque_scale", ascending=False)

        plt.plot(
            group["torque_scale"],
            group["success_percent"],
            marker="o",
            label=format_belay_label(belay_strength),
        )

    plt.xlabel("Torque scale")
    plt.ylabel("% of envs that successfully stand")
    plt.title("Recovery Success vs Reduced Torque by Belay Strength")
    plt.ylim(-5, 105)
    plt.grid(True)
    plt.legend(title="Belay strength")

    # Show torque reducing from left to right
    plt.gca().invert_xaxis()

    plt.savefig(TORQUE_OUTPUT, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Saved torque plot to {TORQUE_OUTPUT}")


def plot_success_vs_latency():
    df = load_nested_results(LATENCY_DIR)

    required_cols = {"latency_steps", "success", "belay_strength"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in latency data: {missing}")

    summary = (
        df.groupby(["belay_strength", "latency_steps"])["success"]
        .agg(["mean", "count"])
        .reset_index()
    )

    summary["success_percent"] = summary["mean"] * 100.0

    print("\nLatency summary:")
    print(
        summary[
            ["belay_strength", "latency_steps", "success_percent", "count"]
        ].sort_values(["belay_strength", "latency_steps"])
    )

    plt.figure(figsize=(9, 6))

    for belay_strength, group in summary.groupby("belay_strength"):
        group = group.sort_values("latency_steps")

        plt.plot(
            group["latency_steps"],
            group["success_percent"],
            marker="o",
            label=format_belay_label(belay_strength),
        )

    plt.xlabel("Latency steps")
    plt.ylabel("% of envs that successfully stand")
    plt.title("Recovery Success vs Control Latency by Belay Strength")
    plt.ylim(-5, 105)
    plt.grid(True)
    plt.legend(title="Belay strength")

    plt.savefig(LATENCY_OUTPUT, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Saved latency plot to {LATENCY_OUTPUT}")


def main():
    plot_success_vs_torque()
    plot_success_vs_latency()


if __name__ == "__main__":
    main()
