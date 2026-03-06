#!/usr/bin/env python3
# scripts/plot_rewards.py

import numpy as np
import matplotlib.pyplot as plt


def moving_avg(x: np.ndarray, k: int) -> np.ndarray:
    if k is None or k <= 1:
        return x
    k = int(k)
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, np.ones(k, dtype=np.float32) / k, mode="valid")


def main(npz_path="reward_logs.npz", out_path="reward_curves.png", smooth=25):
    d = np.load(npz_path, allow_pickle=True)

    names = [str(x) for x in d["reward_names"].tolist()]
    total = d["total_reward"]  # (steps,)
    t = np.arange(total.shape[0])

    plt.figure(figsize=(12, 6))
    plt.plot(t, moving_avg(total, smooth), label="total_reward", linewidth=2)

    for name in names:
        if name in d:
            plt.plot(t, moving_avg(d[name], smooth), label=name)

    plt.title("Rewards vs Step (env 0 only)")
    plt.xlabel("step")
    plt.ylabel("reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved reward plot to {out_path}")


if __name__ == "__main__":
    main()
