#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

LOG_FILE = "height_reward_logs.npz"
OUT_PNG = "height_reward_plot.png"

d = np.load(LOG_FILE)
step = d["step"]
height_rew = d["height_rew"]
h_actual = d["height_actual"]
h_target = d["height_target"]

plt.figure(figsize=(12, 5))

plt.plot(step, h_actual, label="actual height (m)")
plt.plot(step, h_target, label="target height (m)")
plt.plot(step, height_rew, label="_reward_tracking_height")

plt.xlabel("step")
plt.ylabel("value")
plt.title("Height Tracking + Reward (Same Axis)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)

print(f"Saved plot -> {OUT_PNG}")
