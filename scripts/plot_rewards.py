import numpy as np
import matplotlib.pyplot as plt

data = np.load("reward_logs.npz", allow_pickle=True)

reward_names = data["reward_names"]
total_reward = data["total_reward"]

height_command = data["height_command"] if "height_command" in data else None
switch_height = data["switch_height"] if "switch_height" in data else None

steps = np.arange(len(total_reward))

fig, ax1 = plt.subplots(figsize=(14, 8))

# reward plots
ax1.plot(steps, total_reward, label="total_reward", linewidth=2)

for name in reward_names:
    name = str(name)
    if name in data:
        ax1.plot(steps, data[name], label=name, alpha=0.9)

ax1.set_xlabel("Step")
ax1.set_ylabel("Reward")
ax1.set_title("Rewards, Height Command, and Switch Height")
ax1.grid(True)

# second axis for height signals
ax2 = ax1.twinx()

if height_command is not None:
    ax2.plot(
        steps,
        height_command,
        linestyle="--",
        linewidth=2,
        label="height_command",
    )

if switch_height is not None:
    ax2.plot(
        steps,
        switch_height,
        linestyle=":",
        linewidth=2,
        label="switch_height",
    )

ax2.set_ylabel("Height / Switch")

# combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

plt.tight_layout()

# save figure
plt.savefig("reward_plot.png", dpi=300)

print("Saved reward_plot.png")
