import numpy as np
import random

# Command ranges during training:
# x_range = [-2.0, 3.0]  # [m/s]
# y_range = [-1.0, 1.0]  # [m/s]
# yaw_range = [-3.0. 3.0]  # [rad/s]

# Command ranges for finetuning:
x_range = [-1.0, 1.5]  # [m/s]
y_range = [-0.5, 0.5]  # [m/s]
yaw_range = [-1.5, 1.5]  # [rad/s]

# Generate random command sequence
N = 10
steps = 300
commands = np.zeros((N * steps, 3))

for i in range(1, N):  # first iteration 0 commands
    x = random.uniform(x_range[0], x_range[1])
    y = random.uniform(y_range[0], y_range[1])
    yaw = random.uniform(yaw_range[0], yaw_range[1])
    commands[i * steps : (i + 1) * steps] = np.array([x, y, yaw])

# Export to txt
np.savetxt("commands.txt", commands, fmt="%.3f")
