import numpy as np
import random

# Command ranges during training:
# x_range = [-2.0, 3.0]  # [m/s]
# y_range = [-1.0, 1.0]  # [m/s]
# yaw_range = [-3.0, 3.0]  # [rad/s]

# Command ranges for finetuning:
x_range = [-0.67, 1.0]  # [m/s]
y_range = [-0.33, 0.33]  # [m/s]
yaw_range = [-2.0, 2.0]  # [rad/s]

# Generate structured command sequence (fixed lin/ang vel, yaw, some random)
N = 100
cmds_zero = np.array([[0, 0, 0]]).repeat(N, axis=0)
cmds = np.zeros((N, 3))
for _ in range(4):
    cmds = np.append(cmds, np.array([[x_range[1], 0, 0]]).repeat(3 * N, axis=0), axis=0)
    cmds = np.append(cmds, cmds_zero, axis=0)
    cmds = np.append(cmds, np.array([[x_range[0], 0, 0]]).repeat(3 * N, axis=0), axis=0)
    cmds = np.append(cmds, cmds_zero, axis=0)
    cmds = np.append(cmds, np.array([[0, y_range[1], 0]]).repeat(2 * N, axis=0), axis=0)
    cmds = np.append(cmds, cmds_zero, axis=0)
    cmds = np.append(cmds, np.array([[0, y_range[0], 0]]).repeat(2 * N, axis=0), axis=0)
    cmds = np.append(cmds, cmds_zero, axis=0)
    cmds = np.append(
        cmds, np.array([[0, 0, yaw_range[1]]]).repeat(2 * N, axis=0), axis=0
    )
    cmds = np.append(cmds, cmds_zero, axis=0)
    cmds = np.append(
        cmds, np.array([[0, 0, yaw_range[0]]]).repeat(2 * N, axis=0), axis=0
    )
    cmds = np.append(cmds, cmds_zero, axis=0)

    for i in range(5):
        x = random.uniform(x_range[0], x_range[1])
        y = random.uniform(y_range[0], y_range[1])
        yaw = random.uniform(yaw_range[0], yaw_range[1])
        cmds = np.append(cmds, np.array([[x, y, yaw]]).repeat(2 * N, axis=0), axis=0)

print(cmds.shape)

# Export to txt
np.savetxt("commands_long.txt", cmds, fmt="%.3f")
