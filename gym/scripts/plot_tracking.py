from gym.envs import __init__  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np
from gym.utils.helpers import get_load_path
import os

# Read the tracking data from the CSV file
play_path = get_load_path(name="HumanoidTrajectoryTracking", load_run="Dec12_14-58-50_")
data = np.genfromtxt(
    os.path.join(os.path.dirname(play_path), "tracking.csv"), delimiter=","
)

# Extract the desired and actual values of x, y, and z
t = data[:, 0]
x_desired = data[:, 1]
y_desired = data[:, 2]
z_desired = data[:, 3]
x_vel_desired = data[:, 4]
y_vel_desired = data[:, 5]
z_vel_desired = data[:, 6]
x_actual = data[:, 7]
y_actual = data[:, 8]
z_actual = data[:, 9]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot the desired values
ax.plot(x_desired, y_desired, z_desired, color="blue", label="Desired")

# Plot the actual values
ax.plot(x_actual, y_actual, z_actual, color="red", label="Actual")

# Set labels and title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Tracking")

# Add a legend
ax.legend()

# Show the plot
plt.show()

# Plot the desired and actual values of z over time
plt.plot(t, z_desired, color="blue", label="Desired")
plt.plot(t, z_actual, color="red", label="Actual")
plt.xlabel("Time")
plt.ylabel("Z")
plt.title("Tracking")
plt.legend()
plt.show()

# Plot the desired velocities over time
plt.plot(t, x_vel_desired, color="blue", label="X")
plt.plot(t, y_vel_desired, color="red", label="Y")
plt.plot(t, z_vel_desired, color="green", label="Z")
plt.xlabel("Time")
plt.ylabel("Velocity")
plt.title("Desired Velocities")
plt.legend()
plt.show()
