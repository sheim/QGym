import matplotlib.pyplot as plt
import numpy as np

# # Example data
# velocities = np.array([[1, 2], [2, 1], [3, 3], [4, 4], [5, 5]])  # (x, y) velocity commands
# directions = np.array([[0.5, 1], [1, 0.5], [1, 1], [1, 0], [0, 1]])  # (x, y) direction commands
# success_rates = np.array([0.1, 0.5, 0.9, 0.3, 0.7])  # Success rates

# # Extract components for convenience
# start_x = velocities[:, 0]
# start_y = velocities[:, 1]
# dir_x = directions[:, 0]
# dir_y = directions[:, 1]

# # Create the plot
# plt.figure()
# plt.quiver(start_x, start_y, dir_x, dir_y, success_rates, angles='xy', scale_units='xy', scale=1, cmap='viridis')

# # Add color bar
# plt.colorbar(label='Success Rate')

# # Add titles and labels
# plt.title('2D Vector Field with Survival Rates')
# plt.xlabel('Velocity X')
# plt.ylabel('Velocity Y')

# # Set equal scaling
# plt.axis('equal')

# # Show plot
# plt.show()

# start = -2
# end = 2
# step = 0.25

# # Generate the range of values for x and y
# x_values = np.arange(start, end + step, step)
# y_values = np.arange(start, end + step, step)

# # Create the velocity_sweep list with all combinations of x and y
# velocity_sweep = [[x, y] for x in x_values for y in y_values]

# # Print the result
# print(velocity_sweep)



# Import module
import os
 
# Assign directory
directory = r"/home/aileen/QGym/logs/ctrl_freq_sweep"
 
# Iterate over files in directory
for name in os.listdir(directory):
    print(name)