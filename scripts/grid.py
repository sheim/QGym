import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np

# Define the grid parameters
x_values = [-2, -1, 0, 1, 2]
y_values = [-2, -1, 0, 1, 2]
num_rows = len(y_values)
num_cols = len(x_values)

# Create a figure and axes with no space between subplots
fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 6))
name = '/3 layers/May23_16-28-33_osc_[16, 16, 8]'
run_type = 'OSC '
file_path = '/home/aileen/QGym/scripts/dof/' + name#'/home/aileen/QGym/scripts/grf/' + name

# Loop through each position in the grid
for i, y in enumerate(y_values):
    for j, x in enumerate(x_values):
        # Load and display the corresponding image
        img_path = f"{x:.1f}_{y:.1f}_dof_pos.png"
        img_path = os.path.join(file_path, img_path)
        # print(img_path)
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            axs[num_rows-1-i, j].imshow(img)
        else:
            img = np.zeros((6, 10, 3))  # Create a black box image
            axs[num_rows-1-i, j].imshow(img)
        axs[num_rows-1-i, j].set_xticks([])
        axs[num_rows-1-i, j].set_yticks([])
        axs[num_rows-1-i, j].set_aspect('equal')  # Set equal aspect ratio

# Set ticks and labels for subplots along the edges
for i in range(num_rows):
    for j in range(num_cols):
        if i == num_rows - 1:
            axs[i, j].set_xlabel(x_values[j])
        else:
            axs[i, j].set_xticks([])
        if j == 0:
            axs[i, j].set_ylabel((y_values[num_rows-1-i]))
        else:
            axs[i, j].set_yticks([])


# Add x and y labels to the entire grid
fig.text(0.5, 0.04, 'Straight', ha='center')
fig.text(0.08, 0.5, 'Yaw', va='center', rotation='vertical')
fig.suptitle(run_type + name.split('_')[-1]+' GRF', x=0.5, y=0.92)
plt.subplots_adjust(wspace=0, hspace=0)
# plt.show()
plt.savefig(file_path + '/total',dpi=300)
