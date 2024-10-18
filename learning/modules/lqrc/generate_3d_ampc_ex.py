import time
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.spatial import KDTree
import itertools

from gym import LEGGED_GYM_ROOT_DIR


def is_in_union(x, y):
    # Check if point is in either circle
    in_circle1 = x**2 + y**2 <= 0.25
    in_circle2 = (x - 1) ** 2 + y**2 <= 0.5
    return in_circle1 or in_circle2


def generate_data(n_samples):
    x0 = []
    cost = []
    while len(x0) < n_samples:
        # Generate random points in the bounding box
        x = np.random.uniform(-1, 4)
        y = np.random.uniform(-1, 1.5)

        if is_in_union(x, y):
            x0.append([x, y])
            cost.append(2.0 * x**2 + 2.0 * y * x + 2.0 * y**2)

    return np.array(x0), np.array(cost)


def plot_data(title, fn):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.scatter(x0[:, 0], x0[:, 1], cost, c=cost, cmap="viridis", s=20)
    ax.scatter(x0[:, 0], x0[:, 1], np.zeros_like(cost), c="grey")
    # axis limits
    ax.set_xlim(-2, 2.5)
    ax.set_ylim(-2, 2.5)
    ax.set_zlim(-1, 1.5)
    # axis labels
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    # view
    ax.view_init(elev=90, azim=-90, roll=0)
    # ax.view_init(elev=30, azim=45, roll=15)
    plt.tight_layout()
    plt.savefig(f"{fn}.png")


# Generate samples
n_samples = 5000  # 1000
x0, cost = generate_data(n_samples)

# remove top 1% of cost values and corresponding states
num_to_remove = int(0.01 * len(cost))
top_indices = np.argsort(cost)[-num_to_remove:]
mask = np.ones(len(cost), dtype=bool)
mask[top_indices] = False
x0 = x0[mask]
cost = cost[mask]

# min max normalization to put state and cost on [0, 1]
x0_min = x0.min(axis=0)
x0_max = x0.max(axis=0)
x0 = (x0 - x0_min) / (x0_max - x0_min)
cost_min = cost.min()
cost_max = cost.max()
cost = (cost - cost_min) / (cost_max - cost_min)

# make batch for one step MPC eval before adding non-fs synthetic data points
batch_terminal_eval = 100
mpc_eval_ix = random.sample(list(range(x0.shape[0])), batch_terminal_eval)
mpc_mask = np.zeros(len(cost), dtype=bool)
mpc_mask[mpc_eval_ix] = True
mpc_eval_x0 = x0[mpc_mask]
mpc_eval_cost = cost[mpc_mask]
# ensure this validation batch is not seen in training data
x0 = x0[~mpc_mask]
cost = cost[~mpc_mask]

# graph data before transform
plot_data("Data Before Transform (But After Normalization)", "before_transform")


# add in non-fs synthetic data points
def max_pairwise_distance(kdtree):
    # Get all points from the KDTree
    points = kdtree.data

    # Find the bounding box
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)

    # Find the corners of the bounding box
    corners = np.array(list(itertools.product(*zip(min_coords, max_coords))))

    # Find the points closest to each corner
    _, corner_points_indices = kdtree.query(corners)

    # Get the actual points closest to the corners
    corner_points = points[corner_points_indices]

    max_distance = 0
    # Calculate pairwise distances between corner points
    for i in range(len(corner_points)):
        for j in range(i + 1, len(corner_points)):
            distance = np.sqrt(np.sum(np.square(corner_points[i] - corner_points[j])))
            max_distance = max(max_distance, distance)

    return max_distance


print("Building KDTree")
# Build the KDTree to get pairwise point distances
start = time.time()
tree = KDTree(x0)
d_max = max_pairwise_distance(tree)
midpt = np.mean(x0, axis=0)
n_synthetic = 0.2 * n_samples

random_x0 = np.concatenate(
    (
        np.random.uniform(
            low=midpt[0] - 2.0 * d_max,
            high=midpt[0] + 2.0 * d_max,
            size=(int(n_synthetic // 2), x0.shape[1]),
        ),
        np.random.uniform(
            low=midpt[1] - 2.0 * d_max,
            high=midpt[1] + 2.0 * d_max,
            size=(int(n_synthetic // 2), x0.shape[1]),
        ),
    )
)


def process_batch(batch):
    indices = tree.query_ball_point(batch, 0.1)
    mask = np.array([len(idx) == 0 for idx in indices])
    return batch[mask]


print("Creating non feasible state data points")
# filter random_x0 in batches
chunk_size = 100
x0_non_fs_list = []
for i in range(0, len(random_x0), chunk_size):
    batch = random_x0[i : i + chunk_size]
    Y_filtered_batch = process_batch(batch)
    if len(Y_filtered_batch) > 0:
        x0_non_fs_list.append(Y_filtered_batch)
print("")

# concatenate all filtered batches and create matching cost array
x0_non_fs = np.concatenate(x0_non_fs_list) if x0_non_fs_list else None
cost_non_fs = np.ones((x0_non_fs.shape[0],))
# union non feasible and feasible states
x0 = np.concatenate((x0, x0_non_fs))
cost = np.concatenate((cost, cost_non_fs))

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(x0[:, 0], x0[:, 1], cost, c=cost, cmap="viridis", s=20)
plt.savefig("test.png")

# graph data after transform
plot_data("Data After Transform", "after_transform")

data_to_save = {"x0": x0, "X": None, "U": None, "J": None, "cost": cost}
with open(f"{LEGGED_GYM_ROOT_DIR}/learning/modules/lqrc/v_dataset.pkl", "wb") as f:
    pickle.dump(data_to_save, f)
