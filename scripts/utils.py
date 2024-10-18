import torch
import itertools
import math
import random
import numpy as np
from learning.modules.ampc.wheelbot import WheelbotSimulation


DEVICE = "cuda:0"


def generate_rosenbrock(n, lb, ub, steps):
    """
    Generates data based on Rosenbrock function
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """
    assert n > 1, "n must be > 1 for Rosenbrock"
    all_linspaces = [torch.linspace(lb, ub, steps, device=DEVICE) for i in range(n)]
    X = torch.cartesian_prod(*all_linspaces)
    term_1 = 100 * torch.square(X[:, 1:] - torch.square(X[:, :-1]))
    term_2 = torch.square(1 - X[:, :-1])
    y = torch.sum(term_1 + term_2, axis=1)
    return X, y.unsqueeze(1)


def generate_bounded_rosenbrock(n, lb, ub, steps):
    """
    Generates data based on Rosenbrock function
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """

    def bound_function_smoothly(y):
        a = 50.0  # Threshold
        c = 60.0  # Constant to transition to
        k = 0.1  # Sharpness of transition
        return y * (1 - 1 / (1 + torch.exp(-k * (y - a)))) + c * (
            1 / (1 + torch.exp(-k * (y - a)))
        )

    X, y = generate_rosenbrock(n, lb, ub, steps)
    y = bound_function_smoothly(y)
    return X, y


def generate_rosenbrock_g_data_dict(names, learning_rates):
    graphing_data = {
        lr: {
            data_name: {name: {} for name in names}
            for data_name in [
                "critic_obs",
                "values",
                "returns",
                "error",
            ]
        }
        for lr in learning_rates
    }
    return graphing_data


def find_flatline_index(data, threshold=5):
    # Calculate differences between consecutive elements
    diffs = np.diff(data)

    # Find where differences are zero
    zero_diffs = diffs == 0

    # Use a rolling window to find a sequence of zeros
    rolling_sum = np.convolve(zero_diffs, np.ones(threshold, dtype=int), "valid")

    # Find the first occurrence where we have 'threshold' consecutive zeros
    flatline_start = np.argmax(rolling_sum == threshold)

    # If we found a flatline, return the index where it starts
    if flatline_start > 0 or rolling_sum[-1] == threshold:
        return flatline_start
    else:
        return -1  # No flatline found


def is_lipschitz_continuous(X, y, threshold=1e6):
    sample_ix = random.sample(list(range(X.shape[0])), 1000)
    mask = np.zeros(len(y), dtype=bool)
    mask[sample_ix] = True
    X = X[mask]
    y = y[mask]
    n_samples = X.shape[0]
    max_ratio = 0

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            x_diff = np.linalg.norm(X[i] - X[j])
            y_diff = np.abs(y[i] - y[j])

            if x_diff != 0:
                ratio = y_diff / x_diff
                max_ratio = max(max_ratio, ratio)

    return max_ratio < threshold, max_ratio


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


def calc_neighborhood_radius(tree, x0):
    subset_ix = random.sample(list(range(x0.shape[0])), 1000)
    subset_mask = np.zeros(x0.shape[0], dtype=bool)
    subset_mask[subset_ix] = True
    x0_subset = x0[subset_mask]
    distances, _ = tree.query(x0_subset, k=2)
    nn_dist = distances[:, 1]
    return np.mean(nn_dist)


def process_batch(batch, tree, radius=0.1):
    indices = tree.query_ball_point(batch, radius)
    mask = np.array([len(idx) == 0 for idx in indices])
    return batch[mask]


def in_box(x, x_min, x_max):
    return np.all(np.logical_and(x >= x_min, x <= x_max))


def grid_search_u(nn, X, model, u_lb, u_ub, step):
    """
    for each torque pair in grid, simulate one step of using that torque, compute value
    select torque pair of highest value
    """
    grid_size = len(np.arange(u_lb[0], u_ub[0], step))
    state_dim = X.shape[0]
    vf = np.zeros((grid_size, grid_size, 3))
    x_results = np.zeros((grid_size, grid_size, state_dim))
    u_inf = []
    for i, u_1 in enumerate(np.arange(u_lb[0], u_ub[0], step)):
        for j, u_2 in enumerate(np.arange(u_lb[1], u_ub[1], step)):
            x_res = model.run(X, np.array([u_1, u_2]))
            with torch.no_grad():
                vf[i, j, 0] = u_1
                vf[i, j, 1] = u_2
                nn_val = (
                    nn.evaluate(torch.from_numpy(x_res).float().to(DEVICE).unsqueeze(0))
                    .cpu()
                    .detach()
                    .numpy()
                )
                nn_val = (
                    nn_val.item() if not math.isnan(nn_val.item()) else float("inf")
                )
                vf[i, j, 2] = nn_val
                if math.isinf(nn_val):
                    print(
                        f"at i {i} and j {j} and state {x_res} we got nan value from NN"
                    )
                    u_inf.append([u_1, u_2])
                x_results[i, j, :] = x_res
    ix = np.unravel_index(np.argmin(vf[:, :, -1]), vf[:, :, -1].shape)
    return vf[ix][:2], x_results[ix]
