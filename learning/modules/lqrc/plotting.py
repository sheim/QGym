import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap

import numpy as np


def create_custom_bwr_colormap():
    # Define the colors for each segment
    dark_blue = [0, 0, 0.5, 1]
    light_blue = [0.5, 0.5, 1, 1]
    white = [1, 1, 1, 1]
    light_red = [1, 0.5, 0.5, 1]
    dark_red = [0.5, 0, 0, 1]

    # Number of bins for each segment
    n_bins = 128
    mid_band = 5

    # Create the colormap segments
    blue_segment = np.linspace(dark_blue, light_blue, n_bins // 2)
    white_segment = np.tile(white, (mid_band, 1))
    red_segment = np.linspace(light_red, dark_red, n_bins // 2)

    # Stack segments to create the full colormap
    colors = np.vstack((blue_segment, white_segment, red_segment))
    custom_bwr = ListedColormap(colors, name="custom_bwr")

    return custom_bwr


def plot_pendulum_multiple_critics_w_data(
    x, predictions, targets, title, fn, data, colorbar_label="f(x)", grid_size=64
):
    num_critics = len(x.keys())
    fig, axes = plt.subplots(nrows=2, ncols=num_critics, figsize=(4 * num_critics, 6))

    # Determine global min and max error for consistent scaling
    global_min_error = float("inf")
    global_max_error = float("-inf")
    global_min_prediction = float("inf")
    global_max_prediction = float("-inf")
    prediction_cmap = mpl.cm.get_cmap("viridis")
    error_cmap = create_custom_bwr_colormap()

    for critic_name in x:
        np_predictions = predictions[critic_name].detach().cpu().numpy().reshape(-1)
        np_targets = (
            targets["Ground Truth MC Returns"].detach().cpu().numpy().reshape(-1)
        )
        np_error = np_predictions - np_targets
        global_min_error = min(global_min_error, np.min(np_error))
        global_max_error = max(global_max_error, np.max(np_error))
        global_min_prediction = np.min(np_targets)
        global_max_prediction = np.max(np_targets)
    error_norm = mcolors.TwoSlopeNorm(
        vmin=global_min_error, vcenter=0, vmax=global_max_error
    )
    prediction_norm = mcolors.CenteredNorm(
        vcenter=(global_max_prediction + global_min_prediction) / 2,
        halfrange=(global_max_prediction - global_min_prediction) / 2,
    )

    xcord = np.linspace(-2 * np.pi, 2 * np.pi, grid_size)
    ycord = np.linspace(-5, 5, grid_size)

    for ix, critic_name in enumerate(x):
        np_predictions = predictions[critic_name].detach().cpu().numpy().reshape(-1)
        np_targets = (
            targets["Ground Truth MC Returns"].detach().cpu().numpy().reshape(-1)
        )
        np_error = np_predictions - np_targets

        # Predictions
        axes[0, ix].imshow(
            np_predictions.reshape(grid_size, grid_size).T,
            origin="lower",
            extent=(xcord.min(), xcord.max(), ycord.min(), ycord.max()),
            cmap=prediction_cmap,
            norm=prediction_norm,
        )
        axes[0, ix].set_title(f"{critic_name} Prediction")

        if ix == 0:
            continue

        # Errors
        axes[1, ix].imshow(
            np_error.reshape(grid_size, grid_size).T,
            origin="lower",
            extent=(xcord.min(), xcord.max(), ycord.min(), ycord.max()),
            cmap=error_cmap,
            norm=error_norm,
        )
        axes[1, ix].set_title(f"{critic_name} Error")

    # Trajectories
    data = data.detach().cpu().numpy()
    theta = data[:, :, 0]
    omega = data[:, :, 1]
    axes[1, 0].plot(theta, omega, lw=1)
    axes[1, 0].set_xlabel("theta")
    axes[1, 0].set_ylabel("theta_dot")
    fig.suptitle(title, fontsize=16)

    # Ensure the axes are the same for all plots
    for ax in axes.flat:
        ax.set_xlim([xcord.min(), xcord.max()])
        ax.set_ylim([ycord.min(), ycord.max()])

    plt.subplots_adjust(
        top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.4, wspace=0.3
    )

    fig.colorbar(
        mpl.cm.ScalarMappable(norm=prediction_norm, cmap=prediction_cmap),
        ax=axes[0, :].ravel().tolist(),
        shrink=0.95,
        label=colorbar_label,
    )
    fig.colorbar(
        mpl.cm.ScalarMappable(norm=error_norm, cmap=error_cmap),
        ax=axes[1, :].ravel().tolist(),
        shrink=0.95,
        label=colorbar_label,
    )

    plt.savefig(f"{fn}.png")
    print(f"Saved to {fn}.png")
