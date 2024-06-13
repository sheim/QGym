from math import sqrt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import CenteredNorm, ListedColormap
import matplotlib as mpl
import matplotlib.colors as mcolors

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

font = {"size": 12}
matplotlib.rc("font", **font)


def generate_statistics_str(x, add_on=None):
    mu = x.mean()
    median = np.median(x)
    sigma = x.std()
    if add_on is not None:
        textstr = "\n".join(
            (
                add_on,
                r"$\mu=%.2f$" % (mu,),
                r"$\mathrm{median}=%.2f$" % (median,),
                r"$\sigma=%.2f$" % (sigma,),
            )
        )
    else:
        textstr = "\n".join(
            (
                r"$\mu=%.2f$" % (mu,),
                r"$\mathrm{median}=%.2f$" % (median,),
                r"$\sigma=%.2f$" % (sigma,),
            )
        )
    return textstr


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


def create_custom_pink_green_colormap():
    # Define the colors for each segment
    dark_pink = [0.5, 0, 0.25, 1]
    light_pink = [1, 0.5, 0.75, 1]
    white = [1, 1, 1, 1]
    light_green = [0.5, 1, 0.5, 1]
    dark_green = [0, 0.5, 0, 1]

    # Number of bins for each segment
    n_bins = 128
    mid_band = 5

    # Create the colormap segments
    pink_segment = np.linspace(dark_pink, light_pink, n_bins // 2)
    white_segment = np.tile(white, (mid_band, 1))
    green_segment = np.linspace(light_green, dark_green, n_bins // 2)

    # Stack segments to create the full colormap
    colors = np.vstack((pink_segment, white_segment, green_segment))
    custom_pink_green = ListedColormap(colors, name="custom_pink_green")

    return custom_pink_green


def plot_binned_errors(
    data,
    fn,
    lb=0,
    ub=500,
    step=20,
    tick_step=5,
    title_add_on="",
    extension="pdf",
    include_text=True,
):
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    num_cols = len(list(list(list(data.values())[0].values())[0].keys()))
    print("num cols", num_cols)
    fig, axes = plt.subplots(
        nrows=1, ncols=num_cols, figsize=(25, 6), layout="constrained"
    )
    fig.suptitle(f"Pointwise Prediction Error for {title_add_on} \n", fontsize=27)
    colors = dict(
        zip(
            [lr for lr in data.keys()],
            ["red", "green", "blue", "purple", "orange"][:num_cols],
        )
    )
    for lr in data.keys():
        # bins = np.linspace(0.0, 1.5, 50)
        # bins = np.linspace(0.0, 20, 40)
        bins = np.arange(lb, ub, step)
        # bins = np.linspace(lb, ub, (ub-lb)//step)
        bin_labels = [
            "<" + str(np.round(bins[ix], decimals=1))
            for ix in range(0, len(bins), tick_step)
        ]
        y_min = float("inf")
        y_max = -float("inf")

        for ix, critic in enumerate(data[lr]["critic_obs"].keys()):
            critic_data = data[lr]["error"][critic].squeeze().detach().cpu().numpy()
            digitized = np.digitize(critic_data, bins)
            bincount = np.bincount(digitized)

            y_min = bincount.min() if bincount.min() < y_min else y_min
            y_max = bincount.max() + 10 if bincount.max() + 10 > y_max else y_max

            # to avoid label repetitions in legend
            if ix == 0:
                axes[ix].bar(
                    np.arange(len(bincount)),
                    bincount,
                    color=colors[lr],
                    alpha=0.5,
                    label=str(lr),
                )
            else:
                axes[ix].bar(
                    np.arange(len(bincount)), bincount, color=colors[lr], alpha=0.5
                )
            axes[ix].set_title(critic, fontsize=22)
            x_ticks = np.arange(0, len(bins), tick_step)
            axes[ix].set_xticks(x_ticks, labels=bin_labels)
            axes[ix].set_xlim(
                -min((tick_step / 2.0), 2.0),
                min(len(bins) + (tick_step / 2.0), len(bins) + 2.0),
            )
            if lr > 1e-4 and include_text:
                predictions = (
                    data[lr]["values"][critic].squeeze().detach().cpu().numpy()
                )
                axes[ix].text(
                    0.4,
                    0.95,
                    generate_statistics_str(predictions, r"0.001 Learning Rate"),
                    transform=axes[ix].transAxes,
                    fontsize=22,
                    verticalalignment="top",
                    bbox=props,
                )
        for ix, critic in enumerate(data[lr].keys()):
            axes[ix].set_ylim(y_min, y_max)
    fig.legend(loc=" outside upper right", fontsize=18)
    plt.savefig(fn + f".{extension}", bbox_inches="tight", dpi=300)
    print(f"Saved to {fn}.{extension}")


def plot_dim_sweep(x, y, mean_error, max_error, fn, step=5):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
    np_x = x  # .detach().cpu().numpy()
    np_y = y  # .detach().cpu().numpy()
    np_mean_error = mean_error  # .detach().cpu().numpy()
    np_max_error = max_error  # .detach().cpu().numpy()

    extent = [np_x.min() - 0.5, np_x.max() + 0.5, np_y.min() - 0.5, np_y.max() + 0.5]

    # mean_err = axes[0].pcolormesh(np_x, np_y, np_mean_error, cmap="YlOrRd", vmin=0.0)
    mean_err = axes[0].imshow(
        np_mean_error,
        cmap="YlOrRd",
        interpolation="nearest",
        vmin=0,
        origin="lower",
        extent=extent,
    )
    # max_err = axes[1].pcolormesh(np_x, np_y, np_max_error, cmap="PuBuGn", vmin=0.0)
    max_err = axes[1].imshow(
        np_max_error,
        cmap="PuBuGn",
        interpolation="nearest",
        vmin=0,
        origin="lower",
        extent=extent,
    )

    x_ticks = np.arange(np_x.min(), np_x.max(), step)
    y_ticks = np.arange(np_y.min(), np_y.max(), step)
    axes[0].set_xticks(x_ticks)
    axes[0].set_yticks(y_ticks)
    axes[0].set_title("Mean Error")
    axes[0].set_xlabel("Rank(A)")
    axes[0].set_ylabel("Dim(A)")
    # axes[0].set_xlim(0.5, None)
    # axes[0].set_ylim(0.5, None)

    axes[1].set_xticks(x_ticks)
    axes[1].set_yticks(y_ticks)
    axes[1].set_title("Max Error")
    axes[1].set_xlabel("Rank(A)")
    axes[1].set_ylabel("Dim(A)")
    # axes[1].set_xlim(0.5, None)
    # axes[1].set_ylim(0.5, None)

    fig.colorbar(mean_err, ax=axes[0], shrink=0.95, pad=0.1, label="Mean Error")
    fig.colorbar(max_err, ax=axes[1], shrink=0.95, pad=0.1, label="Max Error")
    plt.savefig(fn, bbox_inches="tight", dpi=300)
    print(f"Saved to {fn}")


def plot_dim_sweep_mean_std(
    x,
    y,
    avg_mean_error,
    avg_max_error,
    std_mean_error,
    std_max_error,
    fn,
    trial_num,
    title,
    step=5,
    extension="pdf",
):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(22, 15))

    extent = [x.min() - 0.5, x.max() + 0.5, y.min() - 0.5, y.max() + 0.5]

    avg_mean_err = axes[0, 0].imshow(
        avg_mean_error,
        cmap="YlOrRd",
        interpolation="nearest",
        norm=mcolors.LogNorm(),
        origin="lower",
        extent=extent,
    )
    avg_max_err = axes[0, 1].imshow(
        avg_max_error,
        cmap="PuBuGn",
        interpolation="nearest",
        norm=mcolors.LogNorm(),
        origin="lower",
        extent=extent,
    )
    std_mean_err = axes[1, 0].imshow(
        std_mean_error,
        cmap="YlOrRd",
        interpolation="nearest",
        norm=mcolors.LogNorm(),
        origin="lower",
        extent=extent,
    )
    std_max_err = axes[1, 1].imshow(
        std_max_error,
        cmap="PuBuGn",
        interpolation="nearest",
        norm=mcolors.LogNorm(),
        origin="lower",
        extent=extent,
    )

    x_ticks = np.arange(0, x.max(), step)
    y_ticks = np.arange(0, y.max(), step)
    for ax in axes.ravel().tolist():
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xlabel("Rank(A)")
        ax.set_ylabel("Dim(A)")
    axes[0, 0].set_title(f"Mean Error (Average Over {trial_num} Trials)")
    axes[0, 1].set_title(f"Max Error (Average Over {trial_num} Trials)")
    axes[1, 0].set_title(f"Mean Error (Standard Deviation Over {trial_num} Trials)")
    axes[1, 1].set_title(f"Max Error (Standard Deviation Over {trial_num} Trials)")

    fig.colorbar(
        avg_mean_err,
        ax=axes[0, 0],
        shrink=0.95,
        pad=0.1,
        label=f"Mean Error (Average Over {trial_num} Trials)",
    )
    fig.colorbar(
        avg_max_err,
        ax=axes[0, 1],
        shrink=0.95,
        pad=0.1,
        label=f"Max Error (Average Over {trial_num} Trials)",
    )
    fig.colorbar(
        std_mean_err,
        ax=axes[1, 0],
        shrink=0.95,
        pad=0.1,
        label=f"Mean Error (Standard Deviation Over {trial_num} Trials)",
    )
    fig.colorbar(
        std_max_err,
        ax=axes[1, 1],
        shrink=0.95,
        pad=0.1,
        label=f"Max Error (Standard Deviation Over {trial_num} Trials)",
    )
    fig.suptitle(title, fontsize=20)
    plt.savefig(fn + f".{extension}", bbox_inches="tight", dpi=300)
    print(f"Saved to {fn}.{extension}")


def plot_pendulum_multiple_critics(
    x, predictions, targets, title, fn, colorbar_label="f(x)"
):
    num_critics = len(x.keys())
    fig, axes = plt.subplots(nrows=2, ncols=num_critics, figsize=(6 * num_critics, 10))

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
        global_min_prediction = min(global_min_prediction, np.min(np_predictions))
        global_max_prediction = max(global_max_prediction, np.max(np_predictions))
    error_norm = mcolors.TwoSlopeNorm(
        vmin=global_min_error, vcenter=0, vmax=global_max_error
    )
    # error_norm = mcolors.TwoSlopeNorm(vmin=-2.50, vcenter=0, vmax=2.50)
    # prediction_norm = mcolors.BoundaryNorm(
    #     [global_min_prediction, global_max_prediction], 256
    # )
    prediction_norm = mcolors.CenteredNorm(
        vcenter=(global_max_prediction + global_min_prediction) / 2,
        halfrange=(global_max_prediction - global_min_prediction) / 2,
    )
    # prediction_norm = mcolors.CenteredNorm(
    #     vcenter=0.0,
    #     halfrange=2.5,
    # )
    for ix, critic_name in enumerate(x):
        np_x = x[critic_name].detach().cpu().numpy().reshape(-1, 2)
        np_predictions = predictions[critic_name].detach().cpu().numpy().reshape(-1)
        np_targets = (
            targets["Ground Truth MC Returns"].detach().cpu().numpy().reshape(-1)
        )
        np_error = np_predictions - np_targets

        # predictions
        # if np.any(~np.equal(np_predictions, np.zeros_like(np_predictions))):
        axes[0, ix].scatter(
            np_x[:, 0],
            np_x[:, 1],
            c=np_predictions,
            cmap=prediction_cmap,
            norm=prediction_norm,
            alpha=0.5,
            # norm=CenteredNorm(),
        )
        axes[0, ix].set_title(f"{critic_name} Prediction")

        # error
        # if np.any(~np.equal(np_error, np.zeros_like(np_error))):
        axes[1, ix].scatter(
            np_x[:, 0],
            np_x[:, 1],
            c=np_error,
            cmap=error_cmap,
            norm=error_norm,
            alpha=0.5,
        )
        axes[1, ix].set_title(f"{critic_name} Error")

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
    fig.suptitle(title, fontsize=16)
    plt.savefig(f"{fn}.png")
    print(f"Saved to {fn}.png")


def plot_pendulum_multiple_critics_w_data(
    x, predictions, targets, title, fn, data, colorbar_label="f(x)", grid_size=64
):
    num_critics = len(x.keys())
    fig, axes = plt.subplots(nrows=2, ncols=num_critics, figsize=(6 * num_critics, 10))
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
    # prediction_norm = mcolors.CenteredNorm(
    #     vcenter=0.0,
    #     halfrange=2.5,
    # )
    xcord = np.linspace(-2 * np.pi, 2 * np.pi, grid_size)
    ycord = np.linspace(-5, 5, grid_size)

    for ix, critic_name in enumerate(x):
        np_x = x[critic_name].detach().cpu().numpy().reshape(-1, 2)
        # rescale (hack), value shardcoded from pendulum_config
        np_x[:, 0] = np_x[:, 0] * 2 * np.pi
        np_x[:, 1] = np_x[:, 1] * 5
        np_predictions = predictions[critic_name].detach().cpu().numpy().reshape(-1)
        np_targets = (
            targets["Ground Truth MC Returns"].detach().cpu().numpy().reshape(-1)
        )
        np_error = np_predictions - np_targets

        # todo: clean up this super hacky code

        # predictions
        # if np.any(~np.equal(np_predictions, np.zeros_like(np_predictions))):
        # axes[0, ix].scatter(
        #     np_x[:, 0] * 2 * np.pi,
        #     np_x[:, 1] * 5,
        #     c=np_predictions,
        #     cmap=prediction_cmap,
        #     norm=prediction_norm,
        #     alpha=0.5,
        #     # norm=CenteredNorm(),
        # )
        axes[0, ix].imshow(
            np_predictions.reshape(grid_size, grid_size).T,
            origin="lower",
            extent=(xcord.min(), xcord.max(), ycord.min(), ycord.max()),
            cmap=prediction_cmap,
            norm=prediction_norm,
            # shading="auto",
        )
        axes[0, ix].set_title(f"{critic_name} Prediction")

        if ix == 0:
            continue
        # axes[1, ix].scatter(
        #     np_x[:, 0],
        #     np_x[:, 1],
        #     c=np_error,
        #     cmap=error_cmap,
        #     norm=error_norm,
        #     alpha=0.5,
        # )
        axes[1, ix].imshow(
            np_error.reshape(grid_size, grid_size).T,
            origin="lower",
            extent=(xcord.min(), xcord.max(), ycord.min(), ycord.max()),
            cmap=error_cmap,
            norm=error_norm,
            # shading="auto",
        )
        axes[1, ix].set_title(f"{critic_name} Error")

    data = data.detach().cpu().numpy()
    theta = data[:, :, 0] * 2 * np.pi
    omega = data[:, :, 1] * 5
    # _, _, _, hist = axes[1, 0].hist2d(
    #     theta.flatten(), omega.flatten(), bins=64, cmap="Blues"
    # )
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


def plot_rosenbrock_multiple_critics_w_data(
    x,
    predictions,
    targets,
    title,
    fn,
    data,
    grid_size=64,
    extension="pdf",
    data_title="Test Data Distribution",
):
    num_critics = len(x.keys())
    fig, axes = plt.subplots(
        nrows=2,
        ncols=num_critics,
        figsize=(5 * num_critics, 10.5),
        layout="constrained",
    )
    # Determine global min and max error for consistent scaling
    global_min_error = float("inf")
    global_max_error = float("-inf")
    global_min_prediction = float("inf")
    global_max_prediction = float("-inf")
    prediction_cmap = mpl.cm.get_cmap("viridis")
    error_cmap = create_custom_bwr_colormap()

    data = data.detach().cpu().numpy()
    x_coord = data[:, 0]
    y_coord = data[:, 1]

    for critic_name in x:
        np_predictions = predictions[critic_name].detach().cpu().numpy().reshape(-1)
        np_targets = targets["Rosenbrock"].detach().cpu().numpy().reshape(-1)
        np_error = np_predictions - np_targets
        global_min_error = min(global_min_error, np.min(np_error))
        global_max_error = max(global_max_error, np.max(np_error))
        global_min_prediction = min(global_min_prediction, np.min(np_predictions))
        global_max_prediction = max(global_max_prediction, np.max(np_predictions))
    error_norm = mcolors.TwoSlopeNorm(
        vmin=global_min_error, vcenter=0, vmax=global_max_error
    )
    prediction_norm = mcolors.CenteredNorm(
        vcenter=(global_max_prediction + global_min_prediction) / 2,
        halfrange=(global_max_prediction - global_min_prediction) / 2,
    )
    # error_norm = mcolors.LogNorm()
    # prediction_norm = mcolors.LogNorm(
    #     vmin=global_min_prediction, vmax=global_max_prediction
    # )

    for ix, critic_name in enumerate(x):
        np_x = x[critic_name].detach().cpu().numpy().reshape(-1, 2)
        # rescale (hack), value shardcoded from pendulum_config
        np_x[:, 0] = np_x[:, 0]
        np_x[:, 1] = np_x[:, 1]
        np_predictions = predictions[critic_name].detach().cpu().numpy().reshape(-1)
        np_targets = targets["Rosenbrock"].detach().cpu().numpy().reshape(-1)
        np_error = np_predictions - np_targets

        axes[0, ix].imshow(
            np_predictions.reshape(grid_size, grid_size).T,
            origin="lower",
            extent=(x_coord.min(), x_coord.max(), y_coord.min(), y_coord.max()),
            cmap=prediction_cmap,
            norm=prediction_norm,
            # shading="auto",
        )
        ax_title = (
            f"{critic_name} Ground Truth"
            if "Rosenbrock" in critic_name
            else f"{critic_name} Prediction"
        )
        axes[0, ix].set_title(ax_title)

        if ix == 0:
            continue

        axes[1, ix].imshow(
            np_error.reshape(grid_size, grid_size).T,
            origin="lower",
            extent=(x_coord.min(), x_coord.max(), y_coord.min(), y_coord.max()),
            cmap=error_cmap,
            norm=error_norm,
            # shading="auto",
        )
        axes[1, ix].set_title(f"{critic_name} Error")

    # axes[1, 0].plot(x_coord, y_coord, lw=1)
    axes[1, 0].scatter(x_coord, y_coord, alpha=0.75, s=3)
    axes[1, 0].set_title(data_title)
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("y")

    # Ensure the axes are the same for all plots
    for ax in axes.flat:
        ax.set_xlim([x_coord.min(), x_coord.max()])
        ax.set_ylim([y_coord.min(), y_coord.max()])

    asp = np.diff(axes[1, 0].get_xlim())[0] / np.diff(axes[1, 0].get_ylim())[0]
    axes[1, 0].set_aspect(asp)

    fig.colorbar(
        mpl.cm.ScalarMappable(norm=prediction_norm, cmap=prediction_cmap),
        ax=axes[0, :].ravel().tolist(),
        shrink=0.95,
        # label="Pointwise Prediction",
    )
    fig.colorbar(
        mpl.cm.ScalarMappable(norm=error_norm, cmap=error_cmap),
        ax=axes[1, :].ravel().tolist(),
        shrink=0.95,
        # label="Pointwise Error",
    )
    fig.suptitle(title)
    plt.savefig(f"{fn}.{extension}", bbox_inches="tight", dpi=300)
    print(f"Saved to {fn}.{extension}")


def plot_pendulum_multiple_critics_w_data_grad(
    x,
    predictions,
    targets,
    title,
    fn,
    data,
    pred_grad,
    analytic_grad,
    colorbar_label="f(x)",
):
    num_critics = len(x.keys())
    fig, axes = plt.subplots(nrows=3, ncols=num_critics, figsize=(6 * num_critics, 10))

    # Determine global min and max error for consistent scaling
    global_min_error = float("inf")
    global_max_error = float("-inf")
    global_min_prediction = float("inf")
    global_max_prediction = float("-inf")
    prediction_cmap = mpl.cm.get_cmap("viridis")
    error_cmap = create_custom_bwr_colormap()
    grad_error_cmap = create_custom_pink_green_colormap()

    for critic_name in x:
        np_predictions = predictions[critic_name].detach().cpu().numpy().reshape(-1)
        np_targets = (
            targets["Ground Truth MC Returns"].detach().cpu().numpy().reshape(-1)
        )
        np_error = np_predictions - np_targets
        np_pred_grad = pred_grad[critic_name].detach().cpu().numpy()
        np_analytic_grad = analytic_grad[critic_name].detach().cpu().numpy()
        np_grad_error = np_pred_grad - np_analytic_grad
        global_min_error = min(global_min_error, np.min(np_error))
        global_max_error = max(global_max_error, np.max(np_error))
        global_min_prediction = min(global_min_prediction, np.min(np_targets))
        global_max_prediction = max(global_max_prediction, np.max(np_targets))
        global_min_grad_error = min(global_min_grad_error, np.min(np_grad_error))  # noqa
        global_max_grad_error = max(global_max_grad_error, np.max(np_grad_error))  # noqa
    error_norm = mcolors.TwoSlopeNorm(
        vmin=global_min_error, vcenter=0, vmax=global_max_error
    )
    prediction_norm = mcolors.CenteredNorm(
        vcenter=(global_max_prediction + global_min_prediction) / 2,
        halfrange=(global_max_prediction - global_min_prediction) / 2,
    )
    grad_error_norm = mcolors.TwoSlopeNorm(
        vmin=global_min_grad_error, vcenter=0, vmax=global_max_error
    )

    for ix, critic_name in enumerate(x):
        np_x = x[critic_name].detach().cpu().numpy().reshape(-1, 2)
        np_predictions = predictions[critic_name].detach().cpu().numpy().reshape(-1)
        np_targets = (
            targets["Ground Truth MC Returns"].detach().cpu().numpy().reshape(-1)
        )
        np_error = np_predictions - np_targets
        np_pred_grad = pred_grad[critic_name].detach().cpu().numpy()
        np_analytic_grad = analytic_grad[critic_name].detach().cpu().numpy()
        np_grad_error = np_pred_grad - np_analytic_grad

        # predictions
        axes[0, ix].scatter(
            np_x[:, 0],
            np_x[:, 1],
            c=np_predictions,
            cmap=prediction_cmap,
            norm=prediction_norm,
            alpha=0.5,
        )
        axes[0, ix].set_title(f"{critic_name} Prediction")

        # error
        if ix == 0:
            continue
        axes[1, ix].scatter(
            np_x[:, 0],
            np_x[:, 1],
            c=np_error,
            cmap=error_cmap,
            norm=error_norm,
            alpha=0.5,
        )
        axes[1, ix].set_title(f"{critic_name} Error")

        # pred error
        axes[2, ix].scatter(
            np_x[:, 0],
            np_x[:, 1],
            c=np_grad_error,
            cmap=grad_error_cmap,
            norm=grad_error_norm,
            alpha=0.5,
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
    fig.colorbar(
        mpl.cm.ScalarMappable(norm=grad_error_norm, cmap=grad_error_cmap),
        ax=axes[2, :].ravel().tolist(),
        shrink=0.95,
        label=colorbar_label,
    )

    data = data.detach().cpu().numpy()
    theta = data[:, :, 0]
    omega = data[:, :, 1]
    _, _, _, hist = axes[1, 0].hist2d(
        theta.flatten(), omega.flatten(), bins=64, cmap="Blues"
    )
    axes[1, 0].plot(theta, omega, lw=1)

    axes[1, 0].set_xlabel("theta")
    axes[1, 0].set_ylabel("theta_dot")
    fig.suptitle(title, fontsize=16)
    plt.savefig(f"{fn}.png")
    print(f"Saved to {fn}.png")


def plot_pendulum_single_critic(
    x, predictions, targets, title, fn, colorbar_label="f(x)"
):
    assert False, "Single pendulum plotter is deprecated"
    x = x.detach().cpu().numpy().reshape(-1, 2)
    predictions = predictions.detach().cpu().numpy().reshape(-1)
    targets = targets.detach().cpu().numpy().reshape(-1)
    error = predictions - targets

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # prediction values
    pred_scatter = axes[1].scatter(
        x[:, 0], x[:, 1], c=predictions, cmap="PiYG", alpha=0.5, norm=CenteredNorm()
    )
    # error
    error_scatter = axes[0].scatter(
        x[:, 0], x[:, 1], c=error, cmap="bwr", alpha=0.5, norm=CenteredNorm()
    )

    axes[0].set_title("Error")
    axes[1].set_title("Predictions")
    fig.colorbar(pred_scatter, ax=axes[1], shrink=0.95, label=colorbar_label)
    fig.colorbar(error_scatter, ax=axes[0], shrink=0.95, label=colorbar_label)
    fig.suptitle(title, fontsize=16)
    plt.savefig(f"{fn}.png")
    print(f"Saved to {fn}.png")


def plot_state_data_dist(data, fn):
    data = data.detach().cpu().numpy()
    theta = data[:, :, 0]
    omega = data[:, :, 1]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    _, _, _, hist = axes[0].hist2d(
        theta.flatten(), omega.flatten(), bins=64, cmap="Blues"
    )
    axes[1].plot(theta, omega, lw=1)

    axes[0].set_xlabel("theta")
    axes[0].set_ylabel("theta_dot")
    fig.suptitle("State Space Data Density")
    fig.colorbar(hist, label="Data Density")
    plt.savefig(f"{fn}.png")
    print(f"Saved to {fn}.png")


def graph_3D_helper(ax, contour=False):
    if contour:
        return ax.contourf
    return ax.pcolormesh


def set_titles_labels(axes, titles=[""], xy_labels=["theta (rad)", "omega (rad/s)"]):
    for ix, ax in enumerate(axes):
        ax.set_xlabel(xy_labels[0])
        ax.set_ylabel(xy_labels[1])
        ax.set_title(titles[ix])


def plot_custom_critic(x_actual, y_pred, xT_A_x, c_pred, fn, contour):
    x_actual = x_actual.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    xT_A_x = xT_A_x.detach().cpu().numpy()
    c_pred = c_pred.detach().cpu().numpy()

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))
    sq_len = int(sqrt(x_actual.shape[0]))
    img = graph_3D_helper(axes[0], contour)(
        x_actual[:, 0].reshape(sq_len, sq_len),
        x_actual[:, 1].reshape(sq_len, sq_len),
        xT_A_x.reshape(sq_len, sq_len),
        cmap="RdBu_r",
        norm=CenteredNorm(),
    )
    img = graph_3D_helper(axes[1], contour)(
        x_actual[:, 0].reshape(sq_len, sq_len),
        x_actual[:, 1].reshape(sq_len, sq_len),
        c_pred.reshape(sq_len, sq_len),
        cmap="RdBu_r",
        norm=CenteredNorm(),
    )
    img = graph_3D_helper(axes[2], contour)(
        x_actual[:, 0].reshape(sq_len, sq_len),
        x_actual[:, 1].reshape(sq_len, sq_len),
        y_pred.reshape(sq_len, sq_len),
        cmap="RdBu_r",
        norm=CenteredNorm(),
    )
    fig.colorbar(img, ax=axes.ravel().tolist(), shrink=0.95, pad=0.1)
    set_titles_labels(axes, ["Predicted x.T @ A @ x", "Predicted c", "Predicted y"])
    plt.savefig(fn, bbox_inches="tight", dpi=300)
    print(f"Saved to {fn}")


def plot_critic_prediction_only(x_actual, y_pred, fn, contour):
    x_actual = x_actual.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    sq_len = int(sqrt(x_actual.shape[0]))
    img = graph_3D_helper(ax, contour)(
        x_actual[:, 0].reshape(sq_len, sq_len),
        x_actual[:, 1].reshape(sq_len, sq_len),
        y_pred.reshape(sq_len, sq_len),
        cmap="RdBu_r",
        norm=CenteredNorm(),
    )
    fig.colorbar(img, ax=ax, shrink=0.95, pad=0.1)
    set_titles_labels([ax], ["Predicted y"])
    plt.savefig(fn, bbox_inches="tight", dpi=300)
    print(f"Saved to {fn}")


def plot_value_func_error(
    x_actual, custom_error, standard_error, ground_truth, fn, contour
):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(25, 6))
    sq_len = int(sqrt(x_actual.shape[0]))
    img = graph_3D_helper(axes[0], contour)(
        x_actual[:, 0].reshape(sq_len, sq_len),
        x_actual[:, 1].reshape(sq_len, sq_len),
        custom_error.reshape(sq_len, sq_len),
        cmap="RdBu_r",
        norm=CenteredNorm(),
    )
    img = graph_3D_helper(axes[1], contour)(
        x_actual[:, 0].reshape(sq_len, sq_len),
        x_actual[:, 1].reshape(sq_len, sq_len),
        standard_error.reshape(sq_len, sq_len),
        cmap="RdBu_r",
        norm=CenteredNorm(),
    )
    fig.colorbar(img, ax=axes.ravel().tolist(), shrink=0.95, pad=0.1)
    set_titles_labels(axes, ["Custom Critic Error", "Standard Critic Error"])
    plt.savefig(fn, bbox_inches="tight", dpi=300)
    print(f"Saved to {fn}")


def plot_value_func(x_actual, custom, standard, ground_truth, fn, contour):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25, 6))
    sq_len = int(sqrt(x_actual.shape[0]))
    img = graph_3D_helper(axes[0], contour)(
        x_actual[:, 0].reshape(sq_len, sq_len),
        x_actual[:, 1].reshape(sq_len, sq_len),
        custom.reshape(sq_len, sq_len),
        cmap="PiYG",
        norm=CenteredNorm(),
    )
    img = graph_3D_helper(axes[1], contour)(
        x_actual[:, 0].reshape(sq_len, sq_len),
        x_actual[:, 1].reshape(sq_len, sq_len),
        standard.reshape(sq_len, sq_len),
        cmap="PiYG",
        norm=CenteredNorm(),
    )
    img = graph_3D_helper(axes[2], contour)(
        x_actual[:, 0].reshape(sq_len, sq_len),
        x_actual[:, 1].reshape(sq_len, sq_len),
        ground_truth.reshape(sq_len, sq_len),
        cmap="PiYG",
        norm=CenteredNorm(),
    )
    fig.colorbar(img, ax=axes.ravel().tolist(), shrink=0.95, pad=0.1)
    set_titles_labels(axes, ["Custom Critic", "Standard Critic", "Ground Truth"])
    plt.savefig(fn, bbox_inches="tight", dpi=300)
    print(f"Saved to {fn}")


def plot_training_data_dist(npy_fn, save_fn):
    data = np.load(npy_fn)
    gs_kw = dict(width_ratios=[1.8, 1], height_ratios=[1, 1])
    fig, axd = plt.subplot_mosaic(
        [["left", "upper right"], ["left", "lower right"]],
        gridspec_kw=gs_kw,
        figsize=(15, 6),
        layout="constrained",
    )
    hist = axd["lower right"].hist(data[:, 0], bins=100)
    hist = axd["upper right"].hist(data[:, 1], bins=100, orientation="horizontal")
    _, _, _, hist = axd["left"].hist2d(data[:, 0], data[:, 1], bins=100, cmap="BrBG")
    fig.colorbar(hist, ax=list(axd.values()), shrink=0.95, pad=0.1)
    set_titles_labels([axd["left"]], ["Training Data Distribution"])
    set_titles_labels([axd["lower right"]], xy_labels=["theta (rad)", " "])
    set_titles_labels([axd["upper right"]], xy_labels=[" ", "omega (rad/s)"])
    plt.savefig(save_fn, bbox_inches="tight", dpi=300)
    print(f"Saved to {save_fn}")


def plot_theta_omega_polar(theta, omega, save_fn):
    r = np.ones_like(theta)
    ax = plt.subplot(111, projection="polar")
    c = ax.scatter(theta, r, c=omega, cmap="RdYlGn", alpha=0.75)
    plt.colorbar(c)
    ax.set_title("Pos, Vel of High Return Initial Conditions")
    plt.savefig(save_fn, bbox_inches="tight", dpi=300)
    print(f"Saved to {save_fn}")


def plot_trajectories(pos, vel, torques, rewards, fn, title=""):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
    (hr_pos,) = axes[0, 0].plot(pos[:, :1], label="High Return", color="red", alpha=0.7)
    (lr_pos,) = axes[0, 0].plot(pos[:, 1:], label="Low Return", color="blue", alpha=0.7)
    set_titles_labels([axes[0, 0]], xy_labels=["Steps", "Theta (rad)"])
    axes[0, 0].legend(handles=[hr_pos, lr_pos])
    (hr_vel,) = axes[0, 1].plot(vel[:, :1], label="High Return", color="red")
    (lr_vel,) = axes[0, 1].plot(vel[:, 1:], label="High Return", color="blue")
    set_titles_labels([axes[0, 1]], xy_labels=["Steps", "Omega (rad/s)"])
    axes[0, 1].legend(handles=[hr_vel, lr_vel])
    (hr_torque,) = axes[1, 0].plot(torques[:, :1], label="High Return", color="red")
    (lr_torque,) = axes[1, 0].plot(torques[:, 1:], label="Low Return", color="blue")
    set_titles_labels([axes[1, 0]], xy_labels=["Steps", "Torque (Nm)"])
    axes[1, 0].legend(handles=[hr_torque, lr_torque])
    (hr_rewards,) = axes[1, 1].plot(rewards[:, :1], label="High Return", color="red")
    (lr_rewards,) = axes[1, 1].plot(rewards[:, 1:], label="Low Return", color="blue")
    set_titles_labels([axes[1, 1]], xy_labels=["Steps", "Reward"])
    axes[1, 1].legend(handles=[hr_rewards, lr_rewards])
    fig.suptitle(title)

    plt.savefig(fn, bbox_inches="tight", dpi=300)
    print(f"Saved to {fn}")


def plot_predictions_and_gradients(
    dim,
    x_actual,
    y_pred,
    y_actual,
    pred_grad,
    fn,
    colormap_diff=False,
    colormap_values=True,
    actual_grad=None,
):
    if dim > 3:
        print("Dimension greater than 3, cannot graph")
    elif dim == 3:
        x_actual = x_actual.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        y_actual = y_actual.detach().cpu().numpy()
        if actual_grad is not None:
            actual_grad = (
                actual_grad.detach().cpu().numpy()
                if not isinstance(actual_grad, np.ndarray)
                else actual_grad
            )
            if colormap_diff:
                fig, (ax1, ax2) = plt.subplots(
                    nrows=1, ncols=2, figsize=(16, 8), layout="tight"
                )
                sq_len = int(sqrt(x_actual.shape[0]))
                img1 = graph_3D_helper(ax1)(
                    x_actual[:, 0].reshape(sq_len, sq_len),
                    x_actual[:, 1].reshape(sq_len, sq_len),
                    np.linalg.norm(
                        np.vstack(
                            [grad[-1].detach().cpu().numpy() for grad in pred_grad]
                        )
                        - actual_grad,
                        axis=1,
                    ).reshape(sq_len, sq_len),
                    cmap="OrRd",
                    vmin=0.0,
                    vmax=30.0,
                )
                plt.colorbar(img1, ax=ax1)
                img2 = graph_3D_helper(ax2)(
                    x_actual[:, 0].reshape(sq_len, sq_len),
                    x_actual[:, 1].reshape(sq_len, sq_len),
                    (y_pred - y_actual).reshape(sq_len, sq_len),
                    cmap="bwr",
                    vmin=-2.0,
                    vmax=2.0,
                )
                plt.colorbar(img2, ax=ax2)
                set_titles_labels(
                    [ax1, ax2],
                    [
                        "Gradient Error between Predictions and Targets",
                        "Actual Error between Predictions and Targets",
                    ],
                    ["x", "y"],
                )
                plt.savefig(f"{fn}_diff_colormap.png")
        if colormap_values:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
            scatter = axes[0].scatter(
                x_actual[:, 0],
                x_actual[:, 1],
                c=y_pred,
                cmap="viridis",
                marker="o",
                alpha=0.7,
            )
            scatter = axes[1].scatter(
                x_actual[:, 0],
                x_actual[:, 1],
                c=y_actual,
                cmap="viridis",
                marker="^",
                alpha=0.7,
            )
            fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.95, label="f(x)")
            axes[0].set_title("Pointwise Predictions")
            axes[1].set_title("Pointwise Targets")
            plt.savefig(f"{fn}_values_colormap.png")
        else:
            fig = plt.figure(constrained_layout=True)
            ax = fig.add_subplot(projection="3d")
            ax.scatter(
                x_actual[:, 0], x_actual[:, 1], y_pred, marker="o", label="Predicted"
            )
            ax.scatter(
                x_actual[:, 0], x_actual[:, 1], y_actual, marker="^", label="Actual"
            )
            ax.set_xlim(-15, 15)
            ax.set_ylim(-15, 15)
            ax.set_zlim(-15, 250)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("f(x, y)")
            ax.legend(loc="upper left")
            ax.set_title("Pointwise Predictions vs Actual")
            plt.legend(loc="upper left")
            for a in [45, 135, 180, 225]:
                ax.view_init(elev=10.0, azim=a)
                plt.savefig(f"{fn}_angle_{a}.png")
        if colormap_diff:
            fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
            scatter = axis.scatter(
                x_actual[:, 0],
                x_actual[:, 1],
                c=y_pred - y_actual,
                cmap="bwr",
                marker="o",
                alpha=0.5,
                norm=CenteredNorm(),
            )
            fig.colorbar(scatter, ax=axis, shrink=0.95, label="f(x)")
            axis.set_title("Error Between Pointwise Predictions and Targets")
            plt.savefig(f"{fn}_value_error_colormap.png")
    elif dim == 2:
        x_actual = x_actual.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        y_actual = y_actual.detach().cpu().numpy()

        _, ax = plt.subplots()
        ax.scatter(x_actual, y_pred, label="Predicted")
        ax.scatter(x_actual, y_actual, label="Actual", alpha=0.5)

        for g in pred_grad:
            line_x = np.linspace(g[0].item() - 0.5, g[0].item() + 0.5, 100)
            line_y = g[2].item() * (line_x - g[0].item()) + g[1].item()
            ax.plot(line_x, line_y, color="green", alpha=0.5)

        # * quadratic settings
        ax.set_ylim(-10, 250)
        ax.set_xlim(-15, 15)
        # * cosine settings
        # ax.set_ylim(-0.5, 2.5)
        # ax.set_xlim(-10.5, 10.5)
        ax.set_ylabel("y")
        ax.set_xlabel("x")
        ax.legend(loc="upper left")
        ax.set_title("Pointwise Predictions vs Actual")
        plt.savefig(f"{fn}.png")
    else:
        print("Dimension less than 2, cannot graph")


def plot_autoencoder(targets, predictions, fn):
    targets = targets.detach().cpu().numpy()
    predictions = predictions.detach().cpu().numpy()

    _, ax = plt.subplots()
    ax.scatter(targets[:, 0], targets[:, 1], label="Targets", alpha=0.5)
    ax.scatter(predictions[:, 0], predictions[:, 1], label="Predictions", alpha=0.5)
    ax.set_xlim(
        min(np.min(targets[:, 0]), np.min(predictions[:, 0])),
        max(np.max(targets[:, 0]), np.max(predictions[:, 0])),
    )
    ax.set_ylim(
        min(np.min(targets[:, 1]), np.min(predictions[:, 1])),
        max(np.max(targets[:, 1]), np.max(predictions[:, 1])),
    )
    ax.set_xlabel("theta")
    ax.set_ylabel("theta dot")
    ax.legend(loc="upper left")
    ax.set_title("Pointwise Predictions vs Actual")
    plt.savefig(f"{fn}.png")


def plot_loss(loss_arr, fn, title="Training Loss Across Epochs"):
    _, ax = plt.subplots()
    ax.plot(loss_arr)
    ax.set_ylim(min(0, min(loss_arr) - 1.0), max(loss_arr) + 1.0)
    ax.set_xlim(0, len(loss_arr))
    ax.set_ylabel("Average Batch MSE")
    ax.set_xlabel("Epoch")
    ax.set_title(title)
    plt.savefig(f"{fn}.png")


# def plot_learning_progress(test_error, fn="test_error"):
#     _, ax = plt.subplots()
#     for name, error in test_error.items():
#         error_mean = [x.mean() for x in error]
#         error_std = [x.std() for x in error]

#         # plot shaded error region
#         ax.plot(error_mean, label=name)
#         ax.fill_between(
#             range(len(error_mean)),
#             [x - y for x, y in zip(error_mean, error_std)],
#             [x + y for x, y in zip(error_mean, error_std)],
#             alpha=0.3,
#         )
#     ax.set_ylabel("Average Test Error")
#     plt.savefig(f"{fn}.png")


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def plot_learning_progress(
    test_error, title, fn="test_error", smoothing_window=30, extension="pdf"
):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    for name, error in test_error.items():
        # error_mean = np.array([x.mean() for x in error])
        error_mean = np.array(error)
        # exit()
        # error_std = np.array([x.std() for x in error])
        error_change = np.abs(np.diff(moving_average(error_mean, smoothing_window)))
        # smoothed_error_change = moving_average(error_change, smoothing_window)

        # plot shaded error region
        ax1.plot(error_mean, label=name)
        # ax1.fill_between(
        #     range(len(error_mean)),
        #     [x - y for x, y in zip(error_mean, error_std)],
        #     [x + y for x, y in zip(error_mean, error_std)],
        #     alpha=0.3,
        # )
        # plot change in error
        ax2.plot(error_change, label=name)

    ax1.set_ylabel("Average Test Error")
    ax2.set_ylabel("Change in Average Test Error")
    ax2.set_xlabel("Iteration")
    # ax2.set_yscale("log")
    ax1.legend()
    ax2.legend()
    fig.suptitle(title, fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{fn}.{extension}", dpi=300, bbox_inches="tight")
