from math import sqrt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import CenteredNorm, ListedColormap
import matplotlib as mpl
import matplotlib.colors as mcolors
from tabulate import tabulate

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


def plot_eigenval_hist(
    data, fn, n_bins=100, title="Histogram of Eigenvalues", extension="png"
):
    fig, ax = plt.subplots(1, tight_layout=True)
    ax.hist(data, bins=n_bins)
    ax.set_xlabel("Eigenvalue")
    ax.set_title(title)
    ax.set_yscale("log")
    plt.savefig(f"{fn}.{extension}", dpi=300, bbox_inches="tight")
    print("Histogram of eigenvalues saved to", f"{fn}.{extension}")


def plot_variable_lr(data, fn, title="Scheduled Learning Rate", extension="png"):
    fig, axes = plt.subplots(
        nrows=len(list(data.keys())),
        ncols=1,
        figsize=(5 + 4 * len(list(data.keys())), 10),
    )
    for ix, name in enumerate(data.keys()):
        ax = axes[ix] if isinstance(axes, np.ndarray) else axes
        values = data[name]
        ax.plot(values, label=name)
        ax.legend()
        ax.set_ylabel("Learning Rate")
        ax.set_xlabel("Epoch")
    fig.suptitle(title, fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{fn}.{extension}", dpi=300, bbox_inches="tight")
    print("Learning rate change as a functin of epoch saved to", f"{fn}.{extension}")


def plot_binned_errors_ampc(
    data,
    fn,
    lb=0,
    ub=10e10,
    step=1000,
    tick_step=5000,
    title_add_on="",
    extension="png",
):
    display_names = {
        "OuterProduct": "Outer Product",
        "OuterProductLatent": "Outer Product Latent",
        "CholeskyLatent": "Cholesky Latent",
        "DenseSpectralLatent": "Spectral Latent",
        "Critic": "Critic",
    }
    fig, axes = plt.subplots(
        nrows=len(list(data["critic_obs"].keys())),
        ncols=1,
        figsize=(16, 15),
        layout="constrained",
    )
    fig.suptitle(f"Pointwise Prediction Error for {title_add_on} \n", fontsize=27)

    bins = np.arange(lb, ub, step)
    bin_labels = [
        "<" + str(np.round(bins[ix], decimals=1))
        for ix in range(0, len(bins), tick_step)
    ]
    y_min = float("inf")
    y_max = -float("inf")

    for ix, critic in enumerate(data["critic_obs"].keys()):
        critic_data = data["error"][critic].squeeze().detach().cpu().numpy()
        num_bins = bins.shape[0] + 1
        bincount = np.bincount(np.digitize(critic_data, bins), minlength=num_bins)

        y_min = bincount.min() if bincount.min() < y_min else y_min
        y_max = bincount.max() + 10 if bincount.max() + 10 > y_max else y_max

        axes[ix].bar(
            np.arange(len(bincount)),
            bincount,
            alpha=0.5,
        )
        # axes formatting
        axes[ix].set_title(display_names[critic], fontsize=25)
        x_ticks = np.arange(0, len(bins), tick_step)
        axes[ix].set_xticks(x_ticks, labels=bin_labels)
        axes[ix].set_xlim(
            -min((tick_step / 2.0), 2.0),
            min(len(bins) + (tick_step / 2.0), len(bins) + 2.0),
        )
        axes[ix].tick_params(axis="both", which="major", labelsize=20)
        axes[ix].set_yscale("log")
        axes[ix].set_ylabel("# of Pointwise Comparisons", fontsize=20)
        axes[ix].set_xlabel("Error Magnitude", fontsize=20)
    for ix, critic in enumerate(data["critic_obs"].keys()):
        axes[ix].set_ylim(y_min, y_max)
    fig.legend(
        loc=" outside upper right", fontsize=20, ncol=3, bbox_to_anchor=(0.85, 0.2)
    )
    plt.savefig(fn + f".{extension}", bbox_inches="tight", dpi=300)
    print(f"Saved to {fn}.{extension}")


def plot_binned_errors(
    data,
    fn,
    lb=0,
    ub=500,
    step=20,
    tick_step=5,
    title_add_on="",
    extension="png",
    multi_trial=False,
):
    # set up figure metadata
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    display_names = {
        "Rosenbrock": "Rosenbrock",
        "OuterProduct": "Outer Product",
        "CholeskyLatent": "Cholesky Latent",
        "DenseSpectralLatent": "Spectral Latent",
        "Critic": "Critic",
    }
    num_cols = len(list(list(list(data.values())[0].values())[0].keys()))
    fig, axes = plt.subplots(
        nrows=num_cols, ncols=1, figsize=(16, 15), layout="constrained"
    )
    fig.suptitle(f"Pointwise Prediction Error for {title_add_on} \n", fontsize=27)
    colors = dict(
        zip(
            [param for param in data.keys()],
            ["red", "green", "blue", "purple", "orange"][:num_cols],
        )
    )
    # iterate over the parameter that varies across runs (i.e. learning rate, percentage of training data used, etc.)
    for param in data.keys():
        bins = np.arange(lb, ub, step)
        bin_labels = [
            "<" + str(np.round(bins[ix], decimals=1))
            for ix in range(0, len(bins), tick_step)
        ]
        y_min = float("inf")
        y_max = -float("inf")

        for ix, critic in enumerate(data[param]["critic_obs"].keys()):
            critic_data = data[param]["error"][critic].squeeze().detach().cpu().numpy()
            num_bins = bins.shape[0] + 1
            bincount = np.bincount(
                np.digitize(critic_data.mean(axis=0), bins), minlength=num_bins
            )

            if multi_trial:
                digitized = np.digitize(critic_data, bins)
                bincount_by_row = np.zeros((digitized.shape[0], num_bins))
                for jx, row in enumerate(digitized):
                    bincount_by_row[jx, ...] = np.bincount(row, minlength=num_bins)
                # print out mean error binned and std deviation of error binning across trials
                # print(f"************{critic} at {param}****************")
                # print("Bins", bins)
                # print(np.vstack((bincount, bincount_by_row.std(axis=0))))
                # Calculate percentages
                total_count = np.sum(bincount)
                percentages = (bincount / total_count) * 100

                # Prepare data for the table
                table_data = []
                for i in range(len(bins) - 1):
                    bin_range = f"{bins[i]:.2f} - {bins[i+1]:.2f}"
                    frequency = bincount[i]
                    percentage = percentages[i]
                    table_data.append([bin_range, frequency, f"{percentage:.2f}%"])

                # Create and print the table
                headers = ["Bin Range", "Frequency", "Percentage"]
                table = tabulate(table_data, headers=headers, tablefmt="grid")
                print(f"************{critic} at {param}****************")
                print(table)

            y_min = bincount.min() if bincount.min() < y_min else y_min
            y_max = bincount.max() + 10 if bincount.max() + 10 > y_max else y_max

            # to avoid label repetitions in legend
            if ix == 0:
                axes[ix].bar(
                    np.arange(len(bincount)),
                    bincount,
                    color=colors[param],
                    alpha=0.5,
                    label=str(param),
                )
            else:
                axes[ix].bar(
                    np.arange(len(bincount)), bincount, color=colors[param], alpha=0.5
                )
            # axes formatting
            axes[ix].set_title(display_names[critic], fontsize=25)
            x_ticks = np.arange(0, len(bins), tick_step)
            axes[ix].set_xticks(x_ticks, labels=bin_labels)
            axes[ix].set_xlim(
                -min((tick_step / 2.0), 2.0),
                min(len(bins) + (tick_step / 2.0), len(bins) + 2.0),
            )
            axes[ix].tick_params(axis="both", which="major", labelsize=15)
        for ix, critic in enumerate(data[param].keys()):
            axes[ix].set_ylim(y_min, y_max)
    fig.legend(
        loc=" outside upper right", fontsize=20, ncol=3, bbox_to_anchor=(0.85, 0.2)
    )
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
    extension="png",
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


def plot_multiple_critics_w_data(
    x,
    predictions,
    targets,
    title,
    fn,
    data,
    display_names,
    grid_size=64,
    extension="png",
    data_title="Training Data Distribution",
    log_norm=True,
    task=None,
    data_dist_ix=0,
    scatter=False,
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

    if "Rosenbrock" in task:
        x_coord = data[:, 0]
        y_coord = data[:, 1]
        ground_truth = "Rosenbrock"
    elif "Pendulum" in task:
        x_coord = data[:, :, 0] * 2 * np.pi
        y_coord = data[:, :, 1] * 5
        ground_truth = "Ground Truth MC Returns"
    elif "Unicycle" in task:
        x_coord = data[:, 0]
        y_coord = data[:, 1]
        ground_truth = "Unicycle"
    else:
        raise ValueError(
            "Please specify prediction task when calling plotting function."
        )

    # unify colorbar min maxes across plotting data
    for critic_name in x:
        np_predictions = predictions[critic_name].detach().cpu().numpy().reshape(-1)
        np_targets = targets[ground_truth].detach().cpu().numpy().reshape(-1)
        np_error = np_predictions - np_targets
        global_min_error = min(global_min_error, np.min(np_error))
        global_max_error = max(global_max_error, np.max(np_error))
        global_min_prediction = np.min(np_targets)
        global_max_prediction = np.max(np_targets)

    error_norm = (
        mcolors.LogNorm()
        if log_norm
        else mcolors.TwoSlopeNorm(
            vmin=global_min_error, vcenter=0, vmax=global_max_error
        )
    )  # left this as regular lognorm because symlognorm changes custom colormap markers, will throw error if vmin < -60.0 (so train for 1000 grad steps, not 200)
    pred_vcenter = (global_max_prediction + global_min_prediction) / 2
    pred_halfrange = (global_max_prediction - global_min_prediction) / 2
    prediction_norm = (
        mcolors.SymLogNorm(
            linthresh=0.01,
            linscale=0.01,
            vmin=pred_vcenter - pred_halfrange,
            vmax=pred_vcenter + pred_halfrange,
        )
        if log_norm
        else mcolors.CenteredNorm(vcenter=pred_vcenter, halfrange=pred_halfrange)
    )

    for ix, critic_name in enumerate(x):
        np_x = x[critic_name].detach().cpu().numpy().reshape(-1, 2)
        np_predictions = predictions[critic_name].detach().cpu().numpy().reshape(-1)
        np_targets = targets[ground_truth].detach().cpu().numpy().reshape(-1)
        np_error = np_predictions - np_targets

        if scatter:
            axes[0, ix].scatter(
                np_x[:, 0],
                np_x[:, 1],
                c=np_predictions,
                cmap=prediction_cmap,
                norm=prediction_norm,
            )
        else:
            axes[0, ix].imshow(
                np_predictions.reshape(grid_size, grid_size).T,
                origin="lower",
                extent=(x_coord.min(), x_coord.max(), y_coord.min(), y_coord.max()),
                cmap=prediction_cmap,
                norm=prediction_norm,
            )
        ax_title = (
            ground_truth
            if ground_truth in critic_name
            else f"{display_names[critic_name]} Prediction"
        )
        axes[0, ix].set_title(ax_title, fontsize=25)

        axes[0, ix].tick_params(axis="both", which="major", labelsize=15)
        axes[1, ix].tick_params(axis="both", which="major", labelsize=15)

        # skip error plotting in bottom left subplot so it
        # can later be used for plotting training data distribution
        if ix == data_dist_ix:
            continue

        # plot pointwise error
        if scatter:
            axes[1, ix].scatter(
                np_x[:, 0], np_x[:, 1], c=np_error, cmap=error_cmap, norm=error_norm
            )
        else:
            axes[1, ix].imshow(
                np_error.reshape(grid_size, grid_size).T,
                origin="lower",
                extent=(x_coord.min(), x_coord.max(), y_coord.min(), y_coord.max()),
                cmap=error_cmap,
                norm=error_norm,
            )
        axes[1, ix].set_title(f"{display_names[critic_name]} Error", fontsize=25)

    # plot training data distribution
    axes[1, data_dist_ix].scatter(x_coord, y_coord, alpha=0.75, s=3)
    axes[1, data_dist_ix].set_title(data_title, fontsize=25)
    axes[1, data_dist_ix].set_xlabel(
        "x" if ground_truth == "Rosenbrock" or ground_truth == "Unicycle" else "theta",
        fontsize=15,
    )
    axes[1, 0].set_ylabel(
        "y" if ground_truth == "Rosenbrock" or ground_truth == "Unicycle" else "omega",
        fontsize=15,
    )

    # Ensure the axes are the same for all plots
    for ax in axes.flat:
        ax.set_xlim([x_coord.min(), x_coord.max()])
        ax.set_ylim([y_coord.min(), y_coord.max()])

    # match data plot size to imshow pointwise plot size
    axes[1, 0].set_aspect("equal", adjustable="box")

    pred_cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=prediction_norm, cmap=prediction_cmap),
        ax=axes[0, :].ravel().tolist(),
        shrink=0.95,
    )
    pred_cbar.ax.tick_params(labelsize=25)
    error_cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=error_norm, cmap=error_cmap),
        ax=axes[1, :].ravel().tolist(),
        shrink=0.95,
    )
    error_cbar.ax.tick_params(labelsize=25)
    fig.suptitle(title, fontsize=30)
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


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def plot_learning_progress(
    test_error,
    title,
    fn="test_error",
    smoothing_window=30,
    extension="png",
    y_label1="Average Test Error",
    y_label2="Change in Average Test Error",
):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    for name, error in test_error.items():
        # error_mean = np.array([x.mean() for x in error])
        error_mean = np.array(error)
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

    ax1.set_ylabel(y_label1)
    ax1.set_yscale("log")
    # ax1.set_ylim(bottom=2.08e10, top=2.11e10)
    ax2.set_ylabel(y_label2)
    ax2.set_xlabel("Iteration")
    ax2.set_yscale("log")
    ax1.legend()
    ax2.legend()
    fig.suptitle(title, fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{fn}.{extension}", dpi=300, bbox_inches="tight")
    print("Learning progress plot saved to", f"{fn}.{extension}")


def plot_critic_3d_interactive(
    x,
    targets,
    A,
    xy_eval,
    cost_true,
    cost_eval,
    x_offsets,
    display_names,
    W_latent=None,
    b_latent=None,
    dmax=None,
    ground_truth="Unicycle",
):
    num_critics = len(A.keys())
    # create figure
    fig = plt.figure(figsize=(20, 8))
    axes = []
    for i in range(num_critics):
        num = int(f"1{num_critics}{i+1}")
        axes.append(fig.add_subplot(num, projection="3d"))

    for ix, critic_name in enumerate(A):
        if critic_name == ground_truth:
            continue
        # turn torch tensors to numpy arrays for graphing
        np_x = x.detach().cpu().numpy().reshape(-1, 2)
        np_targets = targets.detach().cpu().numpy().reshape(-1)
        np_A = A[critic_name].detach().cpu().numpy()
        np_xy_eval = xy_eval[critic_name].detach().cpu().numpy()
        np_cost_true = cost_true[critic_name].detach().cpu().numpy()
        np_cost_eval = cost_eval[critic_name].detach().cpu().numpy()
        # optionals
        np_x_offset = (
            np.zeros_like(np_x[0])
            if x_offsets[critic_name] is None
            else x_offsets[critic_name].detach().cpu().numpy()
        )
        np_W_latent = W_latent[critic_name]
        np_b_latent = b_latent[critic_name]

        # plot targets
        axes[ix].scatter(
            np_x[:, 0], np_x[:, 1], np_targets, c=np_targets, cmap="viridis", marker="o"
        )
        # plot eval points
        axes[ix].scatter(
            np_xy_eval[0],
            np_xy_eval[1],
            np_cost_true,
            color="blue",
            marker="X",
            s=100,
            label="Cost True",
        )
        axes[ix].scatter(
            np_xy_eval[0],
            np_xy_eval[1],
            np_cost_eval,
            color="red",
            marker="X",
            s=100,
            label="Cost Eval",
        )

        # plot predictions
        x_range = np.linspace(min(np_x[:, 0]), max(np_x[:, 0]), 50)
        y_range = np.linspace(min(np_x[:, 1]), max(np_x[:, 1]), 50)
        X, Y = np.meshgrid(x_range, y_range)

        # Check if dmax is specified and limit the plot range
        if xy_eval is not None and dmax is not None:
            distances = np.sqrt((X - np_xy_eval[0]) ** 2 + (Y - np_xy_eval[1]) ** 2)
            mask = distances <= dmax  # Mask for points within dmax radius
        else:
            mask = np.ones_like(X, dtype=bool)  # No restriction if dmax is None

        zero_padding = [
            0 for _ in range(x_offsets[critic_name].shape[0] - 2)
        ]  # pad non-xy parts of state with 0s for grid
        if np_W_latent is not None and np_b_latent is not None:
            np_W_latent = np_W_latent.detach().cpu().numpy()
            np_b_latent = np_b_latent.detach().cpu().numpy()
            # Compute Z based on the quadratic surface equation
            Z = np.array(
                [
                    [
                        (
                            (np.array([xi, yi, *zero_padding]) - np_x_offset)
                            @ np_W_latent.T
                            + np_b_latent
                        )
                        @ np_A
                        @ (
                            (np.array([xi, yi, *zero_padding]) - np_x_offset)
                            @ np_W_latent.T
                            + np_b_latent
                        )
                        if mask[i, j]
                        else np.nan
                        for j, xi in enumerate(x_range)  # due to meshgrid "xy" indexing
                    ]
                    for i, yi in enumerate(y_range)
                ]
            )
        else:
            # Compute Z based on the quadratic surface equation
            Z = np.array(
                [
                    [
                        (np.array([xi, yi, *zero_padding]) - np_x_offset)
                        @ np_A
                        @ (np.array([xi, yi, *zero_padding]) - np_x_offset)
                        if mask[i, j]
                        else np.nan
                        for j, xi in enumerate(x_range)
                    ]
                    for i, yi in enumerate(y_range)
                ]
            )

        # Plot the surface
        axes[ix].plot_surface(
            X, Y, np.squeeze(Z), cmap="viridis", alpha=0.5, edgecolor="none"
        )

        axes[ix].set_xlabel("X")
        axes[ix].set_ylabel("Y")
        axes[ix].set_zlabel("Z")
        axes[ix].set_title(f"{display_names[critic_name]} Prediction")
        axes[ix].set_box_aspect([1, 1, 0.75])
        axes[ix].set_zlim(0, 1)
        axes[ix].legend()

        # Set initial viewing angle
        axes[ix].view_init(elev=20, azim=45)

    # Variables to track which axis is being interacted with
    active_ax = None
    button_pressed = False

    # Mouse button press event handler
    def on_button_press(event):
        global active_ax, button_pressed
        if event.inaxes in axes:
            active_ax = event.inaxes
            button_pressed = True

    # Mouse button release event handler
    def on_button_release(event):
        global button_pressed
        button_pressed = False

    # Mouse motion event handler
    def on_move(event):
        if button_pressed and active_ax is not None:
            active_ax.view_init(elev=active_ax.elev, azim=active_ax.azim)
            fig.canvas.draw_idle()

    # Connect event handlers
    fig.canvas.mpl_connect("button_press_event", on_button_press)
    fig.canvas.mpl_connect("button_release_event", on_button_release)
    fig.canvas.mpl_connect("motion_notify_event", on_move)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    plt.show()


def plot_robot(
    shooting_nodes,
    u_max,
    U,
    X_traj,
    x_labels,
    u_labels,
    obst_pos=None,  # List of obstacle positions (x, y)
    obst_rad=None,  # List of obstacle radii
    time_label="t",
    plt_show=True,
    plt_name=None,
    x_max=None,
):
    """
    Params:
        shooting_nodes: time values of the discretization
        u_max: maximum absolute value of u
        U: arrray with shape (N_sim-1, nu) or (N_sim, nu)
        X_traj: arrray with shape (N_sim, nx)
        obst_pos: List of (x, y) obstacle positions
        obst_rad: List of obstacle radii corresponding to obst_pos
    """

    N_sim = X_traj.shape[0]
    nx = X_traj.shape[1]
    nu = U.shape[1]

    # Create gridspec layout: 1 column on the right spanning all rows, and nx + nu rows on the left
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(nx + nu, 2, width_ratios=[2, 3], wspace=0.3)

    t = shooting_nodes
    for i in range(nu):
        ax_u = fig.add_subplot(gs[i, 0])
        (line,) = ax_u.step(t, np.append([U[0, i]], U[:, i]))

        ax_u.set_ylabel(u_labels[i])
        # ax_u.set_xlabel(time_label)
        if u_max[i] is not None:
            ax_u.hlines(u_max[i], t[0], t[-1], linestyles="dashed", alpha=0.7)
            ax_u.hlines(-u_max[i], t[0], t[-1], linestyles="dashed", alpha=0.7)
            ax_u.set_ylim([-1.2 * u_max[i], 1.2 * u_max[i]])
        ax_u.grid()

    for i in range(nx):
        ax_x = fig.add_subplot(gs[i + nu, 0])
        (line,) = ax_x.plot(t, X_traj[:, i])

        ax_x.set_ylabel(x_labels[i])
        if x_max is not None and x_max[i] is not None:
            ax_x.hlines(x_max[i], t[0], t[-1], linestyles="dashed", alpha=0.7)
            ax_x.hlines(-x_max[i], t[0], t[-1], linestyles="dashed", alpha=0.7)
            ax_x.set_ylim(-1.2 * x_max[i], 1.2 * x_max[i])
        ax_x.grid()
    ax_x.set_xlabel(time_label)

    # New plot for X_traj[:, 0] vs X_traj[:, 1] with equal axis scaling
    ax_xy = fig.add_subplot(gs[:, 1])  # Span the entire right column
    ax_xy.plot(X_traj[:, 0], X_traj[:, 1], label="Trajectory")
    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")
    ax_xy.set_xlim([-0.6, 0.6])
    ax_xy.set_ylim([-0.6, 0.6])
    ax_xy.set_aspect("equal")  # Ensure equal scaling of axes
    ax_xy.grid()
    ax_xy.set_title(plt_name.split("/")[-1])

    # Plot obstacles as gray circles
    if obst_pos is not None and obst_rad is not None:
        for (x, y), r in zip(obst_pos, obst_rad):
            circle = plt.Circle((x, y), r, color="gray", alpha=0.5)
            ax_xy.add_patch(circle)

    plt.legend()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)

    if plt_name:
        plt.savefig(f"{plt_name}")

    if plt_show:
        plt.show()

    plt.close()
