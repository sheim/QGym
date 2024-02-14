from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import CenteredNorm, TwoSlopeNorm


def graph_3D_helper(ax, contour):
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
    x_actual = x_actual.detach().cpu().numpy()
    custom_error = custom_error.detach().cpu().numpy()
    standard_error = standard_error.detach().cpu().numpy()
    ground_truth = ground_truth.detach().cpu().numpy()

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
    set_titles_labels(
        axes, ["Custom Critic Error", "Standard Critic Error"]
    )
    plt.savefig(fn, bbox_inches="tight", dpi=300)
    print(f"Saved to {fn}")


def plot_value_func(
    x_actual, custom, standard, ground_truth, fn, contour
):
    x_actual = x_actual.detach().cpu().numpy()
    custom = custom.detach().cpu().numpy()
    standard = standard.detach().cpu().numpy()
    ground_truth = ground_truth.detach().cpu().numpy()

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
    set_titles_labels(
        axes, ["Custom Critic", "Standard Critic", "Ground Truth"]
    )
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
    ax = plt.subplot(111, projection='polar')
    c = ax.scatter(theta, r, c=omega, cmap='hsv', alpha=0.75)
    ax.set_title("Pos, Vel of High Return Initial Conditions")
    plt.savefig(save_fn, bbox_inches="tight", dpi=300)
    print(f"Saved to {save_fn}")


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
            actual_grad = actual_grad.detach().cpu().numpy()
            if colormap_diff:
                fig, (ax1, ax2) = plt.subplots(
                    nrows=1, ncols=2, figsize=(16, 8), layout="tight"
                )
                sq_len = int(sqrt(x_actual.shape[0]))
                img1 = ax1.contourf(
                    x_actual[:, 0].reshape(sq_len, sq_len),
                    x_actual[:, 1].reshape(sq_len, sq_len),
                    np.linalg.norm(
                        np.vstack(
                            [grad[-1].detach().cpu().numpy() for grad in pred_grad]
                        )
                        - actual_grad,
                        axis=1,
                    ).reshape(sq_len, sq_len),
                    cmap="RdBu_r",
                    norm=TwoSlopeNorm(0),
                )
                plt.colorbar(img1, ax=ax1)
                img2 = ax2.contourf(
                    x_actual[:, 0].reshape(sq_len, sq_len),
                    x_actual[:, 1].reshape(sq_len, sq_len),
                    (y_pred - y_actual).reshape(sq_len, sq_len),
                    cmap="PuOr",
                    norm=TwoSlopeNorm(0),
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
                edgecolors="k",
            )
            scatter = axes[1].scatter(
                x_actual[:, 0],
                x_actual[:, 1],
                c=y_actual,
                cmap="viridis",
                marker="^",
                edgecolors="k",
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


def plot_pointwise_predictions(dim, x_actual, y_pred, y_actual, fn):
    if dim > 3:
        print("Dimension greater than 3, cannot graph")
    elif dim == 3:
        x_actual = x_actual.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        y_actual = y_actual.detach().cpu().numpy()

        fig = plt.figure(constrained_layout=True)
        ax = fig.add_subplot(projection="3d")
        ax.scatter(
            x_actual[:, 0], x_actual[:, 1], y_pred, marker="o", label="Predicted"
        )
        ax.scatter(x_actual[:, 0], x_actual[:, 1], y_actual, marker="^", label="Actual")
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_zlim(-15, 250)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("f(x, y)")
        ax.legend(loc="upper left")
        ax.set_title("Pointwise Predictions vs Actual")
        plt.legend(loc="upper left")
        plt.savefig(f"{fn}.png")
    elif dim == 2:
        x_actual = x_actual.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        y_actual = y_actual.detach().cpu().numpy()

        _, ax = plt.subplots()
        ax.scatter(x_actual, y_pred, label="Predicted")
        ax.scatter(x_actual, y_actual, label="Actual", alpha=0.5)
        ax.set_ylim(-10, 250)
        ax.set_xlim(-15, 15)
        ax.set_ylabel("y")
        ax.set_xlabel("x")
        ax.legend(loc="upper left")
        ax.set_title("Pointwise Predictions vs Actual")
        plt.savefig(f"{fn}.png")
    else:
        print("Dimension less than 2, cannot graph")


def plot_loss(loss_arr, fn):
    _, ax = plt.subplots()
    ax.plot(loss_arr)
    ax.set_ylim(min(0, min(loss_arr) - 1.0), max(loss_arr) + 1.0)
    ax.set_xlim(0, len(loss_arr))
    ax.set_ylabel("Average Batch MSE")
    ax.set_xlabel("Epoch")
    ax.set_title("Training Loss Across Epochs")
    plt.savefig(f"{fn}.png")
