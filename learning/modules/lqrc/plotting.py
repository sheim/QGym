from math import sqrt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import CenteredNorm

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

font = {"size": 12}
matplotlib.rc("font", **font)


def plot_pendulum_multiple_critics(
    x, predictions, targets, title, fn, colorbar_label="f(x)"
):
    num_critics = len(x.keys())
    assert (
        num_critics >= 1
    ), "This function requires at least two critics for graphing. To graph a single critic please use its corresponding graphing function."  # noqa:E501
    fig, axes = plt.subplots(nrows=2, ncols=num_critics, figsize=(6 * num_critics, 10))
    for ix, critic_name in enumerate(x):
        np_x = x[critic_name].detach().cpu().numpy().reshape(-1, 2)
        np_predictions = predictions[critic_name].detach().cpu().numpy().reshape(-1)
        np_targets = targets[critic_name].detach().cpu().numpy().reshape(-1)
        np_error = np_predictions - np_targets

        # predictions
        if np.any(~np.equal(np_predictions, np.zeros_like(np_predictions))):
            pred_scatter = axes[0, ix].scatter(
                np_x[:, 0],
                np_x[:, 1],
                c=np_predictions,
                cmap="PiYG",
                alpha=0.5,
                norm=CenteredNorm(),
            )
        axes[0, ix].set_title(f"{critic_name} Prediction")

        # error
        if np.any(~np.equal(np_error, np.zeros_like(np_error))):
            error_scatter = axes[1, ix].scatter(
                np_x[:, 0],
                np_x[:, 1],
                c=np_error,
                cmap="bwr",
                alpha=0.5,
                norm=CenteredNorm(),
            )
        axes[1, ix].set_title(f"{critic_name} Error")

    fig.colorbar(
        pred_scatter, ax=axes[0, :].ravel().tolist(), shrink=0.95, label=colorbar_label
    )
    fig.colorbar(
        error_scatter, ax=axes[1, :].ravel().tolist(), shrink=0.95, label=colorbar_label
    )
    fig.suptitle(title, fontsize=16)
    plt.savefig(f"{fn}.png")
    print(f"Saved to {fn}.png")


def plot_pendulum_single_critic(
    x, predictions, targets, title, fn, colorbar_label="f(x)"
):
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
