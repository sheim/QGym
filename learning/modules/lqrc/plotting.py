import matplotlib.pyplot as plt
import numpy as np


def plot_predictions_and_gradients(
    dim, x_actual, y_pred, y_actual, gradients, fn, contour=False
):
    if dim > 3:
        print("Dimension greater than 3, cannot graph")
    elif dim == 3:
        x_actual = x_actual.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        y_actual = y_actual.detach().cpu().numpy()

        if contour:
            fig, ax = plt.subplots(layout="constrained")
            scatter = ax.scatter(
                x_actual[:, 0],
                x_actual[:, 1],
                c=y_pred,
                cmap="viridis",
                marker="o",
                edgecolors="k",
            )
            scatter = ax.scatter(
                x_actual[:, 0],
                x_actual[:, 1],
                c=y_actual,
                cmap="viridis",
                marker="^",
                edgecolors="k",
            )
            plt.colorbar(scatter, ax=ax, label="f(x)")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title("Pointwise Predictions")
            plt.savefig(f"{fn}_colormap.png")
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

        for g in gradients:
            line_x = np.linspace(g[0].item() - 0.5, g[0].item() + 0.5, 100)
            line_y = g[2].item() * (line_x - g[0].item()) + g[1].item()
            ax.plot(line_x, line_y, color="green", alpha=0.5)

        # quadratic settings
        ax.set_ylim(-10, 250)
        ax.set_xlim(-15, 15)
        # cosine settings
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
