import matplotlib.pyplot as plt


def plot_predictions_and_gradients():
    pass


def plot_pointwise_predictions(x_actual, y_pred, y_actual, fn):
    x_actual = x_actual.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    y_actual = y_actual.detach().cpu().numpy()

    fig, ax = plt.subplots()
    ax.scatter(x_actual, y_pred, label="Predicted")
    ax.scatter(x_actual, y_actual, label="Actual", alpha=0.5)
    ax.set_ylim(-10, 250)
    ax.set_xlim(-15, 15)
    ax.set_ylabel("y")
    ax.set_xlabel("x")
    ax.legend(loc="upper left")
    ax.set_title("Pointwise Predictions vs Actual")
    plt.savefig(f"{fn}.png")


def plot_loss(loss_arr, fn):
    fig, ax = plt.subplots()
    ax.plot(loss_arr)
    ax.set_ylim(min(0, min(loss_arr) - 1.0), max(loss_arr) + 1.0)
    ax.set_xlim(0, len(loss_arr))
    ax.set_ylabel("Average Batch MSE")
    ax.set_xlabel("Epoch")
    ax.set_title("Training Loss Across Epochs")
    plt.savefig(f"{fn}.png")
