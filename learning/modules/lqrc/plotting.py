import os
import matplotlib.pyplot as plt

from learning import LEGGED_GYM_LQRC_DIR


def plot_predictions_and_gradients():
    pass


def plot_pointwise_predictions(x_actual, y_pred, y_actual, fn):
    x_actual = x_actual.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    y_actual = y_actual.cpu().numpy()
    plt.scatter(x_actual, y_pred, label="Predicted")
    plt.scatter(x_actual, y_actual, label="Actual")
    plt.ylabel("y")
    plt.xlabel("x")
    plt.legend(loc="upper left")
    plt.title("Pointwise Predictions vs Actual")
    if not os.path.exists(f"{LEGGED_GYM_LQRC_DIR}/graphs"):
        path = os.path.join(LEGGED_GYM_LQRC_DIR, "graphs")
        os.makedirs(path)
    save_path = os.path.join(LEGGED_GYM_LQRC_DIR, "graphs")
    plt.savefig(f"{save_path}/{fn}.png")

    print("finished")  # using this as a debugger hook, will remove later
