import numpy as np
import matplotlib.pyplot as plt
from learning import LEGGED_GYM_ROOT_DIR
import matplotlib.collections as mcoll


def calculate_total_returns(data, discount_factor=0.995):
    total_rewards = np.stack([data[key] for key in data.keys()], axis=1).sum(axis=1)
    discounts = discount_factor ** np.arange(total_rewards.shape[1])
    returns = (total_rewards * discounts).sum(axis=1)
    return returns


def plot_phase_portraits(data, step=8, jump_threshold=np.pi):
    dof_pos = data["dof_pos"]
    dof_vel = data["dof_vel"]

    plt.figure(figsize=(10, 8))

    for i in range(0, dof_pos.shape[0], step):
        pos = dof_pos[i, :, 0]
        vel = dof_vel[i, :, 0]
        segments = []
        start = 0

        # Detect jumps and segment the data
        for j in range(1, len(pos)):
            if np.abs(pos[j] - pos[j - 1]) > jump_threshold:
                segments.append((start, j))
                start = j
        segments.append((start, len(pos)))  # Add the last segment

        # Plot each segment to avoid jumps
        for start, end in segments:
            plt.plot(pos[start:end], vel[start:end])

        # Mark initial condition
        plt.scatter(pos[0], vel[0], marker="o", s=50, c="blue")  # Initial condition
        # Mark final condition
        plt.scatter(pos[-1], vel[-1], marker="x", s=50, c="red")  # Final condition

    plt.xlabel("dof_pos")
    plt.ylabel("dof_vel")
    plt.title("Phase Portraits for Every 8th Environment")
    plt.grid(True)


def calculate_cumulative_returns(data, discount_factor=0.995):
    total_rewards = np.stack([data[key] for key in data.keys()], axis=1).sum(axis=1)
    discounts = discount_factor ** np.arange(total_rewards.shape[1])
    cumulative_returns = np.cumsum(total_rewards * discounts, axis=1)

    test_return = np.zeros_like(total_rewards)
    for k in reversed(range(cumulative_returns.shape[1] - 1)):
        test_return[:, k] = (
            total_rewards[:, k] + discount_factor * test_return[:, k + 1]
        )
    return test_return


def plot_initial_and_final_conditions(data):
    dof_pos_unwrapped = np.unwrap(data["dof_pos"], axis=-1)
    dof_vel = data["dof_vel"]

    plt.figure(figsize=(10, 8))

    # Plot initial conditions
    plt.scatter(
        dof_pos_unwrapped[:, 0, 0],
        dof_vel[:, 0, 0],
        c="blue",
        label="Initial Conditions",
        alpha=0.5,
    )

    # Plot final conditions
    plt.scatter(
        dof_pos_unwrapped[:, -1, 0],
        dof_vel[:, -1, 0],
        c="red",
        label="Final Conditions",
        alpha=0.5,
    )

    plt.xlabel("dof_pos (unwrapped)")
    plt.ylabel("dof_vel")
    plt.title("Initial and Final Conditions for All Environments")
    plt.legend()
    plt.grid(True)


def make_segments(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def colorline(
    x, y, z=None, cmap=plt.get_cmap("viridis"), norm=plt.Normalize(0.0, 1.0), ax=None
):
    if ax is None:
        ax = plt.gca()
    if z is None:
        z = np.linspace(0, 1, len(x))

    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(
        segments, array=z, cmap=cmap, norm=norm, linewidth=2, alpha=0.7
    )

    ax.add_collection(lc)
    return lc


# can use instead of colorline
def colorscatter(
    x, y, z, cmap=plt.get_cmap("viridis"), norm=plt.Normalize(0.0, 1.0), ax=None
):
    if ax is None:
        ax = plt.gca()
    ax.scatter(x, y, c=z, cmap=cmap, norm=norm, alpha=0.3)


def plot_phase_portraits_colored_by_return(
    data,
    cumulative_returns,
    cmap=plt.get_cmap("viridis"),
    norm=plt.Normalize(0.0, 1.0),
    step=8,
    jump_threshold=np.pi,
    stop_threshold=1e-5,
    max_segment_length=100,
):
    dof_pos = data["dof_pos"]
    dof_vel = data["dof_vel"]

    fig, ax = plt.subplots(figsize=(10, 8))

    for i in range(0, dof_pos.shape[0], step):
        pos = dof_pos[i, :, 0]
        vel = dof_vel[i, :, 0]
        returns = cumulative_returns[i, :]
        # Normalize returns for colormap
        start_segment = 0
        for j in range(1, len(pos)):
            segment_length = j - start_segment
            # Check for jump, stop condition, or max segment length
            if (
                np.abs(pos[j] - pos[j - 1]) > jump_threshold
                or segment_length >= max_segment_length
            ):
                colorline(
                    pos[start_segment:j],
                    vel[start_segment:j],
                    returns[start_segment:j],
                    cmap=cmap,
                    norm=norm,
                    ax=ax,
                )
                start_segment = j
                if np.mean(np.abs(vel[j:])) < np.mean(np.abs(vel[j : j + 100])) < 1e-6:
                    plt.scatter(pos[j], vel[j], marker="x", s=50, color="red")
                    break

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax, label="Cumulative Return")
    # plt.colorbar(sm, label="Cumulative Return")
    plt.xlabel("dof_pos")
    plt.ylabel("dof_vel")
    plt.title("Phase Portraits Colored by Cumulative Return")
    plt.grid(True)


def plot_initial_conditions_colored_by_return(
    data,
    returns,
    cmap=plt.get_cmap("viridis"),
    norm=plt.Normalize(0.0, 1.0),
):
    dof_pos_unwrapped = np.unwrap(data["dof_pos"], axis=-1)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        dof_pos_unwrapped[:, 0, 0],
        data["dof_vel"][:, 0, 0],
        c=returns,
        cmap=cmap,
        norm=norm,
        alpha=0.7,
    )
    plt.colorbar(scatter, label="Total Return")
    plt.xlabel("dof_pos (unwrapped)")
    plt.ylabel("dof_vel")
    plt.title("Initial Conditions Colored by Total Return")
    plt.grid(True)


if __name__ == "__main__":
    experiment = "Feb23_14-33-51"
    file_path = f"{LEGGED_GYM_ROOT_DIR}/logs/lqrc/{experiment}/data.npz"
    data = np.load(file_path)

    rewards_data = {key: data[key] for key in data.files if key[:2] == "r_"}
    data = {key: data[key] for key in data.files if key[:2] != "r_"}

    save_path = f"{LEGGED_GYM_ROOT_DIR}/logs/lqrc/{experiment}"
    returns = calculate_cumulative_returns(rewards_data)

    # setting colormap outside, so that all returns-related plots use the same
    returns_norm = plt.Normalize(returns.min(), returns.max())
    returns_cmap = plt.get_cmap("viridis")

    plot_phase_portraits(data, step=13)
    plt.savefig(f"{save_path}/phase_portraits.png")
    plot_initial_and_final_conditions(data)
    plt.savefig(f"{save_path}/initial_and_final_conditions.png")
    plot_initial_conditions_colored_by_return(
        data, returns[:, 0], cmap=returns_cmap, norm=returns_norm
    )
    plt.savefig(f"{save_path}/initial_conditions_colored_by_return.png")
    plot_phase_portraits_colored_by_return(
        data, returns, cmap=returns_cmap, norm=returns_norm, step=13
    )
    plt.savefig(f"{save_path}/phase_portraits_colored_by_return.png")
