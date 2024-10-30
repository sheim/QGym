import os
import matplotlib.pyplot as plt
import numpy as np


import matplotlib.pyplot as plt
import numpy as np

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
    plt_name=None
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
    gs = fig.add_gridspec(nx + nu, 2, width_ratios=[3, 1], wspace=0.3)
    
    t = shooting_nodes
    for i in range(nu):
        ax_u = fig.add_subplot(gs[i, 0])
        (line,) = ax_u.step(t, np.append([U[0, i]], U[:, i]))

        ax_u.set_ylabel(u_labels[i])
        ax_u.set_xlabel(time_label)
        if u_max[i] is not None:
            ax_u.hlines(u_max[i], t[0], t[-1], linestyles="dashed", alpha=0.7)
            ax_u.hlines(-u_max[i], t[0], t[-1], linestyles="dashed", alpha=0.7)
            ax_u.set_ylim([-1.2 * u_max[i], 1.2 * u_max[i]])
        ax_u.grid()

    for i in range(nx):
        ax_x = fig.add_subplot(gs[i + nu, 0])
        (line,) = ax_x.plot(t, X_traj[:, i])

        ax_x.set_ylabel(x_labels[i])
        ax_x.set_xlabel(time_label)
        ax_x.set_ylim(-0.5, 0.5)
        ax_x.grid()

    # New plot for X_traj[:, 0] vs X_traj[:, 1] with equal axis scaling
    ax_xy = fig.add_subplot(gs[:, 1])  # Span the entire right column
    ax_xy.plot(X_traj[:, 0], X_traj[:, 1], label="Trajectory")
    ax_xy.set_xlabel('x')
    ax_xy.set_ylabel('y')
    ax_xy.set_xlim([-0.5, 0.5])
    ax_xy.set_ylim([-0.5, 0.5])
    ax_xy.set_aspect('equal')  # Ensure equal scaling of axes
    ax_xy.grid()

    # Plot obstacles as gray circles
    if obst_pos is not None and obst_rad is not None:
        for (x, y), r in zip(obst_pos, obst_rad):
            circle = plt.Circle((x, y), r, color='gray', alpha=0.5)
            ax_xy.add_patch(circle)

    plt.legend()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)

    if plt_name:
        plt.savefig(f"{plt_name}")
        
    if plt_show:
        plt.show()
        
    plt.close()
    
    
    
# def plot_3d_costs(xy_coords, costs, plt_show=True, plt_name=None):
#     # Unpack x and y coordinates
#     x_values = [coord[0] for coord in xy_coords]
#     y_values = [coord[1] for coord in xy_coords]

#     # Create a 3D plot
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Plot the points
#     # ax_clip becomes available in matplotlib 3.10
#     # ax.scatter(x_values, y_values, costs, c=costs, cmap='viridis', marker='o', axlim_clip=True)
#     ax.scatter(x_values, y_values, costs, c=costs, cmap='viridis', marker='o')

#     # Set axis labels
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Cost')

#     # Set aspect ratio of the x and y axes to be equal
#     ax.set_box_aspect([1, 1, 0.75])  # x, y, z aspect ratio
#     ax.set_zlim(0, 20)

#     if plt_name:
#         plt.savefig(f"{plt_name}")
        
#     if plt_show:
#         plt.show()
        
#     plt.close()
    
    
def plot_3d_costs(
    xy_coords, 
    costs, 
    xy_eval=None, 
    cost_true=None, 
    cost_eval=None, 
    P=None, 
    offset=[0, 0], 
    linLatW=None,
    linLatb=None,
    plt_show=True, 
    plt_name=None,
    zlim=20,
    cost_gradient=None  # New parameter for cost gradient
):
    # Unpack x and y coordinates
    x_values = [coord[0] for coord in xy_coords]
    y_values = [coord[1] for coord in xy_coords]

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of costs
    ax.scatter(x_values, y_values, costs, c=costs, cmap='viridis', marker='o')

    # Plot cost_true and cost_eval as bold blue and red crosses at xy_eval
    if xy_eval is not None and cost_true is not None:
        ax.scatter(
            xy_eval[0], xy_eval[1], cost_true, color='blue', marker='X', s=100, label='Cost True'
        )
    if xy_eval is not None and cost_eval is not None:
        ax.scatter(
            xy_eval[0], xy_eval[1], cost_eval, color='red', marker='X', s=100, label='Cost Eval'
        )

    # Add a legend if cost_true or cost_eval are plotted
    if cost_true is not None or cost_eval is not None:
        ax.legend()

    # Plot the surface if P is provided
    if P is not None:
        x_range = np.linspace(min(x_values), max(x_values), 50)
        y_range = np.linspace(min(y_values), max(y_values), 50)
        X, Y = np.meshgrid(x_range, y_range)
        
        if linLatW is not None and linLatb is not None:
            # Compute Z based on the quadratic surface equation
            Z = np.array([
                [((np.array([xi, yi]) - offset)@linLatW.T+linLatb) @ P @ ((np.array([xi, yi]) - offset)@linLatW.T+linLatb) for xi in x_range]
                for yi in y_range
            ])
        else:
            # Compute Z based on the quadratic surface equation
            Z = np.array([
                [(np.array([xi, yi]) - offset) @ P @ (np.array([xi, yi]) - offset) for xi in x_range]
                for yi in y_range
            ])

        # Plot the surface
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, edgecolor='none')

    # Plot cost gradients as arrows if provided
    if cost_gradient is not None:
        cost_gradient_plotting = [(-grad[0]/np.linalg.norm(grad), -grad[1]/np.linalg.norm(grad), -np.linalg.norm(grad)) for grad in cost_gradient]
        
        cost_gradient_plotting_normalized = [
            (
                grad[0]/np.linalg.norm(grad),
                grad[1]/np.linalg.norm(grad),
                grad[2]/np.linalg.norm(grad)
                ) for grad in cost_gradient_plotting]
        
        gradient_x, gradient_y, gradient_z = zip(*cost_gradient_plotting)
        
        ax.quiver(
            x_values, y_values, costs,   # Starting points (x, y, z)
            gradient_x, gradient_y, gradient_z,  # Gradient directions (dx, dy, dz)
            length=0.1, color='blue', 
            pivot='middle',
            arrow_length_ratio=0, alpha=0.5,
            # normalize=True
        )
        
        
    # Set axis labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Cost')
    ax.set_box_aspect([1, 1, 0.75])  # x, y, z aspect ratio
    ax.set_zlim(0, zlim)

    # Save or show the plot
    if plt_name:
        plt.savefig(f"{plt_name}")
    if plt_show:
        plt.show()

    plt.close()
    
def plot_3d_surface(xy_coords, costs, zlim=(0, 20), plt_show=True, plt_name=None):
    # Unpack x and y coordinates
    x_values = [coord[0] for coord in xy_coords]
    y_values = [coord[1] for coord in xy_coords]

    # Convert to numpy arrays for easier reshaping
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    costs = np.array(costs)

    # Reshape the data assuming it's a grid (N x N)
    N = int(np.sqrt(len(xy_coords)))  # Assuming a perfect square number of points
    X = x_values.reshape(N, N)
    Y = y_values.reshape(N, N)
    Z = costs.reshape(N, N)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a surface plot
    # ax_clip becomes available in matplotlib 3.10
    # surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', axlim_clip=True)
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Cost')

    # Set aspect ratio of the x and y axes to be equal
    ax.set_box_aspect([1, 1, 0.75])  # x, y, z aspect ratio

    # Set limits for the z-axis (cost)
    ax.set_zlim(zlim)

    # Add a color bar which maps values to colors
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    if plt_name:
        plt.savefig(f"{plt_name}")
        
    if plt_show:
        plt.show()
        
    plt.close()
    
def plot_costs_histogramm(costs, plt_show=True, plt_name=None):
    plt.figure()
    plt.hist(costs, bins=50)
    plt.xlabel("Cost")
    plt.ylabel("Frequency")
    plt.title(f"Cost Distribution")
    
    # Save the plot
    if plt_name:
        plt.savefig(f"{plt_name}")
        
    if plt_show:
        plt.show()
        
    plt.close()


