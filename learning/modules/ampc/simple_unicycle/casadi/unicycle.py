import os
from tqdm import tqdm
from casadi import *
from utils import plot_robot, plot_3d_costs, plot_3d_surface
import numpy as np
import tqdm
import pickle
from gym import LEGGED_GYM_ROOT_DIR

graph_path = (
    f"{LEGGED_GYM_ROOT_DIR}/learning/modules/ampc/simple_unicycle/casadi/graphs"
)
if not os.path.exists(graph_path):
    os.makedirs(graph_path)


def solve_open_loop(
    x0,
    soft_constraint_scale=1e3,
    soft_state_constr=False,
    silent=False,
    plt_show=False,
    plt_save=False,
):
    N = 30  # number of control intervals
    dt = 0.2  # length of a control interval

    opti = Opti()  # Optimization problem

    # ---- decision variables ---------
    X = opti.variable(2, N + 1)  # state trajectory
    x = X[0, :]
    y = X[1, :]
    U = opti.variable(2, N)  # control trajectory
    v = U[0, :]
    theta = U[1, :]

    # ---- objective          ---------
    Q = np.diag([1, 1])  # [x,y]
    R = np.diag([1e-1, 1e-1])
    Qf = 100.0 * Q
    cost = 0
    for i in range(N):
        cost += X[:, i].T @ Q @ X[:, i]
        cost += U[:, i].T @ R @ U[:, i]

    cost += X[:, -1].T @ Qf @ X[:, -1]

    # ---- dynamic constraints --------
    f = lambda x, u: vertcat(u[0] * cos(u[1]), u[0] * sin(u[1]))  # dx/dt = f(x,u)
    for k in range(N):  # constrain optimization to dynamics
        x_next = X[:, k] + dt * f(X[:, k], U[:, k])
        opti.subject_to(X[:, k + 1] == x_next)  # close the gaps

    opti.subject_to(opti.bounded(-0.1, v, 0.1))
    opti.subject_to(opti.bounded(np.deg2rad(-50), theta, np.deg2rad(50)))

    # obstacle and x-y bounds
    obstacle_positions = [[0, 0.5], [0.2, -0.2]]
    obstacle_radiuses = [0.3, 0.2]
    if not soft_state_constr:
        opti.subject_to(opti.bounded(-0.5, X, 0.5))
        for (xobs, yobs), r in zip(obstacle_positions, obstacle_radiuses):
            for k in range(N + 1):
                opti.subject_to((x[k] - xobs) ** 2 + (y[k] - yobs) ** 2 >= r**2)
    else:
        eps = opti.variable()
        opti.subject_to(eps >= 0)
        # opti.subject_to(opti.bounded(-0.5, X, 0.5))  # lower bound
        opti.subject_to(opti.bounded(-0.5, X + eps, inf))  # lower bound
        opti.subject_to(opti.bounded(-inf, X - eps, 0.5))  # upper bound
        for (xobs, yobs), r in zip(obstacle_positions, obstacle_radiuses):
            for k in range(N + 1):
                opti.subject_to((x[k] - xobs) ** 2 + (y[k] - yobs) ** 2 >= r**2 - eps)
        cost += soft_constraint_scale * eps**2 + soft_constraint_scale * eps

    # ---- boundary conditions --------
    opti.subject_to(x[0] == x0[0])  # initial pos
    opti.subject_to(y[0] == x0[1])  # initial pos

    # ---- initial values for solver ---
    opti.set_initial(x, np.linspace(x0[0], 0, N + 1))
    opti.set_initial(y, np.linspace(x0[1], 0, N + 1))

    # ---- solve NLP              ------
    opti.minimize(cost)  # get to (0, 0)
    if silent:
        opti.solver(
            "ipopt",
            {
                "ipopt.print_level": 0,  # Suppress output from Ipopt
                "ipopt.sb": "yes",  # Suppress information messages
                "print_time": 0,  # Suppress timing information
            },
        )
    else:
        opti.solver("ipopt")  # set numerical backend
    try:
        sol = opti.solve()  # actual solve

        # print(f"cost: {sol.value(cost)}")

        plt_name = None
        if plt_save:
            plt_name = f"open_loop_x0=[{x0[0]:.2f},{x0[1]:.2f}].png"

        # ---- post-processing        ------
        plot_robot(
            np.linspace(0, N, N + 1),
            [0.1, np.deg2rad(50)],
            sol.value(U).T,
            sol.value(X).T,
            x_labels=["x", "y"],
            u_labels=["v", "theta"],
            obst_pos=obstacle_positions,
            obst_rad=obstacle_radiuses,
            time_label="Sim steps",
            plt_show=plt_show,
            plt_name=plt_name,
        )

        return sol.value(U), sol.value(X), sol.value(cost)

    except RuntimeError as e:
        if "Infeasible" in str(e):
            return None, None, None


def evaluate_grid():
    Ngrid = 20
    x_values = np.linspace(-0.52, 0.52, Ngrid)
    y_values = np.linspace(-0.52, 0.52, Ngrid)

    x0s, Utrajs, Xtrajs, costs = [], [], [], []

    X, Y = np.meshgrid(x_values, y_values)
    with tqdm.tqdm(total=Ngrid * Ngrid) as pbar:
        for i in range(Ngrid):
            for j in range(Ngrid):
                x0 = [X[i, j], Y[i, j]]
                Utraj, Xtraj, cost = solve_open_loop(
                    x0,
                    soft_constraint_scale=1e3,
                    soft_state_constr=True,
                    silent=True,
                    plt_show=False,
                )
                if cost is not None:
                    x0s.append(x0)
                    Utrajs.append(Utraj)
                    Xtrajs.append(Xtraj)
                    costs.append(cost)
                pbar.set_postfix(
                    {"last_cost": cost if cost is not None else "N/A"}
                )  # Update with last cost
                pbar.update(1)  # Update the progress bar for each iteration

    # plot_3d_surface(x0s, costs, plt_show=True, plt_name="cost_grid.png")
    plot_3d_costs(x0s, costs, plt_show=True, plt_name="cost_grid.png")


def create_dataset(Ngrid, lb, ub):
    x_values = np.linspace(lb[0], ub[0], Ngrid)
    y_values = np.linspace(lb[1], ub[1], Ngrid)

    x0s, Utrajs, Xtrajs, costs = [], [], [], []

    X, Y = np.meshgrid(x_values, y_values)
    with tqdm.tqdm(total=Ngrid * Ngrid) as pbar:
        for i in range(Ngrid):
            for j in range(Ngrid):
                x0 = [X[i, j], Y[i, j]]
                Utraj, Xtraj, cost = solve_open_loop(
                    x0,
                    soft_constraint_scale=1e3,
                    soft_state_constr=True,
                    silent=True,
                    plt_show=False,
                )
                if cost is not None:
                    x0s.append(x0)
                    Utrajs.append(Utraj)
                    Xtrajs.append(Xtraj)
                    costs.append(cost)
                pbar.set_postfix(
                    {"last_cost": cost if cost is not None else "N/A"}
                )  # Update with last cost
                pbar.update(1)  # Update the progress bar for each iteration
    data_to_save = {"x0": x0s, "X": Xtrajs, "U": Utrajs, "J": None, "cost": costs}
    with open(
        f"{LEGGED_GYM_ROOT_DIR}/learning/modules/ampc/simple_unicycle/casadi/100_unicycle_dataset_soft_constraints.pkl",
        "wb",
    ) as f:
        pickle.dump(data_to_save, f)


if __name__ == "__main__":
    # interesting initial conditions:
    # x0s = [[0.5, 0.5], [0.5, -0.5], [-0.5, 0.5], [-0.5, -0.5]]
    # for x0 in x0s:
    #     solve_open_loop(x0, silent=False, plt_show=True, plt_save=True)

    # evaluate_grid()
    create_dataset(50, [-0.52, -0.52], [0.52, 0.52])
