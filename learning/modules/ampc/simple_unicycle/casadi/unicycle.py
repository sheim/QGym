import os
from tqdm import tqdm
from casadi import *
from utils import plot_robot, plot_3d_costs, plot_3d_surface
import numpy as np
import tqdm
import pickle
import fire
# from gym import LEGGED_GYM_ROOT_DIR


# graph_path = (
#     f"{LEGGED_GYM_ROOT_DIR}/learning/modules/ampc/simple_unicycle/casadi/graphs"
# )
# savepath = f"{LEGGED_GYM_ROOT_DIR}/learning/modules/ampc/simple_unicycle/casadi/100_unicycle_dataset.pkl"
# if not os.path.exists(graph_path):
#     os.makedirs(graph_path)


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
    
    x0_param = opti.parameter(2)

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
                opti.subject_to((x[k] - xobs) ** 2 + (y[k] - yobs) ** 2 >= (r-eps)**2)
        cost += soft_constraint_scale * eps**2 + soft_constraint_scale/10 * eps

    # ---- boundary conditions --------
    opti.subject_to(x[0]-x0_param[0] == 0)  # initial pos
    opti.subject_to(y[0]-x0_param[1] == 0)  # initial pos
    
    # min f(x), s.t. g_lb <= g(x,p) <= g_ub
    # 
    # L = f(x) + lam.T @ g(x,p)
    # 
    # diehl numerical optimization 11.22
    # df(x*(p),p)/dp = dL/dp(x*(p),lam*(p), p)
    # 
    # dL/dp = lam.T@ dg/dp(x,p)
    # df(x*(p),p)/dp = lam*.T @ dg/dp(x*(p),p)

    # ---- initial values for solver ---
    opti.set_initial(x, np.linspace(x0[0], 0, N + 1))
    opti.set_initial(y, np.linspace(x0[1], 0, N + 1))
    
    # set x0 parameter
    opti.set_value(x0_param, x0)

    # ---- solve NLP              ------
    opti.minimize(cost)  # get to (0, 0)
    if silent:
        opti.solver(
            "ipopt",
            {
                "ipopt.print_level": 0,  # Suppress output from Ipopt
                "ipopt.sb": "yes",  # Suppress information messages
                "print_time": 0,  # Suppress timing information
                # "ipopt.tol": 1e-12,
                # "ipopt.dual_inf_tol": 0.1
            },
        )
    else:
        opti.solver("ipopt")  # set numerical backend
    try:
        sol = opti.solve()  # actual solve

        plt_name = None
        if plt_save:
            plt_name = f"open_loop_x0=[{x0[0]:.2f},{x0[1]:.2f}].png"

        # ---- post-processing        ------
        if plt_show or plt_name:
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

        dg_dp_fcn = Function("constraint_parameter_gradient", [opti.x, opti.p], [jacobian(opti.g, opti.p)], ['primal', 'param'], ['dg_dp'])
        cost_gradient = sol.value(opti.lam_g).reshape(-1,1).T @ dg_dp_fcn(sol.value(opti.x), x0)
    
        U_res = np.array(sol.value(U)),
        X_res = np.array(sol.value(X)),
        cost_res = float(sol.value(cost))
        cost_gradient_res = np.array(cost_gradient).flatten()
        # print(f"{cost_gradient=}")
        
        return U_res, X_res, cost_res, cost_gradient_res

    except RuntimeError as e:
        if "Infeasible" in str(e):
            return None, None, None


def evaluate_grid(Ngrid=20, filename=None):
    filename = f"{filename}_{Ngrid**2}"
    # Ngrid = 20
    x_values = np.linspace(-0.55, 0.55, Ngrid)
    y_values = np.linspace(-0.55, 0.55, Ngrid)

    x0s, Utrajs, Xtrajs, costs, cost_gradients = [], [], [], [], []

    X, Y = np.meshgrid(x_values, y_values)
    with tqdm.tqdm(total=Ngrid * Ngrid) as pbar:
        for i in range(Ngrid):
            for j in range(Ngrid):
                x0 = [X[i, j], Y[i, j]]
                Utraj, Xtraj, cost, dcost_dx0 = solve_open_loop(
                    x0,
                    soft_constraint_scale=7e2,
                    soft_state_constr=True,
                    silent=True,
                    plt_show=False,
                )
                if cost is not None:
                    x0s.append(x0)
                    Utrajs.append(Utraj)
                    Xtrajs.append(Xtraj)
                    costs.append(cost)
                    cost_gradients.append(dcost_dx0)
                pbar.set_postfix(
                    {"last_cost": cost if cost is not None else "N/A"}
                )  # Update with last cost
                pbar.update(1)  # Update the progress bar for each iteration

    data_to_save = {"x0": x0s, "X": Xtrajs, "U": Utrajs, "cost": costs, "cost_gradient": cost_gradients}
    if filename is not None:
        with open(f"data/{filename}.pkl", "wb") as f:
            pickle.dump(data_to_save, f)
        plot_3d_costs(x0s, costs, plt_show=True, plt_name=f"plots/{filename}_cost_landscape.png", cost_gradient=cost_gradients)
    # plot_3d_surface(x0s, costs, plt_show=True, plt_name="cost_grid.png")
    else:
        plot_3d_costs(x0s, costs, plt_show=True, plt_name=None)


def evaluate_local_batch(Ntotal=5, filename=None, cost_threshold=50, resample_threshold=50, num_samples=64, sample_variance_scale=0.5):
    filename = f"{filename}_{Ntotal}"
    xy_min, xy_max = -0.52, 0.52
    # xmin, xmax = np.array([xy_min, xy_min]), np.array([xy_max, xy_max])

    # Covariance for sampling around each grid point
    covariance = sample_variance_scale**2*(xy_max - xy_min)/Ntotal**0.5
    cov_matrix = np.array([[covariance, 0], [0, covariance]])

    # Outer lists to store lists of lists
    all_x0s, all_Utrajs, all_Xtrajs, all_costs, all_cost_gradients = [], [], [], [], []
    
    valid_points_found = 0
    total_required_points = Ntotal

    with tqdm.tqdm(total=total_required_points) as pbar:
        while valid_points_found < total_required_points:
            outer_x0 = np.random.uniform(low=xy_min, high=xy_max, size=2)
            Utraj, Xtraj, cost, dcost_dx0 = solve_open_loop(
                outer_x0,
                soft_constraint_scale=1e3,
                soft_state_constr=True,
                silent=True,
                plt_show=False,
            )
            
            # Skip this grid point if the cost exceeds the threshold
            if cost is None or cost > cost_threshold:
                continue
            valid_points_found += 1

            # Initialize inner lists, starting with the grid point
            x0s, Utrajs, Xtrajs, costs, cost_gradients = [outer_x0], [Utraj], [Xtraj], [cost], [dcost_dx0]

            # Sample additional points around the grid point
            samples_added = 1
            while samples_added < num_samples:
                sampled_x0 = np.random.multivariate_normal(outer_x0, cov_matrix)
                Utraj, Xtraj, cost, dcost_dx0 = solve_open_loop(
                    sampled_x0,
                    soft_constraint_scale=1e3,
                    soft_state_constr=True,
                    silent=True,
                    plt_show=False,
                )
                
                # Only add if cost is within the acceptable range
                if cost is not None and cost <= resample_threshold:
                    x0s.append(sampled_x0)
                    Utrajs.append(Utraj)
                    Xtrajs.append(Xtraj)
                    costs.append(cost)
                    cost_gradients.append(dcost_dx0)
                    samples_added += 1

            # Append inner lists to the outer lists
            all_x0s.append(x0s)
            all_Utrajs.append(Utrajs)
            all_Xtrajs.append(Xtrajs)
            all_costs.append(costs)
            all_cost_gradients.append(cost_gradients)

            # Progress bar update
            pbar.set_postfix({"last_cost": cost})
            pbar.update(1)

    # Flatten lists for saving and plotting
    flat_x0s = [item for sublist in all_x0s for item in sublist]
    flat_Utrajs = [item for sublist in all_Utrajs for item in sublist]
    flat_Xtrajs = [item for sublist in all_Xtrajs for item in sublist]
    flat_costs = [item for sublist in all_costs for item in sublist]
    flat_cost_gradients = [item for sublist in all_cost_gradients for item in sublist]

    data_to_save = {
        "x0": flat_x0s,
        "X": flat_Xtrajs,
        "U": flat_Utrajs,
        "cost": flat_costs,
        "cost_gradient": flat_cost_gradients,
        "nested_x0": all_x0s,
        "nested_U": all_Utrajs,
        "nested_cost": all_costs,
        "nested_cost_gradient": all_cost_gradients,
    }
    
    if filename is not None:
        with open(f"data/{filename}.pkl", "wb") as f:
            pickle.dump(data_to_save, f)
        plot_3d_costs(all_x0s, all_costs, plt_show=True, plt_name=f"plots/{filename}_cost_landscape.png", cost_gradient=None)
    else:
        plot_3d_costs(all_x0s, all_costs, plt_show=True, plt_name=None)


def test():
    # interesting initial conditions:
    x0s = [[0.5, 0.5], [0.5, -0.5], [-0.5, 0.5], [-0.5, -0.5]]
    for x0 in x0s:
        solve_open_loop(x0, silent=False, plt_show=True, plt_save=True)

if __name__ == "__main__":
    fire.Fire({
        "evaluate_grid": evaluate_grid,
        "evaluate_local_batch": evaluate_local_batch,
        "test": test
    })
    # x0s = [[0.5, 0.5], [0.5, -0.5], [-0.5, 0.5], [-0.5, -0.5]]
    # for x0 in x0s:
    #     solve_open_loop(x0, silent=False, plt_show=True, plt_save=True)

    # evaluate_grid()
    # create_dataset(100, [-0.5, -0.5], [0.5, 0.5])
