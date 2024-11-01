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

obstacle_positions = [[0, 0.5], [0.25, -0.25]]
obstacle_radiuses = [0.25, 0.15]
# Q = np.diag([1e3, 1e3, 1e-4, 1e-3, 1e-3])  # [x,y,x_d,y_d,th,th_d]
# R = np.diag([0.5e-1, 0.5e-1])
Q = np.diag([2, 2, 1, 1e-3])  # [x,y,x_d,y_d,th,th_d]
R = np.diag([0.5e-1, 0.5])
# umin = np.array([-0.1, np.deg2rad(-50)])
# umax = -umin
amax = 1

dt = 0.1  # length of a control interval

def make_run_fcn(
    x0,
    soft_constraint_scale=1e2,
    silent=False,
    plt_show=False,
    plt_save=False,
):
    soft_state_constr=True,
    N = 30  # number of control intervals

    opti = Opti()  # Optimization problem

    # ---- decision variables ---------
    X = opti.variable(4, N + 1)  # state trajectory
    px = X[0, :]
    py = X[1, :]
    v = X[2, :]
    theta = X[3, :]
    U = opti.variable(2, N)  # control trajectory
    a = U[0, :]
    dtheta = U[1, :]
    
    x0_param = opti.parameter(4)

    # ---- objective          ---------
    # Q = np.diag([2, 2])  # [x,y]
    # R = np.diag([1e-1, 1e-1])
    Qf = 100.0 * Q
    cost = 0
    for i in range(N):
        cost += X[:, i].T @ Q @ X[:, i]
        cost += U[:, i].T @ R @ U[:, i]

    cost += X[:, -1].T @ Qf @ X[:, -1]

    # ---- dynamic constraints --------
    def f(x_,u_):
        # px_, py_, v_, theta_ = x_
        # a_, dtheta_ = u_
        return vertcat(
            x_[2]*cos(x_[3]),
            x_[2]*sin(x_[3]),
            u_[0],
            u_[1]
        )
        
    for k in range(N):  # constrain optimization to dynamics
        k1 = f(X[:,k],         U[:,k])
        k2 = f(X[:,k]+dt/2*k1, U[:,k])
        k3 = f(X[:,k]+dt/2*k2, U[:,k])
        k4 = f(X[:,k]+dt*k3,   U[:,k])
        x_next = X[:,k] + dt/6*(k1+2*k2+2*k3+k4) 
        opti.subject_to(X[:,k+1]==x_next) # close the gaps

    opti.subject_to(opti.bounded(-amax,  a,  amax))
    # opti.subject_to(opti.bounded(umin[1], dtheta, umax[1]))

    # obstacle and x-y bounds
    # if not soft_state_constr:
    #     opti.subject_to(opti.bounded(-0.5, X, 0.5))
    #     for (xobs, yobs), r in zip(obstacle_positions, obstacle_radiuses):
    #         for k in range(N + 1):
    #             opti.subject_to((px[k] - xobs) ** 2 + (py[k] - yobs) ** 2 >= r**2)
    # else:
    eps = opti.variable(2)
    opti.subject_to(eps >= 0)
    # opti.subject_to(opti.bounded(-0.5, X, 0.5))  # lower bound
    opti.subject_to(opti.bounded(-0.5, X[:2,:] + eps[0], inf))  # lower bound
    opti.subject_to(opti.bounded(-inf, X[:2,:] - eps[0], 0.5))  # upper bound
    opti.subject_to(opti.bounded(-0.3, X[2,:] + eps[1], inf))  # lower bound
    opti.subject_to(opti.bounded(-inf, X[2,:] - eps[1], 0.3))  # upper bound
    
    # opti.subject_to(opti.bounded(-0.3, X[2,:]+10*eps, inf))  # lower bound
    # opti.subject_to(opti.bounded(-inf, X[2,:]-10*eps, 0.3))  # lower bound
    inflation=0.01
    eps_obst = opti.variable(N+1)
    opti.subject_to(eps_obst >= 0)
    for (xobs, yobs), r in zip(obstacle_positions, obstacle_radiuses):
        for k in range(N + 1):
            opti.subject_to((px[k] - xobs) ** 2 + (py[k] - yobs) ** 2 >= (r+inflation-eps_obst[k])**2)
    for i in range(eps.shape[0]):
        cost += 50*soft_constraint_scale * eps[i]**2 + soft_constraint_scale * eps[i]
    for i in range(eps_obst.shape[0]):
        cost += 50*soft_constraint_scale * eps_obst[i]**2 + soft_constraint_scale * eps_obst[i]
    # ---- boundary conditions --------
    opti.subject_to(X[:,0]-x0_param == 0)  # initial pos
    # opti.subject_to(py[0]-x0_param[1] == 0)  # initial pos
    
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
    opti.set_initial(px, np.linspace(x0[0], 0, N + 1))
    opti.set_initial(py, np.linspace(x0[1], 0, N + 1))
    
    # set x0 parameter
    # opti.set_value(x0_param, x0)

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
        
    def run(x0):
        opti.set_value(x0_param, x0)
        try:
            sol = opti.solve()  # actual solve

            plt_name = None
            if plt_save:
                plt_name = f"plots/open_loop_ipopt_x0=[{x0[0]:.2f},{x0[1]:.2f}].pdf"

            # ---- post-processing        ------
            if plt_show or plt_name:
                plot_robot(
                    np.linspace(0, N, N + 1),
                    [amax, None],
                    sol.value(U).T,
                    sol.value(X).T,
                    x_labels=["x", "y", "v", "theta", "dtheta"],
                    u_labels=["a", "ddtheta"],
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
    return run


def evaluate_grid(Ngrid=20, filename=None):
    filename = f"{filename}_{Ngrid**2}"
    # Ngrid = 20
    x_values = np.linspace(-0.52, 0.52, Ngrid)
    y_values = np.linspace(-0.52, 0.52, Ngrid)

    x0s, Utrajs, Xtrajs, costs, cost_gradients = [], [], [], [], []

    X, Y = np.meshgrid(x_values, y_values)
    with tqdm.tqdm(total=Ngrid * Ngrid) as pbar:
        for i in range(Ngrid):
            for j in range(Ngrid):
                x0 = [X[i, j], Y[i, j]]
                Utraj, Xtraj, cost, dcost_dx0 = solve_open_loop(
                    x0,
                    soft_constraint_scale=1e2,
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


def solve_closed_loop(xinit):
    run_fcn = make_run_fcn(xinit,silent=True, plt_show=False, plt_save=False)
    Nsim = 50
    Usim = []
    Xsim = [xinit]
    costsim = []
    u = np.zeros(2)
    print(f"Simulating x0={xinit}")
    for i in tqdm.tqdm(range(Nsim)):
        # print(f"Sim: {i}")
        U_res, X_res, cost_res, cost_gradient_res = run_fcn(Xsim[i])
        # print()
        u = U_res[0].T[0]
        Usim.append(u)
        Xsim.append(X_res[0].T[1])
        costsim.append(cost_res)
    
    plot_robot(
            np.linspace(0, dt*Nsim, Nsim + 1),
            [amax, None,None],
            np.concatenate((np.array(Usim),np.array(costsim).reshape(-1, 1)), axis=1),
            np.array(Xsim),
            x_labels=["x", "y", "v", "theta", "dtheta"],
            u_labels=["a", "ddtheta","cost"],
            obst_pos=obstacle_positions,
            obst_rad=obstacle_radiuses,
            time_label="Sim steps",
            plt_show=True,
            plt_name=f"plots/closed_loop_ipopt_x0=[{xinit[0]:.2f},{xinit[1]:.2f}].pdf",
            x_max = [0.5,0.5,0.3,None]
        )

def test_open_loop():
    # interesting initial conditions:
    x0s = [[0.51, 0.51,0,0], [0.51, -0.51,0,0], [-0.51, 0.51,0,0], [-0.51, -0.51,0,0]]
    # x0s = [[0.5, 0.5,0,0], [0.5, -0.5,0,0], [-0.5, 0.5,0,0], [-0.5, -0.5,0,0]]
    for x0 in x0s:
        run_fcn = make_run_fcn(x0,silent=False, plt_show=True, plt_save=True)
        run_fcn(x0)

def test_closed_loop():
    # interesting initial conditions:
    x0s = [[0.25,-0.25-0.15+0.01,0,3.14/2], [0.52, 0.52,0,0], [0.52, -0.52,0,0], [-0.52, 0.52,0,0], [-0.52, -0.52,0,0]]
    # x0s = [[0.5, 0.5,0,0], [0.5, -0.5,0,0], [-0.5, 0.5,0,0], [-0.5, -0.5,0,0]]

    for x0 in x0s:
        solve_closed_loop(x0)

if __name__ == "__main__":
    # test()
    fire.Fire({
        "evaluate_grid": evaluate_grid,
        "evaluate_local_batch": evaluate_local_batch,
        "test_open_loop": test_open_loop,
        "test_closed_loop": test_closed_loop
    })
    # x0s = [[0.5, 0.5], [0.5, -0.5], [-0.5, 0.5], [-0.5, -0.5]]
    # for x0 in x0s:
    #     solve_open_loop(x0, silent=False, plt_show=True, plt_save=True)

    # evaluate_grid()
    # create_dataset(100, [-0.5, -0.5], [0.5, 0.5])
