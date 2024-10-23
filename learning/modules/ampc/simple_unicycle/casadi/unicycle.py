#
#     MIT No Attribution
#
#     Copyright (C) 2010-2023 Joel Andersson, Joris Gillis, Moritz Diehl, KU Leuven.
#
#     Permission is hereby granted, free of charge, to any person obtaining a copy of this
#     software and associated documentation files (the "Software"), to deal in the Software
#     without restriction, including without limitation the rights to use, copy, modify,
#     merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
#     permit persons to whom the Software is furnished to do so.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#     INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
#     PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
#     HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
#     OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
#     SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

# Car race along a track
# ----------------------
# An optimal control problem (OCP),
# solved with direct multiple-shooting.
#
# For more information see: http://labs.casadi.org/OCP
from casadi import *
from utils import plot_robot, plot_3d_costs
import numpy as np

def solve_open_loop(x0, silent=False, plt_show=False):
    N = 30  # number of control intervals

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
    # state_cost = horzcat(*[mtimes(Q, X[:, i]) for i in range(X.shape[1])])
    # control_cost = horzcat(*[mtimes(R, U[:, i]) for i in range(U.shape[1])])
    # opti.minimize(sum1(sum2(state_cost)) + sum1(sum2(control_cost)))  # get to (0, 0)
    cost = 0
    for i in range(N):
        cost += X[:, i].T @ Q @ X[:, i]
        cost += U[:, i].T @ R @ U[:, i]
    # make state terminal cost especially high
    cost += X[:, -1].T@ Qf @X[:, -1]
    # set objective
    opti.minimize(cost)  # get to (0, 0)

    # ---- dynamic constraints --------
    f = lambda x, u: vertcat(u[0] * cos(u[1]), u[0] * sin(u[1]))  # dx/dt = f(x,u)

    dt = 0.2  # length of a control interval
    for k in range(N):  # constrain optimization to dynamics
        x_next = X[:, k] + dt * f(X[:, k], U[:, k])
        opti.subject_to(X[:, k + 1] == x_next)  # close the gaps

    # ---- path constraints -----------
    opti.subject_to(opti.bounded(-0.1, v, 0.1))
    opti.subject_to(opti.bounded(np.deg2rad(-50), theta, np.deg2rad(50)))
    opti.subject_to(opti.bounded(-0.5, X, 0.5))

    # terminal constr
    # opti.subject_to(x[N] == 0)
    # opti.subject_to(y[N] == 0)


    # ---- boundary conditions --------
    opti.subject_to(x[0] == x0[0])  # initial pos
    opti.subject_to(y[0] == x0[1])  # initial pos
    # opti.subject_to(v[0] == 0.0)  # start from standstill

    obstacle_positions = [[0, 0.5], [0.2,-0.2]]
    obstacle_radiuses = [0.3, 0.2]
    for (xobs,yobs), r in zip(obstacle_positions, obstacle_radiuses):
        for k in range(N+1):
            opti.subject_to((x[k]-xobs)**2+(y[k]-yobs)**2 >= r**2)

    # ---- initial values for solver ---
    opti.set_initial(x, np.linspace(x0[0], 0, N+1))
    opti.set_initial(y, np.linspace(x0[1], 0, N+1))
    # these can be guesses right?^

    # ---- solve NLP              ------
    if silent:
        opti.solver('ipopt', {
            'ipopt.print_level': 0,   # Suppress output from Ipopt
            'ipopt.sb': 'yes',        # Suppress information messages
            'print_time': 0           # Suppress timing information
        })
    else:
        opti.solver("ipopt")  # set numerical backend
    try:
        sol = opti.solve()  # actual solve

        print(f"cost: {sol.value(cost)}")

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
            plt_name=f"open_loop_x0=[{x0[0]:.2f},{x0[1]:.2f}].png"
        )

        return sol.value(U), sol.value(X), sol.value(cost)

    except RuntimeError as e:
        if "Infeasible" in str(e):
            return None, None, None


def evaluate_grid():
    Ngrid = 10
    x_values = np.linspace(-0.5, 0.5, Ngrid)
    y_values = np.linspace(-0.5, 0.5, Ngrid)
    
    x0s, Utrajs, Xtrajs, costs = [],[],[],[]

    X, Y = np.meshgrid(x_values, y_values)
    for i in range(Ngrid):
        for j in range(Ngrid):
            x0 = [X[i, j], Y[i, j]]
            Utraj, Xtraj, cost = solve_open_loop(x0, silent=True, plt_show=False)
            if cost is not None:
                x0s.append(x0)
                Utrajs.append(Utraj)
                Xtrajs.append(Xtraj)
                costs.append(cost)
                
    plot_3d_costs(x0s, costs)

if __name__=="__main__":
    # interesting initial conditions:
    # x0s = [[0.5, 0.5], [0.5, -0.5], [-0.5, 0.5], [-0.5, -0.5]]
    # for x0 in x0s:
    #     solve_open_loop(x0, silent=False, plt_show=True)
        
    evaluate_grid()