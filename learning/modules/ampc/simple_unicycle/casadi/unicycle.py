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
from utils import plot_robot
import numpy as np

N = 20  # number of control intervals

opti = Opti()  # Optimization problem

# ---- decision variables ---------
X = opti.variable(2, N + 1)  # state trajectory
x = X[0, :]
y = X[1, :]
U = opti.variable(2, N)  # control trajectory
v = U[0, :]
theta = U[1, :]

# ---- objective          ---------
Q = 2 * np.diag([1e-1, 1e-1])  # [x,y]
R = 2 * np.diag([1e-1, 1e-2])
# state_cost = horzcat(*[mtimes(Q, X[:, i]) for i in range(X.shape[1])])
# control_cost = horzcat(*[mtimes(R, U[:, i]) for i in range(U.shape[1])])
# opti.minimize(sum1(sum2(state_cost)) + sum1(sum2(control_cost)))  # get to (0, 0)
state_cost = horzcat(
    *[mtimes(mtimes(X[:, i].T, Q), X[:, i]) for i in range(X.shape[1] - 1)]
)
control_cost = horzcat(
    *[mtimes(mtimes(U[:, i].T, R), U[:, i]) for i in range(U.shape[1])]
)
# make state terminal cost especially high
state_cost = horzcat(state_cost, mtimes(mtimes(X[:, -1].T, 100.0 * Q), X[:, -1]))
# set objective
opti.minimize(sum1(sum2(state_cost)) + sum1(sum2(control_cost)))  # get to (0, 0)
# ---- dynamic constraints --------
f = lambda x, u: vertcat(u[0] * cos(u[1]), u[0] * sin(u[1]))  # dx/dt = f(x,u)

dt = 0.2  # length of a control interval
for k in range(N):  # constrain optimization to dynamics
    x_next = X[:, k] + dt * f(X[:, k], U[:, k])
    opti.subject_to(X[:, k + 1] == x_next)  # close the gaps

# ---- path constraints -----------
opti.subject_to(opti.bounded(-0.1, v, 0.1))
opti.subject_to(opti.bounded(-0.35, theta, 0.35))
opti.subject_to(opti.bounded(-0.5, X, 0.5))

# ---- boundary conditions --------
opti.subject_to(x[0] == 0.15)  # initial pos
opti.subject_to(y[0] == 0.1)  # initial pos
opti.subject_to(v[0] == 0.0)  # start from standstill

# ---- initial values for solver ---
opti.set_initial(x, 0.1)
opti.set_initial(y, 0.1)
# these can be guesses right?^

# ---- solve NLP              ------
opti.solver("ipopt")  # set numerical backend
sol = opti.solve()  # actual solve

# ---- post-processing        ------
plot_robot(
    np.linspace(0, N, N + 1),
    [0.1, 0.35],
    sol.value(U).T,
    sol.value(X).T,
    x_labels=["x", "y"],
    u_labels=["v", "theta"],
    time_label="Sim steps",
)
