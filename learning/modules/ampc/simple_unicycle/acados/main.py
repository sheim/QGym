from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from robot_model import export_robot_model
import numpy as np
import scipy.linalg
from utils import plot_robot
import casadi as cs


# X0 = np.array([0.1, 0.0])  # Intital state
X0 = np.array([0.1, 0.2])  # Intital state
x_max = 0.5
T_horizon = 4.0  # Define the prediction horizon


def create_ocp_solver_description() -> AcadosOcp:
    N_horizon = 20  # Define the number of discretization steps

    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    model = export_robot_model()
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()

    # set dimensions
    ocp.solver_options.N_horizon = N_horizon

    # set cost
    Q_mat = 2 * np.diag([1e-1, 1e-1])  # [x,y]
    R_mat = 2 * np.diag([1e-1, 1e-2])

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    ny = nx + nu
    ny_e = nx

    ocp.cost.W_e = 100 * Q_mat
    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)

    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, 0:nu] = np.eye(nu)
    ocp.cost.Vu = Vu

    ocp.cost.Vx_e = np.eye(nx)

    # ocp.model.cost_y_expr = cs.vertcat(model.x, model.u)
    # ocp.model.cost_y_expr_e = model.x
    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    # set system constraints
    ocp.constraints.lbx = np.array([-x_max, -x_max])  # lower bounds on x, y
    ocp.constraints.ubx = np.array([x_max, x_max])  # upper bounds on x, y
    ocp.constraints.lbu = np.array([-0.1, -0.35])  # lower bounds on v, theta
    ocp.constraints.ubu = np.array([0.1, 0.35])  # upper bounds on v, theta
    # ocp.constraints.lbu = np.array([-0.1, -1.5])  # lower bounds on v, theta
    # ocp.constraints.ubu = np.array([0.1, 1.5])  # upper bounds on v, theta
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.idxbx = np.array([0, 1])

    # TODO: set obstacle constraint

    ocp.constraints.x0 = X0  # initial condition constraint

    # set options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_DAQP"  # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.sim_method_newton_iter = 4
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.nlp_solver_max_iter = 1000
    ocp.solver_options.globalization = "MERIT_BACKTRACKING"

    # set prediction horizon
    ocp.solver_options.tf = T_horizon

    return ocp


def closed_loop_simulation():
    # create solvers
    ocp = create_ocp_solver_description()
    model = ocp.model
    acados_ocp_solver = AcadosOcpSolver(ocp)
    acados_integrator = AcadosSimSolver(ocp)

    N_horizon = acados_ocp_solver.N

    # prepare simulation
    Nsim = 100
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()

    simX = np.zeros((Nsim + 1, nx))
    simU = np.zeros((Nsim, nu))

    xcurrent = X0
    simX[0, :] = xcurrent

    yref = np.array([0, 0, 0, 0])
    yref_N = np.array([0, 0])

    # initialize solver
    for stage in range(N_horizon + 1):
        acados_ocp_solver.set(stage, "x", np.array([0.1, 0.1]))
    for stage in range(N_horizon):
        acados_ocp_solver.set(stage, "u", np.zeros((nu,)))

    # closed loop
    for i in range(Nsim):
        # update yref
        # TODO: is this necessary?
        for j in range(N_horizon):
            acados_ocp_solver.set(j, "yref", yref)
        acados_ocp_solver.set(N_horizon, "yref", yref_N)

        # solve ocp
        # if i != 0:
        simU[i, :] = acados_ocp_solver.solve_for_x0(xcurrent)
        print("i", i, "sim U[i, :]", simU[i, :])
        status = acados_ocp_solver.get_status()

        if status not in [0, 2]:
            acados_ocp_solver.print_statistics()
            plot_robot(
                np.linspace(0, T_horizon / N_horizon * i, i + 1),
                [0.1, 0.35],
                simU[:i, :],
                simX[: i + 1, :],
            )
            raise Exception(
                f"acados acados_ocp_solver returned status {status} in closed loop instance {i} with {xcurrent}"
            )

        # simulate system
        xcurrent = acados_integrator.simulate(xcurrent, simU[i, :])
        simX[i + 1, :] = xcurrent

    # plot results
    plot_robot(
        np.linspace(0, T_horizon / N_horizon * Nsim, Nsim + 1),
        [0.1, 0.35],
        simU,
        simX,
        x_labels=model.x_labels,
        u_labels=model.u_labels,
        time_label=model.t_label,
    )


if __name__ == "__main__":
    closed_loop_simulation()
