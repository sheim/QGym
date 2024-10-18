from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from robot_model import export_robot_model
import numpy as np
import scipy.linalg
from utils import plot_robot
import casadi as cs

X0 = np.array([0.0, 0.0])  # Intital state
u_max = 0.5
T_horizon = 6.0  # Define the prediction horizon


def create_ocp_solver_description() -> AcadosOcp:
    N_horizon = 30  # Define the number of discretization steps

    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    model = export_robot_model()
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()

    # set dimensions
    ocp.solver_options.N_horizon = N_horizon

    # set cost
    Q_mat = 2 * np.diag([1e3, 1e3])  # [x,y]
    R_mat = 2 * 5 * np.diag([1e-1, 1e-2])

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    ny = nx + nu
    ny_e = nx

    ocp.cost.W_e = Q_mat
    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)

    ocp.model.cost_y_expr = cs.vertcat(model.x, model.u)
    ocp.model.cost_y_expr_e = model.x
    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    # set system constraints
    ocp.constraints.lbx = np.array([-u_max, -u_max])  # lower bounds on x, y
    ocp.constraints.ubx = np.array([u_max, u_max])  # upper bounds on x, y
    ocp.constraints.lbu = np.array([-0.1, -0.1])  # lower bounds on u_s, u_w
    ocp.constraints.ubu = np.array([0.1, 0.1])  # upper bounds on u_s, u_w

    # set obstacle constraint

    ocp.constraints.x0 = X0  # initial condition constraint

    # set options
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP"

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

    # yref = np.array([1, 1, 0, 0, 0, 0, 0])
    # yref_N = np.array([1, 1, 0, 0, 0])

    # initialize solver
    for stage in range(N_horizon + 1):
        acados_ocp_solver.set(stage, "x", 0.0 * np.ones(xcurrent.shape))
    for stage in range(N_horizon):
        acados_ocp_solver.set(stage, "u", np.zeros((nu,)))

    # closed loop
    for i in range(Nsim):
        # # update yref
        # for j in range(N_horizon):
        #     acados_ocp_solver.set(j, "yref", yref)
        # acados_ocp_solver.set(N_horizon, "yref", yref_N)

        # solve ocp
        simU[i, :] = acados_ocp_solver.solve_for_x0(xcurrent)
        status = acados_ocp_solver.get_status()

        if status not in [0, 2]:
            acados_ocp_solver.print_statistics()
            plot_robot(
                np.linspace(0, T_horizon / N_horizon * i, i + 1),
                u_max,
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
        [u_max, None],
        simU,
        simX,
        x_labels=model.x_labels,
        u_labels=model.u_labels,
        time_label=model.t_label,
    )


if __name__ == "__main__":
    closed_loop_simulation()
