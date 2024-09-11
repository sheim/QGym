import numpy as np
import os
import pickle
import json
import matplotlib.pyplot as plt
import scipy

import casadi as cs
from acados_template import (
    AcadosSim,
    AcadosSimSolver,
    AcadosSimBatchSolver,
    AcadosOcp,
    AcadosOcpSolver,
)
from learning.modules.ampc.wheelbot_model import export_wheelbot_ode_model

# system parameters
dt = 20e-3
N = 60
nx = 10
nu = 2
ntheta = 11

x_max = np.array([0.1, 0.05, 0.05, 0.05, 0.05, 0.05])
x_min = -x_max


def load_parameter_values(path_to_json=None):
    if path_to_json is None:
        path_to_json = os.path.join(os.path.dirname(__file__), "system_parameters.json")

    with open(path_to_json) as f:
        data = json.load(f)

    parameter_values = np.array(
        [
            data["m_WR"],
            data["m_B"],
            data["I_Wxz_Ryz"],
            data["I_Wy_Rx"],
            data["I_Bx"],
            data["I_By"],
            data["I_Bz"],
            data["r_W"],
            data["l_WB"],
            data["fric_magn"],
            data["fric_slope"],
        ]
    )
    return parameter_values


def load_dataset(max_size=None):
    infile = os.path.join(os.path.dirname(__file__), "dataset.pkl")
    with open(infile, "rb") as file:
        data = pickle.load(file)

        x0_raw = np.array(data["x0"][:max_size])
        u_raw = np.array(data["U"][:max_size])
        J_raw = np.array(data["J"][:max_size])
        cost_raw = np.array(data["cost"][:max_size])

    Nsamples = len(x0_raw)
    x0_dataset = x0_raw.reshape(Nsamples, nx)
    u_dataset = u_raw.reshape(Nsamples, nu)
    J_dataset = J_raw.reshape(Nsamples, nu, ntheta)

    return x0_dataset, u_dataset, J_dataset, cost_raw


class WheelbotBatchSimulation:
    def __init__(
        self,
        num_threads_in_batch_solve,
        batch_size,
        delta_parameter_values=np.zeros(11),
    ):
        self.batch_size = batch_size
        self.sim = AcadosSim()
        self.sim.model,_ = export_wheelbot_ode_model()
        self.sim.solver_options.T = dt
        self.sim.solver_options.integrator_type = "IRK"
        self.sim.solver_options.collocation_type = "GAUSS_RADAU_IIA"
        self.sim.solver_options.num_stages = 6
        self.sim.solver_options.num_steps = 3
        self.sim.solver_options.newton_iter = 10
        self.sim.solver_options.newton_tol = 1e-8
        self.sim.solver_options.num_threads_in_batch_solve = num_threads_in_batch_solve
        self.sim.parameter_values = load_parameter_values() + delta_parameter_values
        self.batch_integrator = AcadosSimBatchSolver(
            self.sim, batch_size, verbose=False
        )

    def run_batch(self, X_, U_):
        this_batch_size = min(U_.shape[0], self.batch_size)
        for n in range(this_batch_size):
            self.batch_integrator.sim_solvers[n].set("u", U_[n])
            self.batch_integrator.sim_solvers[n].set("x", X_[n])

        self.batch_integrator.solve()

        X_res = np.array(X_)
        for n in range(this_batch_size):
            X_res[n] = self.batch_integrator.sim_solvers[n].get("x")

        return X_res

    def run_batch_horizon(self, X_, U_):
        this_batch_size = min(U_.shape[0], self.batch_size)
        N = U_.shape[1]
        X_res = np.repeat(X_[:, np.newaxis, :], N + 1, axis=1)
        for k in range(N):
            for n in range(this_batch_size):
                self.batch_integrator.sim_solvers[n].set("u", U_[n, k])
                self.batch_integrator.sim_solvers[n].set("x", X_res[n, k])

            self.batch_integrator.solve()

            for n in range(this_batch_size):
                X_res[n, k + 1] = self.batch_integrator.sim_solvers[n].get("x")

        return X_res


class WheelbotSimulation:
    def __init__(self):
        self.sim = AcadosSim()
        self.sim.model,_ = export_wheelbot_ode_model()
        self.sim.solver_options.T = dt
        self.sim.solver_options.integrator_type = "IRK"
        self.sim.solver_options.collocation_type = "GAUSS_RADAU_IIA"
        self.sim.solver_options.num_stages = 6
        self.sim.solver_options.num_steps = 3
        self.sim.solver_options.newton_iter = 10
        self.sim.solver_options.newton_tol = 1e-8
        self.sim.parameter_values = load_parameter_values()
        self.integrator = AcadosSimSolver(self.sim)

    def run(self, X_, U_):
        self.integrator.set("u", U_)
        self.integrator.set("x", X_)
        self.solve()
        X_res = self.integrator.get("x")
        return X_res

    def run_horizon(self, X_, U_):
        N = U_.shape[1]
        X_res = np.repeat(X_[:, np.newaxis], N + 1, axis=1)

        print(f"{X_.shape=}")
        print(f"{U_.shape=}")

        for k in range(N):
            self.integrator.set("u", U_[:, k])
            self.integrator.set("x", X_res[:, k])
            self.integrator.solve()
            X_res[:, k + 1] = self.integrator.get("x")

        return X_res


default_subplot_labels_x = [
    "yaw [rad]",
    "roll [rad]",
    "pitch [rad]",
    "yaw_rate [rad/s]",
    "roll_rate [rad/s]",
    "pitch_rate [rad/s]",
    "driving angle [rad]",
    "reaction angle [rad]",
    "driving rate [rad/s]",
    "reaction rate [rad/s]",
]

default_subplot_labels_u = ["torque driving [Nm]", "torque reaction [Nm]"]


def plot_time_to_failure(data, filename):
    fig, axes = plt.subplots(
        nrows=len(list(data.keys())),
        ncols=1,
    )
    for ix, name in enumerate(data.keys()):
        ax = axes[ix] if isinstance(axes, list) else axes
        values = data[name]
        # ax.scatter(values[:, 0], values[:, 1], label="mean")
        # ax.scatter(values[:, 0], values[:, 2], label="std")
        ax.errorbar(values[:, 0], values[:, 1], yerr=values[:, 2], fmt="ro", capsize=2)
        # ax.legend(loc=1)
        ax.set_ylabel(f"{name} Index of Flatline")
        ax.set_xlabel("Epochs")
        ax.set_title("# of One-Step MPC Rollouts Before Flatline")
    plt.savefig(f"{filename}.pdf", format="pdf")
    print("Saving time to failure plot to", filename)


def plot_u_diff(data, filename):
    fig, axes = plt.subplots(
        nrows=len(list(data.keys())),
        ncols=1,
    )
    for ix, name in enumerate(data.keys()):
        ax = axes[ix] if isinstance(axes, list) else axes
        values = data[name]
        ax.scatter(values[:, 0], values[:, 1], label="U Error")
        ax.legend(loc=1)
        ax.set_ylabel(f"{name} Average Error to Optimal U (Nm)")
        ax.set_xlabel("Epochs")
        ax.set_title(
            "Error Between 1st Step MPC U and Optimal Dataset U (Avg Across Batch)"
        )
        ax.set_yscale("log")
    plt.savefig(f"{filename}.pdf", format="pdf")
    print("Saving U error plot to", filename)


def plot_wheelbot_all_inits(
    data_x,
    data_u,
    plot_labels,
    subplot_labels_x=default_subplot_labels_x,
    subplot_labels_u=default_subplot_labels_u,
    filename="wheelbot_plot",
    show=True,
):
    # Plot each column
    n_init_conditions = data_x.shape[0]
    nx = data_x.shape[1]
    nu = data_u.shape[1]

    fig_width = 7.2  # inches
    fig_height = fig_width * (16 / 9)  # maintain 9:16 aspect ratio

    fig, axs = plt.subplots(
        nrows=nx + nu, ncols=1, figsize=(1.5 * fig_width, 1.5 * fig_height)
    )

    for ix, ax in enumerate(axs):
        if ix == 0:  # yaw
            ax.set_ylim(-np.pi, np.pi)
        if ix in [1, 2]:  # roll, pitch
            ax.set_ylim(-np.pi / 6.0, np.pi / 6.0)
        if ix in [3, 4, 5]:  # yaw_rate, roll_rate, pitch_rate
            # ax.set_ylim(-1.0 / 20.0, 1.0 / 20.0)
            ax.set_ylim(-100.0, 100.0)
        if ix in [6, 7]:  # driving angle, reaction angle
            ax.set_ylim(-1000.0, 1000.0)
        if ix == 8:  # driving rate
            ax.set_ylim(-200.0, 200.0)
        if ix == 9:  # reaction rate
            ax.set_ylim(-1000.0, 1000.0)

    # x0 = list(data_x[0][:, 0])
    # x0_formatted_floats = (
    #     "[" + ", ".join(["{:.2f}".format(number) for number in x0]) + "]"
    # )

    fig.suptitle("Wheelbot w/ all initial conditions")
    for ic in range(n_init_conditions):
        dat_x = data_x[ic]
        dat_u = data_u[ic]
        for i in range(nx):
            ax = axs[i]

            ax.plot(dat_x[i, :], label=f"{plot_labels[0]}_{ic}", linestyle="dashed")
            if ic == n_init_conditions - 1:
                ax.set_ylabel(f"{subplot_labels_x[i]}")
                ax.grid()
                ax.legend(loc=1)

        for i in range(nu):
            ax = axs[i + nx]

            ax.step(
                range(len(dat_u[i, :])),
                np.append(dat_u[i, 0], dat_u[i, :-1]),
                label=f"{plot_labels[0]}_{ic}",
                linestyle="dashed",
            )

            if ic == n_init_conditions - 1:
                ax.set_ylabel(f"{subplot_labels_u[i]}")
                ax.grid()
                ax.legend(loc=1)

    plt.xlabel("t [ms]")
    plt.tight_layout(pad=0.0, w_pad=0.75, h_pad=0.0)
    plt.subplots_adjust(
        left=0.1, right=0.97, bottom=0.04, top=0.95, wspace=0.2, hspace=0.35
    )

    if filename is not None:
        plt.savefig(f"{filename}.pdf", format="pdf")
    if show:
        plt.show()

    plt.close()


def plot_wheelbot(
    data_x,
    data_u,
    plot_labels,
    subplot_labels_x=default_subplot_labels_x,
    subplot_labels_u=default_subplot_labels_u,
    filename="wheelbot_plot",
    show=True,
):
    # Plot each column
    nx = data_x[0].shape[0]
    nu = data_u[0].shape[0]

    fig_width = 7.2  # inches
    fig_height = fig_width * (16 / 9)  # maintain 9:16 aspect ratio

    fig, axs = plt.subplots(
        nrows=nx + nu, ncols=1, figsize=(1.5 * fig_width, 1.5 * fig_height)
    )

    x0 = list(data_x[0][:, 0])
    x0_formatted_floats = (
        "[" + ", ".join(["{:.2f}".format(number) for number in x0]) + "]"
    )

    fig.suptitle(f"Wheelbot x0={x0_formatted_floats}")
    for i in range(nx):
        ax = axs[i]

        for idx, dat in enumerate(data_x):
            ax.plot(dat[i, :], label=f"{plot_labels[idx]}", linestyle="dashed")

        ax.set_ylabel(f"{subplot_labels_x[i]}")
        ax.grid()
        ax.legend(loc=1)

    for i in range(nu):
        ax = axs[i + nx]

        for idx, dat in enumerate(data_u):
            ax.step(
                range(len(dat[i, :])),
                np.append(dat[i, 0], dat[i, :-1]),
                label=f"{plot_labels[idx]}",
                linestyle="dashed",
            )

        ax.set_ylabel(f"{subplot_labels_u[i]}")
        ax.grid()
        ax.legend(loc=1)

    plt.xlabel("t [ms]")
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
    plt.subplots_adjust(
        left=0.06, right=0.97, bottom=0.04, top=0.95, wspace=0.2, hspace=0.35
    )

    if filename is not None:
        plt.savefig(f"{filename}.pdf", format="pdf")
    if show:
        plt.show()

    plt.close()


class WheelbotOneStepMPC:
    def __init__(self):
        # create ocp object to formulate the OCP
        ocp = AcadosOcp()

        # set model
        model, constraint= export_wheelbot_ode_model()
        ocp.model = model

        nx = model.x.rows()
        nu = model.u.rows()
        ny = nx + nu
        ny_e = nx

        ocp.dims.N = 1

        # set cost module
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"

        q_vec = [100, 1, 1, 1e-3, 1e-2, 1, 1, 1e-4, 0.25, 1e-3]
        r_vec = [10, 1e-2]

        Q_mat = 2 * np.diag(q_vec)
        R_mat = 2 * np.diag(r_vec)

        # A = np.random.rand(10,10)

        # print(Q_mat.shape)

        ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
        ocp.cost.W_e = 10 * Q_mat  # doesn't matter, we'll set this later ;-)

        ocp.model.cost_y_expr = cs.vertcat(model.x, model.u)
        ocp.model.cost_y_expr_e = model.x
        ocp.cost.yref = np.zeros((ny,))
        ocp.cost.yref_e = np.zeros((ny_e,))

        # set constraints
        ocp.constraints.con_h_expr = constraint.expr
        ocp.constraints.lh = np.array([-0.5, -0.5])
        ocp.constraints.uh = np.array([ 0.5,  0.5])

        ocp.constraints.x0 = np.zeros(10)
        ocp.constraints.idxbu = np.array([0, 1])

        ocp.solver_options.qp_solver = (
            "PARTIAL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
        )
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "IRK"
        ocp.solver_options.sim_method_newton_iter = 10

        # ocp.solver_options.print_level = 0
        ocp.solver_options.print_level = 0

        # if RTI:
        #     ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        # else:
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.globalization = (
            "MERIT_BACKTRACKING"  # turns on globalization
        )
        ocp.solver_options.nlp_solver_max_iter = 10

        # ocp.solver_options.qp_solver_cond_N = 1

        # set prediction horizon
        ocp.solver_options.tf = 20e-3
        ocp.parameter_values = load_parameter_values()

        self.solver_json = "acados_ocp_" + model.name + ".json"
        self.ocp = ocp
        self.acados_ocp_solver = AcadosOcpSolver(self.ocp, json_file=self.solver_json)
        # create an integrator with the same settings as used in the OCP solver.
        # acados_integrator = AcadosSimSolver(ocp, json_file = solver_json)

    def reset(self):
        self.acados_ocp_solver = AcadosOcpSolver(
            self.ocp, json_file=self.solver_json, generate=False, build=False
        )

    def run(self, x0, A):
        # print(self.acados_ocp_solver.get_option)
        self.acados_ocp_solver.cost_set(1, "W", A, api="old")
        self.acados_ocp_solver.set(0, "lbx", x0)
        self.acados_ocp_solver.set(0, "ubx", x0)
        self.acados_ocp_solver.solve()
        u = self.acados_ocp_solver.get(0, "u")
        xnext = self.acados_ocp_solver.get(1, "x")
        status = self.acados_ocp_solver.get_status()
        return u, xnext, status


if __name__ == "__main__":
    load_dataset()
