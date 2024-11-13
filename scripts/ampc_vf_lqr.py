import os
import pickle
import torch
import numpy as np
import math
import casadi as cs
from utils import (
    DEVICE,
)
from learning.modules.lqrc.plotting import plot_robot
from learning.modules.ampc.simple_unicycle.casadi.unicycle4d import (
    umin,
    umax,
    R,
    dt,
    obstacle_positions,
    obstacle_radiuses,
    f,
)
from gym import LEGGED_GYM_ROOT_DIR
from critic_params_ampc import critic_params
from learning.modules.lqrc import *  # noqa F401
from learning.modules.lqrc.utils import get_latent_matrix


model_path = os.path.join(LEGGED_GYM_ROOT_DIR, "models")
lqr_path = os.path.join(model_path, "lqr")
if not os.path.exists(lqr_path):
    os.makedirs(lqr_path)


def zoh_lin(T):
    nx = 4
    nu = 2
    x = cs.SX.sym("x", nx)
    u = cs.SX.sym("u", nu)
    Ac = cs.jacobian(f(x, u), x)
    Bc = cs.jacobian(f(x, u), u)
    Ad = cs.SX.eye(nx)
    Bd = cs.SX.zeros(nx, nu)
    for i in range(4):
        Toverifac = T ** (i + 1) / math.factorial(i + 1)
        Ad += Toverifac * cs.mpower(Ac, i + 1)
        Bd += Toverifac * cs.mpower(Ac, i) @ Bc
    zoh_lin_fcn = cs.Function("zoh_lin", [x], [Ad, Bd], ["x"], ["Ad", "Bd"])
    return zoh_lin_fcn


def rk4_nonlin(T):
    nx = 4
    nu = 2
    x = cs.SX.sym("x", nx)
    u = cs.SX.sym("u", nu)
    k1 = f(x, u)
    k2 = f(x + T / 2 * k1, u)
    k3 = f(x + T / 2 * k2, u)
    k4 = f(x + T * k3, u)
    x_next = x + T / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    rk4_nl_fcn = cs.Function("rk4_nonlin", [x, u], [x_next], ["x", "u"], ["x_next"])
    return rk4_nl_fcn


zoh_lin_fcn = zoh_lin(dt)
f_fcn = rk4_nonlin(dt)


@torch.no_grad()
def eval_lqr(model, xinit, model_name, W_latent=None, b_latent=None):
    with open(f"{model_path}/normalization.pkl", "rb") as f:
        data = pickle.load(f)
        V_max = data["cost_max"]
        V_min = data["cost_min"]
        X_max = data["x0_max"]
        X_min = data["x0_min"]

    model.load_state_dict(torch.load(f"{model_path}/{type(model).__name__}.pth"))
    model.eval()

    def eval_model(x, model_name):
        x_normalized = 2 * (x - X_min) / (X_max - X_min) - 1
        res = model(
            torch.tensor(x_normalized, dtype=torch.float32, device=DEVICE),
            return_all=True,
        )
        A_normalized_ = res["A"]
        x_off_normalized_ = res["x_offsets"]
        A_normalized = A_normalized_.cpu().detach().numpy()
        x_off_normalized = x_off_normalized_.cpu().detach().numpy()
        Dx = np.diag(2 / (X_max - X_min))
        Dxinv = np.diag((X_max - X_min) / 2)
        Dvinv = V_max - V_min
        Ptilde = Dvinv * Dx.T @ A_normalized @ Dx
        xtilde = x - Dxinv @ x_off_normalized

        # gridding
        method = "lqr"
        if method == "gridding":
            u0_vals = np.linspace(umin[0], umax[0], 20)
            u1_vals = np.linspace(umin[1], umax[1], 20)
            costs = np.zeros((20, 20))
            for i, u0 in enumerate(u0_vals):
                for j, u1 in enumerate(u1_vals):
                    u = np.array([u0, u1])

                    # DT = 0.5

                    # Compute x_next based on the dynamics
                    x_next = np.array(f_fcn(x, u)).flatten()

                    # Compute the cost J
                    control_cost = u.T @ R @ u
                    state_cost = (
                        (x_next - Dxinv @ x_off_normalized).T
                        @ Ptilde
                        @ (x_next - Dxinv @ x_off_normalized)
                    )
                    J = control_cost + state_cost

                    # Store the cost in the grid
                    costs[i, j] = J

            min_index = np.unravel_index(np.argmin(costs), costs.shape)
            min_cost = costs[min_index]
            optimal_u0 = u0_vals[min_index[0]]
            optimal_u1 = u1_vals[min_index[1]]
            optimal_u = np.array([optimal_u0, optimal_u1])
            optimal_J = min_cost
            state_cost = (
                (x - Dxinv @ x_off_normalized).T
                @ Ptilde
                @ (x - Dxinv @ x_off_normalized)
            )
        elif method == "lqr":
            # for i in range(10):
            A_, B_ = zoh_lin_fcn(x)
            A = np.array(A_)
            B = np.array(B_)
            K = np.linalg.inv(R + B.T @ Ptilde @ B) @ B.T @ Ptilde @ A
            # K = np.linalg.inv(R)@B.T@Ptilde
            unext = -K @ xtilde
            unext = np.clip(unext, umin, umax)
            control_cost = unext.T @ R @ unext
            # state_cost = (x_next - Dxinv@x_off_normalized).T @ Ptilde @ (x_next - Dxinv@x_off_normalized)
            # J = control_cost + state_cost
            # print(f"  {i}:  {u-unext}")
            optimal_u = unext
            # optimal_J = J
            state_cost = (
                (x - Dxinv @ x_off_normalized).T
                @ Ptilde
                @ (x - Dxinv @ x_off_normalized)
            )

        return optimal_u, state_cost

    Nsim = 40
    Usim = []
    costsim = []
    Xsim = [xinit]
    for i in range(Nsim):
        # print(f"Sim: {i}")
        u, cost = eval_model(Xsim[i], model_name)
        Usim.append(np.copy(np.array(u)))
        costsim.append(cost)
        Xsim.append(np.copy(np.array(f_fcn(Xsim[i], u)).flatten()))

    plot_robot(
        np.linspace(0, dt * Nsim, Nsim + 1),
        [None] + umax.tolist(),
        np.concatenate((np.array(costsim).reshape(-1, 1), np.array(Usim)), axis=1),
        np.array(Xsim),
        x_labels=["x [m]", "y [m]", "v [m/s]", "theta [rad]"],
        u_labels=["cost", "a [m/s^2]", "dtheta [rad/s]"],
        obst_pos=obstacle_positions,
        obst_rad=obstacle_radiuses,
        time_label="time [s]",
        plt_show=True,
        plt_name=f"{lqr_path}/closed_loop_{model_name}_x0=[{xinit[0]:.2f},{xinit[1]:.2f}].pdf",
        x_max=[0.5, 0.5, 0.2, None],
    )


if __name__ == "__main__":
    x0s = [
        [0.25, -0.25 - 0.15 + 0.01, 0, 3.14 / 2],
        [0.5, 0.5, 0, 0],
        [0.5, -0.5, 0, 0],
        [-0.5, 0.5, 0, 0],
        [-0.5, -0.5, 0, 0],
    ]
    model_names = [
        "Diagonal",
        "OuterProduct",
        "CholeskyInput",
        # "CholeskyLatent",
        # "DenseSpectralLatent", # ! TODO: deal with latent denorm
    ]
    n_dim = len(x0s[0])
    latent_weight = None
    latent_bias = None
    for name in model_names:
        params = critic_params[name]
        if "critic_name" in params.keys():
            params.update(critic_params[params["critic_name"]])
        params["num_obs"] = n_dim
        model_class = globals()[name]
        model = model_class(**params).to(DEVICE)
        if "Latent" in name:
            latent_weight, latent_bias = get_latent_matrix(
                [1, n_dim], model.latent_NN, device=DEVICE
            )
        for x in x0s:
            eval_lqr(
                model,
                xinit=x,
                model_name=name,
                W_latent=latent_weight,
                b_latent=latent_bias,
            )
