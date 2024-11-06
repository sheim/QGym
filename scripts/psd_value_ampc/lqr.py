import pickle
import torch 
import numpy as np

from train import PsdCholOff
from utils import plot_robot
# from unicycle import umin, umax, R, dt, obstacle_positions, obstacle_radiuses
from unicycle4d import umin, umax, R, dt, obstacle_positions, obstacle_radiuses, f
import math
import casadi as cs

def zoh_lin(T):
    nx = 4
    nu = 2
    x = cs.SX.sym('x',nx)
    u = cs.SX.sym('u',nu)
    Ac = cs.jacobian(f(x,u),x)
    Bc = cs.jacobian(f(x,u),u)
    Ad = cs.SX.eye(nx)
    Bd = cs.SX.zeros(nx,nu)
    for i in range(4):
        Toverifac = T**(i+1)/math.factorial(i+1)
        Ad += Toverifac*cs.mpower(Ac,i+1)
        Bd += Toverifac*cs.mpower(Ac, i)@Bc
    zoh_lin_fcn = cs.Function("zoh_lin", [x], [Ad, Bd], ["x"], ["Ad", "Bd"])
    return zoh_lin_fcn

def rk4_nonlin(T):
    nx = 4
    nu = 2
    x = cs.SX.sym('x',nx)
    u = cs.SX.sym('u',nu)
    k1 = f(x,         u)
    k2 = f(x+T/2*k1,  u)
    k3 = f(x+T/2*k2,  u)
    k4 = f(x+T*k3,    u)
    x_next = x + T/6*(k1+2*k2+2*k3+k4)
    rk4_nl_fcn = cs.Function("rk4_nonlin", [x, u], [x_next], ["x", "u"], ["x_next"])
    return rk4_nl_fcn

zoh_lin_fcn = zoh_lin(dt)
f_fcn = rk4_nonlin(dt)

@torch.no_grad()
def eval_lqr(filename, xinit):
    with open(f"models/{filename}_normalization.pkl", "rb") as f:
        data = pickle.load(f)
        V_max = data["V_max"]
        V_min = data["V_min"]
        X_max = data["X_max"]
        X_min = data["X_min"]
    
    model = PsdCholOff(input_dim=len(X_max))
    model.load_state_dict(torch.load(f"models/{filename}_{type(model).__name__}.pth", map_location="cpu"))
    model.eval()
        
    def eval_model(x):
        x_normalized = 2*(x - X_min) / (X_max - X_min)-1
        _, A_normalized_, x_off_normalized_ = model(torch.tensor(x_normalized, dtype=torch.float32))
        A_normalized = A_normalized_.detach().numpy()
        x_off_normalized = x_off_normalized_.detach().numpy()
        Dx = np.diag(2/(X_max - X_min))
        Dxinv = np.diag((X_max - X_min)/2)
        Dvinv = (V_max-V_min)
        Ptilde = Dvinv*Dx.T@A_normalized@Dx
        xtilde = x-Dxinv@x_off_normalized
        
        # gridding
        method = "lqr"
        if method=="gridding":
            u0_vals = np.linspace(umin[0], umax[0], 20)
            u1_vals = np.linspace(umin[1], umax[1], 20)
            costs = np.zeros((20, 20))
            for i, u0 in enumerate(u0_vals):
                for j, u1 in enumerate(u1_vals):
                    u = np.array([u0, u1])
                    
                    # DT = 0.5
                    
                    # Compute x_next based on the dynamics
                    x_next = np.array(f_fcn(x,u)).flatten()
                    
                    # Compute the cost J
                    control_cost = u.T @ R @ u
                    state_cost = (x_next - Dxinv@x_off_normalized).T @ Ptilde @ (x_next - Dxinv@x_off_normalized)
                    J = control_cost + state_cost
                    
                    # Store the cost in the grid
                    costs[i, j] = J
            
            min_index = np.unravel_index(np.argmin(costs), costs.shape)
            min_cost = costs[min_index]
            optimal_u0 = u0_vals[min_index[0]]
            optimal_u1 = u1_vals[min_index[1]]
            optimal_u = np.array([optimal_u0, optimal_u1])
            optimal_J = min_cost
            state_cost = (x - Dxinv@x_off_normalized).T @ Ptilde @ (x - Dxinv@x_off_normalized)
        elif method=='lqr':
            # for i in range(10):
            A_,B_ = zoh_lin_fcn(x)
            A = np.array(A_)
            B = np.array(B_)
            K = np.linalg.inv( R + B.T @ Ptilde @ B ) @ B.T @ Ptilde @ A
            # K = np.linalg.inv(R)@B.T@Ptilde
            unext = -K@xtilde
            unext = np.clip(unext, umin, umax)
            control_cost = unext.T @ R @ unext
            # state_cost = (x_next - Dxinv@x_off_normalized).T @ Ptilde @ (x_next - Dxinv@x_off_normalized)
            # J = control_cost + state_cost
            # print(f"  {i}:  {u-unext}")
            optimal_u = unext
            # optimal_J = J
            state_cost = (x - Dxinv@x_off_normalized).T @ Ptilde @ (x - Dxinv@x_off_normalized)
            
        return optimal_u, state_cost
        
    Nsim = 40
    Usim = []
    costsim = []
    Xsim = [xinit]
    for i in range(Nsim):
        # print(f"Sim: {i}")
        u,cost = eval_model(Xsim[i])
        Usim.append(np.copy(np.array(u)))
        costsim.append(cost)
        Xsim.append(np.copy(np.array(f_fcn(Xsim[i],u)).flatten()))
    
    
    plot_robot(
            np.linspace(0, dt*Nsim, Nsim + 1),
            [None]+umax.tolist(),
            np.concatenate((np.array(costsim).reshape(-1, 1),np.array(Usim)), axis=1),
            np.array(Xsim),
            x_labels=["x [m]", "y [m]", "v [m/s]", "theta [rad]"],
            u_labels=["cost", "a [m/s^2]", "dtheta [rad/s]"],
            obst_pos=obstacle_positions,
            obst_rad=obstacle_radiuses,
            time_label="time [s]",
            plt_show=True,
            plt_name=f"plots/closed_loop_{filename}_x0=[{xinit[0]:.2f},{xinit[1]:.2f}].pdf",
            x_max = [0.5,0.5,0.2,None]
        )
    
if __name__=="__main__":
    x0s = [[0.25,-0.25-0.15+0.01,0,3.14/2], [0.5, 0.5,0,0], [0.5, -0.5,0,0], [-0.5, 0.5,0,0], [-0.5, -0.5,0,0]]
    for x in x0s:
        eval_lqr(filename="unicycle_4D_lessnoise_cl_3375", xinit=x)