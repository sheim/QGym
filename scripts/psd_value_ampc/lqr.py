import pickle
import torch 
import numpy as np

from train import PsdCholOff
from utils import plot_robot
from unicycle import umin, umax, R, dt, obstacle_positions, obstacle_radiuses

@torch.no_grad()
def eval_lqr(filename, xinit):
    assert "PsdCholOff" in filename
    with open(f"models/{filename}.pkl", "rb") as f:
        data = pickle.load(f)
        V_max = data["V_max"]
        V_min = data["V_min"]
        X_max = data["X_max"]
        X_min = data["X_min"]
    
    model = PsdCholOff(input_dim=len(X_max))
    model.load_state_dict(torch.load(f"models/{filename}.pth", map_location="cpu"))
    model.eval()
        
    def eval_model(x, u=np.zeros(2)):
        x_normalized = 2*(x - X_min) / (X_max - X_min)-1
        _, A_normalized_, x_off_normalized_ = model(torch.tensor(x_normalized, dtype=torch.float32))
        A_normalized = A_normalized_.detach().numpy()
        x_off_normalized = x_off_normalized_.detach().numpy()
        Dx = np.diag(2/(X_max - X_min))
        Dxinv = np.diag((X_max - X_min)/2)
        Dvinv = (V_max-V_min)
        Ptilde = Dvinv*Dx.T@A_normalized@Dx
        xtilde = x-Dxinv@x_off_normalized
        
        u0_vals = np.linspace(umin[0], umax[0], 20)
        u1_vals = np.linspace(umin[1], umax[1], 20)
        costs = np.zeros((20, 20))
        for i, u0 in enumerate(u0_vals):
            for j, u1 in enumerate(u1_vals):
                u = np.array([u0, u1])
                
                DT = 0.5
                
                # Compute x_next based on the dynamics
                x_next = x + DT * np.array([u[0] * np.cos(u[1]), u[0] * np.sin(u[1])])
                
                # Compute the cost J
                control_cost = DT/dt*u.T @ R @ u
                state_cost = (x_next - Dxinv@x_off_normalized).T @ Ptilde @ (x_next - Dxinv@x_off_normalized)
                J = control_cost + state_cost
                
                # Store the cost in the grid
                costs[i, j] = J
        
        min_index = np.unravel_index(np.argmin(costs), costs.shape)
        min_cost = costs[min_index]
        optimal_u0 = u0_vals[min_index[0]]
        optimal_u1 = u1_vals[min_index[1]]
        optimal_u = np.array([optimal_u0, optimal_u1])
        
        # u = np.zeros(2)
        # def dt_linearize_system(x,u):
        #     A = np.diag([1,1])
        #     B = np.array([
        #             [ np.cos(u[1]), -u[0]*np.sin(u[1])  ],
        #             [ np.sin(u[1]), u[0]*np.cos(u[1]) ]
        #         ])
        #     return A, B
        
        # # for i in range(10):
        # A,B = dt_linearize_system(x,u)
        # # K = np.linalg.inv( R + B.T @ Ptilde @ B ) @ B.T @ Ptilde @ A
        # K = np.linalg.inv(R)@B.T@Ptilde
        # unext = -K@xtilde
        # unext = np.clip(unext, umin, umax)
        # print(f"  {i}:  {u-unext}")
        # u = unext
            
        return optimal_u
        
    Nsim = 100
    Usim = []
    Xsim = [xinit]
    u = np.zeros(2)
    for i in range(Nsim):
        # print(f"Sim: {i}")
        u = eval_model(Xsim[i], u)
        Usim.append(u)
        Xsim.append(Xsim[i]+dt*np.array([u[0]*np.cos(u[1]), u[0]*np.sin(u[1])]))
    
    
    plot_robot(
            np.linspace(0, Nsim, Nsim + 1),
            list(umax),
            np.array(Usim),
            np.array(Xsim),
            x_labels=["x", "y"],
            u_labels=["v", "theta"],
            obst_pos=obstacle_positions,
            obst_rad=obstacle_radiuses,
            time_label="Sim steps",
            plt_show=True,
            plt_name=f"plots/closed_loop_{filename}_x0=[{xinit[0]:.2f},{xinit[1]:.2f}].pdf"
        )
    
if __name__=="__main__":
    x0s = [[0.5, 0.5], [0.5, -0.5], [-0.5, 0.5], [-0.5, -0.5]]
    for x in x0s:
        eval_lqr(filename="unicycle_2D_64_PsdCholOff", xinit=x)