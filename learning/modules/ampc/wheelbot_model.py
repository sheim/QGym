import casadi as cs
from acados_template import AcadosModel
from learning.modules.ampc.fb import fb
from learning.modules.ampc.fM import fM
from learning.modules.ampc.fg import fg

def export_wheelbot_ode_model():

    model_name = "wheelbot_ode"

    # set up states & controls   
    psi     = cs.SX.sym("psi")
    phi     = cs.SX.sym("phi")
    theta   = cs.SX.sym("theta")
    dpsi    = cs.SX.sym("dpsi")
    dphi    = cs.SX.sym("dphi")
    dtheta  = cs.SX.sym("dtheta")
    q_W     = cs.SX.sym("q_W")
    q_R     = cs.SX.sym("q_R")
    omega_W = cs.SX.sym("omega_W")
    omega_R = cs.SX.sym("omega_R")
    
    tau_W = cs.SX.sym("tau_W")
    tau_R = cs.SX.sym("tau_R")

    # xdot
    psi_dot     = cs.SX.sym("psi")
    phi_dot     = cs.SX.sym("phi")
    theta_dot   = cs.SX.sym("theta")
    dpsi_dot    = cs.SX.sym("dpsi")
    dphi_dot    = cs.SX.sym("dphi")
    dtheta_dot  = cs.SX.sym("dtheta")
    q_W_dot     = cs.SX.sym("q_W")
    q_R_dot     = cs.SX.sym("q_R")
    omega_W_dot = cs.SX.sym("omega_W")
    omega_R_dot = cs.SX.sym("omega_R")
    
    # parameters
    m_WR        = cs.SX.sym("m_WR")
    m_B         = cs.SX.sym("m_B")
    I_Wxz_Ryz   = cs.SX.sym("I_Wxz_Ryz")
    I_Wy_Rx     = cs.SX.sym("I_Wy_Rx")
    I_Bx        = cs.SX.sym("I_Bx")
    I_By        = cs.SX.sym("I_By")
    I_Bz        = cs.SX.sym("I_Bz")
    r_W         = cs.SX.sym("r_W")
    l_WB        = cs.SX.sym("l_WB")
    fric_magn   = cs.SX.sym("fric_magn")
    fric_slope  = cs.SX.sym("fric_slope")
    

    u = cs.vertcat(tau_W, tau_R)
    p = cs.vertcat(m_WR, m_B, I_Wxz_Ryz, I_Wy_Rx, I_Bx, I_By, I_Bz, r_W, l_WB, fric_magn, fric_slope)
    
    x = cs.vertcat(psi, phi, theta, dpsi, dphi, dtheta, q_W, q_R, omega_W, omega_R)
    x_dot = cs.vertcat(psi_dot, phi_dot, theta_dot, dpsi_dot, dphi_dot, dtheta_dot, q_W_dot, q_R_dot, omega_W_dot, omega_R_dot)
    
        
    K_W = cs.horzcat(-400e-3, -40e-3, -3e-3,  -3e-3)
    K_R = cs.horzcat(-1.3e0,  -1.6e-1,  -0.8e-04, -4e-04)
    tau_W_ = -K_W@cs.vertcat(theta, dtheta, q_W, omega_W) + tau_W
    tau_R_ = -K_R@cs.vertcat(phi, dphi, q_R, omega_R) + tau_R
    
    constraint = cs.types.SimpleNamespace()
    constraint.expr = cs.vertcat(tau_W_, tau_R_)

    # dynamics
    M = fM(phi, theta, m_WR, m_B, m_WR, I_Wxz_Ryz, I_Wy_Rx, I_Wxz_Ryz, I_Bx, I_By, I_Bz, I_Wy_Rx, I_Wxz_Ryz, I_Wxz_Ryz, r_W, l_WB)
    b = fb(phi, theta, dpsi, dphi, dtheta, omega_W, omega_R, m_WR, m_B, m_WR, I_Wxz_Ryz, I_Wy_Rx, I_Wxz_Ryz, I_Bx, I_By, I_Bz, I_Wy_Rx, I_Wxz_Ryz, I_Wxz_Ryz, r_W, l_WB)
    g = fg(phi, theta, tau_W_, tau_R_, m_WR, m_B, m_WR, r_W, l_WB)
    tau_contact = cs.vertcat(cs.tanh(dpsi*fric_slope)*fric_magn, 0,0,0,0)
    
    f_impl_q = cs.vertcat(psi_dot, phi_dot, theta_dot, q_W_dot, q_R_dot)-cs.vertcat(dpsi, dphi, dtheta, omega_W, omega_R)
    f_impl_dq = M @ cs.vertcat(dpsi_dot, dphi_dot, dtheta_dot, omega_W_dot, omega_R_dot) + b + g + tau_contact

    f_impl = cs.vertcat(f_impl_q, f_impl_dq)

    model = AcadosModel()

    model.f_impl_expr = f_impl
    # model.f_expl_expr = f_expl
    model.x = x
    model.xdot = x_dot
    model.u = u
    model.p = p
    model.name = model_name

    return model