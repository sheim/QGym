from casadi import sin, cos, vertcat
def fg(phi, theta, tau_W, tau_R, m_W, m_B, m_R, r_W, l_WB):
    return vertcat(
        0,
        -(981*sin(phi)*(m_B*r_W+m_R*r_W+m_W*r_W+l_WB*m_B*cos(theta)+2*l_WB*m_R*cos(theta)))/100,
        tau_W-(981*l_WB*m_B*cos(phi)*sin(theta))/100-(981*l_WB*m_R*cos(phi)*sin(theta))/50,
        -tau_W,
        -tau_R
    )