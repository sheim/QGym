class DeployConfig:
    ctrl_freq = 500  # Hz

    timeout_duration = 1.0  # s; how long to wait for obs from robot before shutdown

    obs_vector = [
        "base_ang_vel",
        "projected_gravity",
        "commands",
        "dof_pos_obs",
        "dof_vel",
        "dof_pos_target",
    ]
    # dof_acc
    # foot contact
