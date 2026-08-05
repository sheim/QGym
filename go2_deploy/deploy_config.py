class DeployConfig:
    ctrl_freq = 500  # Hz

    kp = 30.0  # Stiffness constant
    kd = 2.0  # Damping constant

    # Observation vector used to train robot
    obs_vector = [
        "base_ang_vel",
        "projected_gravity",
        "commands",
        "dof_pos_obs",
        "dof_vel",
        "dof_pos_target",
    ]  # add later? foot contact

    # Set ranges of joystick commands
    command_limits = {
        "lin_vel_x": 2.0,  # m/s
        "lin_vel_y": 1.0,  # m/s
        "yaw_vel": 3.0,  # rad/s
    }

    # Specify size of observation vector. should not have to modify this
    obs_sizes = {
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "commands": 3,
        "dof_pos_obs": 12,
        "dof_vel": 12,
        "dof_accel": 12,
        "dof_pos_target": 12,
    }

    # Check for unsafe config
    def __init__(self):
        if self.ctrl_freq < 50:
            raise ValueError("ctrl_freq should be > 50")
        if self.kp > 500:
            raise ValueError("kp should be < 500")
        if self.kd < 1:
            raise ValueError("kd should be > 1")
        for obs in self.obs_vector:
            if obs not in self.obs_sizes:
                raise KeyError("observation " + obs + " not supported")
