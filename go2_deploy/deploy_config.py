class DeployConfig:
    interface = "eno1"  # Name of robot network interface (terminal: `ifconfig`)

    ctrl_freq = 100  # Hz

    timeout_duration = 1.0  # s; how long to wait for obs from robot before shutdown
