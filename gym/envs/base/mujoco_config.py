"""MuJoCo settings shared by CPU and Warp model construction."""


class MuJoCoCfg:
    # Native defaults; robot configs override only their intentional tuning.
    njmax = -1
    ccd_iterations = 35
    disableflags = 0
    solref = [0.02, 1.0]
