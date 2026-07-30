from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from main_controller import MainController
import sys

import time


def main():
    ChannelFactoryInitialize(0, sys.argv[1])
    controller = MainController()
    controller.start_unitree_clients()

    controller.sc.Damp()
    print("Damping")
    time.sleep(10)
    controller.sc.RecoveryStand()
    print("Standing")
    time.sleep(10)
    controller.sc.Damp()
    print("Damping")


if __name__ == "__main__":
    main()
