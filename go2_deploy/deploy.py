from unitree_sdk2py.core import channel
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from main_controller import MainController
import sys

import time


def main():
    if len(sys.argv) < 2:
        print(
            "Must pass the name of the robot network interface as an argument, ex. eth0"
        )
        sys.exit()

    # CycloneDDS 0.10.2 bug workaround
    channel.ChannelConfigHasInterface = channel.ChannelConfigHasInterface.replace(
        "<Verbosity>config</Verbosity>", "<Verbosity>none</Verbosity>"
    )

    print("Starting up default controller")
    ChannelFactoryInitialize(
        0, sys.argv[1]
    )  # Name of robot network interface (terminal: `ifconfig`)
    controller = MainController()
    controller.start_unitree_clients()

    print("receiving msgs for 300s")
    time.sleep(300)
    print("done")


if __name__ == "__main__":
    main()
