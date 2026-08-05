from unitree_sdk2py.core import channel
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from main_controller import MainController
from keyboard_handler import KeyboardHandler
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

    ChannelFactoryInitialize(
        0,
        sys.argv[1],  # Name of robot network interface (terminal: `ifconfig`)
    )
    controller = MainController()  # noqa: F841
    print(
        "\nConnection successful! Keep the robot away from obstacles and "
        + "always have access to the remote controller (Emergency stop = L2+B)\n"
    )

    keyboard_handler = KeyboardHandler(controller)  # noqa: F841

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
