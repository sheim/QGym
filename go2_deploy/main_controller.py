from enum import Enum, auto

import time

from rl_controller import RLController
from deploy_config import DeployConfig
from unitree_sdk2py.core.channel import (
    ChannelSubscriber,
    ChannelPublisher,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import (
    LowCmd_,
    LowState_,
)  # , SportModeState_
from unitree_sdk2py.go2.sport.sport_client import SportClient


class State(Enum):
    DEFAULT_CTRL = auto()  # Unitree sport mode controller
    CUSTOM_CTRL = auto()  # custom RL controller
    E_STOP = auto()  # damping mode (sport mode)
    RECOVER = auto()  # stand up (sport mode)


class MainController:
    def __init__(self):
        self._state = None
        self.rl_controller = RLController()
        self.cfg = DeployConfig()

        self.last_obs = None

        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.sc = SportClient()

    # start unitree sdk clients
    def start_unitree_clients(self):
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber.Init(self._on_lowstate_msg, 10)
        self.sc.SetTimeout(self.cfg.timeout_duration)
        self.sc.Init()

    # convert LowState_ to obs vector
    def lowstate_to_obs(self, msg):
        pass

    def _on_lowstate_msg(self, msg):
        print("lowstate msg received! time = " + str(time.monotonic()))
