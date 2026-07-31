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
    SportModeState_,
)
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
        self.last_obs_time = 0.0

        self.last_action = None
        self.last_action_time = 0.0

        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.sportmodestate_subscriber = ChannelSubscriber(
            "rt/sportmodestate", SportModeState_
        )
        self.sport_client = SportClient()
        self.sport_client.SetTimeout(self.cfg.timeout_duration)

    def start_unitree_clients(self):
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber.Init(self._on_lowstate_msg, 10)
        self.sportmodestate_subscriber.Init(self._on_sportmodestate_msg, 10)
        self.sport_client.Init()

    def lowstate_to_obs(self, msg: LowState_):
        pass

    def action_to_lowcmd(self, action) -> LowCmd_:
        pass

    def _on_lowstate_msg(self, msg):
        self.last_obs = self.lowstate_to_obs(msg)
        t = time.monotonic()
        print(f"lowstate obs recieved, obs_freq = {1 / (t - self.last_obs_time)}")
        self.last_obs_time = t

    def _on_sportmodestate_msg(self, msg):
        print(msg.error_code)
