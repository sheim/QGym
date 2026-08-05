from enum import Enum, auto
import time
import threading

from rl_controller import RLController
from unitree_remote_controller import UnitreeRemoteController
from deploy_config import DeployConfig
from go2_deploy.utility import deploy_utility
from go2_deploy.utility.thread import RecurrentThread

import torch
from unitree_sdk2py.core.channel import (
    ChannelSubscriber,
    ChannelPublisher,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import (
    LowCmd_,
    LowState_,
)
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)
from unitree_sdk2py.utils.crc import CRC


class State(Enum):
    # Robot is in sport mode ("mcf" mode), lowcmd_thread is not started
    # Can only be accessed from RECOVERY
    DEFAULT_CTRL = auto()

    # Robot is in low state mode, lowcmd_thread is running
    # Can only be accessed from RECOVERY
    CUSTOM_CTRL = auto()

    # After emergency_stop() finishes processing, the robot is in low state mode
    # a new lowcmd_thread is created + not started, self.emergency_lowcmd_thread = None
    # Can be accessed from any other state, triggered automatically in
    # dangerous conditions
    EMERGENCY_STOP = auto()

    # After recovering, the robot is in sport mode, is standing up,
    # lowcmd_thread is not started
    # Can be accessed from any other state
    RECOVERY = auto()


class MainController:
    def __init__(self):
        self._state = State.RECOVERY
        self.rl_controller = RLController()
        self.remote_controller = UnitreeRemoteController()
        self.cfg = DeployConfig()

        self.obs_vec_size = 0
        for obs in self.cfg.obs_vector:
            self.obs_vec_size += self.cfg.obs_sizes[obs]

        # Last observation vector (extracted from LowState_ msg)
        self.last_obs = torch.zeros(self.obs_vec_size)
        self.last_obs_lock = threading.Lock()

        # Last action (target joint positions)
        self.last_action = torch.zeros(12)
        self.last_action_lock = threading.Lock()

        # Last commanded velocity (x, y, yaw)
        self.last_command = torch.zeros(3)
        self.last_command_lock = threading.Lock()

        # Log obs freq / control freq about once per second
        self.last_terminal_output_time = 0.0
        self.obs_count = 0
        self.action_count = 0

        # Unitree SDK pub/sub
        self._create_lowcmd_thread()
        self.emergency_lowcmd_thread = None

        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self._on_lowstate_msg, 10)
        self.motion_switcher_client = MotionSwitcherClient()
        self.motion_switcher_client.SetTimeout(10.0)
        self.motion_switcher_client.Init()
        self.sport_client = SportClient()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()

        self.crc = CRC()
        self.default_lowcmd = deploy_utility.default_lowcmd()
        self.emergency_lowcmd = deploy_utility.emergency_lowcmd()

    # Process incoming states and outgoing commands

    def action_to_lowcmd(self, action):
        return deploy_utility.action_to_lowcmd(self, action)

    def lowstate_to_obs(self, lowstate_msg):
        return deploy_utility.lowstate_to_obs(self, lowstate_msg)

    # Record LowState_ msg as observation vector
    def _on_lowstate_msg(self, msg):
        with self.last_obs_lock:
            self.last_obs = self.lowstate_to_obs(msg)

        t = time.monotonic()
        self.obs_count += 1
        if t > self.last_terminal_output_time + 1.0:
            self._output_terminal_info(t)

    # Process most recent obs and publish LowCmd_ msg
    def _control_loop(self):
        with self.last_obs_lock:
            action = self._act()
        lowcmd = self.action_to_lowcmd(self.last_action)
        with self.last_action_lock:
            self.last_action = action

        lowcmd.crc = self.crc.Crc(lowcmd)
        self.lowcmd_publisher.Write(lowcmd)

        t = time.monotonic()
        self.action_count += 1
        if t > self.last_terminal_output_time + 1.0:
            self._output_terminal_info(t)

    def _act(self):
        # act using lowcmd thread
        if self._state == State.CUSTOM_CTRL:
            return self.rl_controller.act(self.last_obs)
        else:
            print("Should not be using RL controller! This should not happen")
            self.emergency_stop()
            return None

    # Safety features

    def emergency_stop(self):
        self._state = State.EMERGENCY_STOP
        if self.lowcmd_thread.IsAlive():
            while not self.lowcmd_thread.Wait():
                print(
                    "\nWARNING: RL control loop did not stop during emergency stop, "
                    + "press L2+B to manually stop. Trying again in 2s\n"
                )
                time.sleep(2)

        self.emergency_lowcmd_thread = RecurrentThread(
            interval=0.01, target=self._emergency_control_loop
        )
        self.motion_switcher_client.ReleaseMode()
        self.emergency_lowcmd_thread.Start()
        print("Emergency stop activated! Recovering in 10 s")
        time.sleep(10)
        self.emergency_lowcmd_thread.Wait()
        self.emergency_lowcmd_thread = None
        self._create_lowcmd_thread()
        self.switch_to_recovery()

    # Check if robot is in unsafe condition, emergency stop if necessary
    def termination_check(self, lowstate_msg):
        terminate = False
        if terminate:
            self.emergency_stop()
        return

    # publish LowCmd_ damping messages
    def _emergency_control_loop(self):
        self.lowcmd_publisher.Write(self.emergency_lowcmd)

    # Switch between states

    def switch_to_recovery(self):
        print("Switching to recovery state")
        self._state = State.RECOVERY

        if self.lowcmd_thread.IsAlive():
            while not self.lowcmd_thread.Wait():
                print(
                    "\nWARNING: RL control loop did not stop during recovery, "
                    + "press L2+B to manually stop. Trying again in 1s\n"
                )
                time.sleep(1)
            self._create_lowcmd_thread()

        self.motion_switcher_client.SelectMode("mcf")
        mode = self.motion_switcher_client.CheckMode()[1]["name"]
        while mode != "mcf":
            print("Failed to switch to sport mode, trying again in 5s")
            time.sleep(5)
            mode = self.motion_switcher_client.CheckMode()[1]["name"]

        self.sport_client.RecoveryStand()
        time.sleep(10)
        print("Recovered")

    def switch_to_custom_controller(self):
        if self._state != State.RECOVERY:
            print("Must be in recovery state to switch to a custom controller!")
            return

        print("Switching to custom controller")
        self._state = State.CUSTOM_CTRL
        self.motion_switcher_client.ReleaseMode()
        mode = self.motion_switcher_client.CheckMode()[1]["name"]
        while mode != "":
            print("Failed to switch to low state mode, trying again in 5s")
            time.sleep(5)
            mode = self.motion_switcher_client.CheckMode()[1]["name"]

        self.lowcmd_thread.Start()

    def switch_to_default_controller(self):
        if self._state != State.RECOVERY:
            print("Must be in recovery state to switch to the default controller!")
            return

        print("Switching to default controller")
        self._state = State.DEFAULT_CTRL
        mode = self.motion_switcher_client.CheckMode()[1]["name"]
        while mode != "mcf":
            print("Failed to switch to sport mode, trying again in 5s")
            time.sleep(5)
            mode = self.motion_switcher_client.CheckMode()[1]["name"]

    # Utility

    def _create_lowcmd_thread(self):
        self.lowcmd_thread = RecurrentThread(
            interval=1 / self.cfg.ctrl_freq, target=self._control_loop
        )

    def _output_terminal_info(self, current_time):
        print(
            "\nObservation freq = "
            + f"{self.obs_count / (current_time - self.last_terminal_output_time):.5}"
            + " Hz, "
            + "custom policy control freq = "
            + f"{self.action_count / (current_time - self.last_terminal_output_time):.5}"  # noqa: E501
            + " Hz"
        )
        print("Current mode: " + self._state.name)
        print("\nTo switch modes: enter in terminal <key> and press enter")
        print(
            "(enter key pressed without input): emergency stop "
            + "(or press L2+B on wireless controller)"
        )
        print("r: recovery (return to standing position)")
        print("c: custom RL policy")
        print(
            "d: default controller (high level control with the wireless controller)\n"
        )
        self.last_terminal_output_time = time.monotonic()
        self.obs_count = 0
        self.action_count = 0
