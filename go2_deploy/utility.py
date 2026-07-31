"""Observation assembly for Go2 hardware deployment.

Turns a Unitree ``LowState_`` message into the observation vector the policy was
trained on.  Every constant here is derived from the training config so there is
exactly one source of truth; the assembly mirrors
``gym/envs/base/task_skeleton.py::get_states`` and the per-quantity definitions in
``gym/envs/base/legged_robot.py``.

Three conventions differ between the simulator and the robot, and each is crossed
in exactly one place:

* joint order -- sim/URDF is FL, FR, RL, RR while the SDK is FR, FL, RR, RL
  (``LEG_PERM``, applied once in :func:`decode_lowstate`),
* quaternion layout -- the IMU reports (w, x, y, z), the repo's quaternion math
  expects (x, y, z, w) (also handled in :func:`decode_lowstate`),
* angular velocity frame -- see :meth:`ObsBuilder._get_obs_base_ang_vel`.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch

from gym.envs.go2.go2_config import Go2Cfg, Go2RunnerCfg
from gym.utils.helpers import class_to_dict
from gym.utils.torch_quat import quat_rotate_inverse

if TYPE_CHECKING:  # * the DDS stack is not importable off-robot / in CI
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

NUM_DOF = 12
NUM_FEET = 4

# * DOF order as MuJoCo reads it out of resources/robots/go2/urdf/go2.urdf
JOINT_NAMES = [
    f"{leg}_{joint}_joint"
    for leg in ("FL", "FR", "RL", "RR")
    for joint in ("hip", "thigh", "calf")
]
# * motor_state / motor_cmd order, from unitree_go2_const.py::LegID
SDK_JOINT_NAMES = [
    f"{leg}_{joint}_joint"
    for leg in ("FR", "FL", "RR", "RL")
    for joint in ("hip", "thigh", "calf")
]
FOOT_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
SDK_FOOT_NAMES = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]

# * index an SDK-ordered array to get sim order.  Both permutations are their own
# * inverse, so the same array converts commands back to SDK order.
LEG_PERM = np.array([SDK_JOINT_NAMES.index(n) for n in JOINT_NAMES])
FOOT_PERM = np.array([SDK_FOOT_NAMES.index(n) for n in FOOT_NAMES])

# * gravity_vec in legged_robot is get_axis_params(-1.0, up_axis_idx=2)
GRAVITY_VEC = np.array([0.0, 0.0, -1.0], dtype=np.float32)

# * byte offsets of the float32 stick axes inside LowState_.wireless_remote[40],
# * from unitree_sdk2py/utils/joystick.py::Joystick.extract.  Parsed directly
# * rather than through Joystick, whose Axis.__call__ applies an undocumented
# * 0.03 low-pass -- the policy was trained on unfiltered commands.
_STICK_OFFSETS = {"lx": 4, "rx": 8, "ry": 12, "ly": 20}

OBS_LIST = list(Go2RunnerCfg.actor.obs)
ACTION_LIST = list(Go2RunnerCfg.actor.actions)
DECIMATION = int(Go2Cfg.control.desired_sim_frequency / Go2Cfg.control.ctrl_frequency)


def _scales() -> dict[str, np.ndarray]:
    """cfg.scaling as float32 arrays, the same dict the env builds."""
    return {
        key: np.asarray(val, dtype=np.float32)
        for key, val in class_to_dict(Go2Cfg.scaling).items()
    }


def _match_by_substring(table: dict, joint_names: list[str]) -> np.ndarray:
    """Resolve a per-joint config dict, as legged_robot._init_buffers does.

    The env falls back to zero and prints when a joint matches nothing; here an
    unmatched joint is a deployment-time error, so it raises instead.
    """
    values = np.zeros(len(joint_names), dtype=np.float32)
    for i, name in enumerate(joint_names):
        matches = [key for key in table.keys() if key in name]
        assert len(matches) == 1, f"{name} matched {matches} in {list(table.keys())}"
        values[i] = table[matches[0]]
    return values


DEFAULT_DOF_POS = _match_by_substring(
    Go2Cfg.init_state.default_joint_angles, JOINT_NAMES
)
P_GAINS = _match_by_substring(Go2Cfg.control.stiffness, JOINT_NAMES)
D_GAINS = _match_by_substring(Go2Cfg.control.damping, JOINT_NAMES)


def quat_rotate_inverse_np(quat_xyzw: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Single-sample wrapper around the repo's own quaternion math."""
    rotated = quat_rotate_inverse(
        torch.from_numpy(np.asarray(quat_xyzw, dtype=np.float32)).unsqueeze(0),
        torch.from_numpy(np.asarray(vec, dtype=np.float32)).unsqueeze(0),
    )
    return rotated.squeeze(0).numpy()


@dataclass(frozen=True)
class LowStateFrame:
    """A LowState_ decoded into simulator conventions.

    Joint quantities are in sim (FL, FR, RL, RR) order and the quaternion is
    (x, y, z, w); everything is float32 in SI units, except foot_force which is
    the raw int16 pressure count.
    """

    dof_pos: np.ndarray  # (12,) rad
    dof_vel: np.ndarray  # (12,) rad/s
    dof_acc: np.ndarray  # (12,) rad/s^2
    tau_est: np.ndarray  # (12,) Nm, diagnostics only
    quat_xyzw: np.ndarray  # (4,)
    gyro_body: np.ndarray  # (3,) rad/s, IMU frame
    foot_force: np.ndarray  # (4,) raw counts
    stick: dict[str, float]  # lx, ly, rx, ry in [-1, 1]
    tick_s: float  # message timestamp, seconds


def decode_lowstate(msg: "LowState_") -> LowStateFrame:
    """Decode a LowState_ into sim conventions.  The only layout-aware function."""
    motors = msg.motor_state
    dof_pos = np.array([motors[i].q for i in range(NUM_DOF)], dtype=np.float32)
    dof_vel = np.array([motors[i].dq for i in range(NUM_DOF)], dtype=np.float32)
    dof_acc = np.array([motors[i].ddq for i in range(NUM_DOF)], dtype=np.float32)
    tau_est = np.array([motors[i].tau_est for i in range(NUM_DOF)], dtype=np.float32)

    quat_wxyz = np.asarray(msg.imu_state.quaternion, dtype=np.float32)
    quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
    norm = float(np.linalg.norm(quat_xyzw))
    assert abs(norm - 1.0) < 1e-2, f"IMU quaternion is not unit ({norm:.4f})"

    remote = bytes(msg.wireless_remote)
    stick = {
        name: struct.unpack_from("<f", remote, offset)[0]
        for name, offset in _STICK_OFFSETS.items()
    }

    return LowStateFrame(
        dof_pos=dof_pos[LEG_PERM],
        dof_vel=dof_vel[LEG_PERM],
        dof_acc=dof_acc[LEG_PERM],
        tau_est=tau_est[LEG_PERM],
        quat_xyzw=quat_xyzw,
        gyro_body=np.asarray(msg.imu_state.gyroscope, dtype=np.float32),
        foot_force=np.asarray(msg.foot_force, dtype=np.float32)[FOOT_PERM],
        stick=stick,
        tick_s=float(np.uint32(msg.tick)) * 1e-3,
    )


def stick_to_commands(stick: dict[str, float]) -> np.ndarray:
    """Map joystick axes to [vx, vy, yaw_rate] over the trained command ranges.

    Reproduces the deadband applied when commands are resampled in training
    (legged_robot._resample_commands).  Axis polarity is a bench-verify item.
    """
    ranges = Go2Cfg.commands.ranges
    forward, backward = max(ranges.lin_vel_x), -min(ranges.lin_vel_x)
    lin_vel_x = stick["ly"] * (forward if stick["ly"] > 0.0 else backward)
    commands = np.array(
        [lin_vel_x, -stick["lx"] * ranges.lin_vel_y, -stick["rx"] * ranges.yaw_vel],
        dtype=np.float32,
    )
    if np.linalg.norm(commands[:2]) <= 0.2:
        commands[:2] = 0.0
    return commands


class ObsBuilder:
    """Assembles the policy observation from LowState_ messages.

    Each ``_get_obs_*`` returns the raw, unscaled quantity in sim order and shape
    ``(OBS_SIZE[name],)``; :meth:`get_obs` applies ``cfg.scaling`` and concatenates
    in ``obs_list`` order, exactly as TaskSkeleton.get_states does in training.
    """

    OBS_SIZE = {
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "commands": 3,
        "dof_pos_obs": NUM_DOF,
        "dof_vel": NUM_DOF,
        "dof_accel": NUM_DOF,
        "dof_pos_target": NUM_DOF,
        "foot_contact": NUM_FEET,
    }

    def __init__(
        self,
        obs_list: list[str] | None = None,
        contact_threshold: float = 200.0,
    ):
        self.obs_list = list(OBS_LIST if obs_list is None else obs_list)
        assert set(self.obs_list) <= set(self.OBS_SIZE), (
            f"unknown observations {set(self.obs_list) - set(self.OBS_SIZE)}"
        )
        self.contact_threshold = contact_threshold
        self.scales = _scales()
        self.default_dof_pos = DEFAULT_DOF_POS

        # * controller state that no single message carries
        self.last_action = np.zeros(NUM_DOF, dtype=np.float32)
        self.commands = np.zeros(3, dtype=np.float32)

    @property
    def num_obs(self) -> int:
        return sum(self.OBS_SIZE[name] for name in self.obs_list)

    # ── observation generators ────────────────────────────────────────────────

    def _get_obs_base_ang_vel(self, frame: LowStateFrame) -> np.ndarray:
        # * The MuJoCo backends copy the free joint's qvel[3:6] -- which MuJoCo
        # * already expresses in the body frame -- into root_states[:, 10:13], and
        # * legged_robot._post_physx_step then applies quat_rotate_inverse to it
        # * again.  The policy was therefore trained on R(q)^T @ omega_body.  The
        # * IMU gyro is omega_body, so one rotation here reproduces training.
        # * Revisit together with mujoco_cpu_backend._sync_state_from_mujoco if
        # * that double rotation is ever fixed (it requires retraining).
        return quat_rotate_inverse_np(frame.quat_xyzw, frame.gyro_body)

    def _get_obs_projected_gravity(self, frame: LowStateFrame) -> np.ndarray:
        return quat_rotate_inverse_np(frame.quat_xyzw, GRAVITY_VEC)

    def _get_obs_commands(self, frame: LowStateFrame) -> np.ndarray:
        return self.commands

    def _get_obs_dof_pos_obs(self, frame: LowStateFrame) -> np.ndarray:
        return frame.dof_pos - self.default_dof_pos

    def _get_obs_dof_vel(self, frame: LowStateFrame) -> np.ndarray:
        return frame.dof_vel

    def _get_obs_dof_accel(self, frame: LowStateFrame) -> np.ndarray:
        # * not part of any trained observation: no dof_accel exists in gym/, and
        # * cfg.scaling has no entry for it, so it passes through unscaled.
        return frame.dof_acc

    def _get_obs_dof_pos_target(self, frame: LowStateFrame) -> np.ndarray:
        # * training writes dof_pos_target = action * scale (TaskSkeleton.set_state)
        # * and observes it divided by the same scale, so this slot is the previous
        # * raw action.  Returning the scaled target keeps the raw/scaled split of
        # * this class intact.
        return self.last_action * self.scales["dof_pos_target"]

    def _get_obs_foot_contact(self, frame: LowStateFrame) -> np.ndarray:
        # * raw int16 pressure counts, not Newtons -- calibrate the threshold on
        # * the standing robot before trusting this.
        return (frame.foot_force > self.contact_threshold).astype(np.float32)

    GET_OBS: dict[str, Callable[["ObsBuilder", LowStateFrame], np.ndarray]] = {
        "base_ang_vel": _get_obs_base_ang_vel,
        "projected_gravity": _get_obs_projected_gravity,
        "commands": _get_obs_commands,
        "dof_pos_obs": _get_obs_dof_pos_obs,
        "dof_vel": _get_obs_dof_vel,
        "dof_accel": _get_obs_dof_accel,
        "dof_pos_target": _get_obs_dof_pos_target,
        "foot_contact": _get_obs_foot_contact,
    }

    # ── assembly ──────────────────────────────────────────────────────────────

    def get_state(self, name: str, frame: LowStateFrame) -> np.ndarray:
        """One scaled observation component, mirroring TaskSkeleton.get_state."""
        raw = self.GET_OBS[name](self, frame)
        assert raw.shape == (self.OBS_SIZE[name],), f"{name} has shape {raw.shape}"
        if name in self.scales:
            return raw / self.scales[name]
        return raw

    def get_obs_from_frame(self, frame: LowStateFrame) -> np.ndarray:
        obs = np.concatenate(
            [self.get_state(name, frame) for name in self.obs_list]
        ).astype(np.float32)
        assert obs.shape == (self.num_obs,)
        assert np.isfinite(obs).all(), "non-finite observation"
        return obs

    def get_obs(self, msg: "LowState_") -> np.ndarray:
        return self.get_obs_from_frame(decode_lowstate(msg))

    def get_obs_torch(self, msg: "LowState_") -> torch.Tensor:
        """(1, num_obs) float32, the shape actor.act_inference expects."""
        return torch.from_numpy(self.get_obs(msg)).unsqueeze(0)

    # ── controller state ──────────────────────────────────────────────────────

    def update_commands(self, frame: LowStateFrame) -> None:
        self.commands = stick_to_commands(frame.stick)

    def commit_action(self, action: np.ndarray) -> None:
        """Store the raw network output, to be observed on the next policy tick."""
        action = np.asarray(action, dtype=np.float32).reshape(NUM_DOF)
        self.last_action = action

    def reset(self) -> None:
        self.last_action = np.zeros(NUM_DOF, dtype=np.float32)
        self.commands = np.zeros(3, dtype=np.float32)


# ── action path (the mirror image of the observation path) ────────────────────


def action_to_dof_pos_target(action: np.ndarray) -> np.ndarray:
    """Raw network output -> commanded joint angles [rad], sim order.

    Mirrors TaskSkeleton.set_state followed by legged_robot._compute_torques,
    where the PD law tracks ``dof_pos_target + default_dof_pos``.
    """
    action = np.asarray(action, dtype=np.float32).reshape(NUM_DOF)
    scale = np.asarray(Go2Cfg.scaling.dof_pos_target, dtype=np.float32)
    return action * scale + DEFAULT_DOF_POS


def to_sdk_order(joint_values: np.ndarray) -> np.ndarray:
    """Sim (FL, FR, RL, RR) order -> SDK (FR, FL, RR, RL) order."""
    return np.asarray(joint_values)[LEG_PERM]


if __name__ == "__main__":
    pass
