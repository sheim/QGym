import torch
from deploy_config import DeployConfig
from gym.utils.torch_quat import quat_rotate_inverse
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC


def _get_obs_base_ang_vel(main_controller, lowstate_msg):
    return torch.tensor(lowstate_msg.imu_state.gyroscope)


def _get_obs_projected_gravity(main_controller, lowstate_msg):
    # Convert Unitree WXYZ to QGym XYZW
    base_quat = torch.tensor(lowstate_msg.imu_state.quaternion)[[1, 2, 3, 0]]
    gravity_vec = torch.tensor([0.0, 0.0, -1.0])
    return quat_rotate_inverse(base_quat, gravity_vec)


def _get_obs_commands(main_controller, lowstate_msg):
    return main_controller.last_command


def _get_obs_dof_pos_obs(main_controller, lowstate_msg):
    motor_states = lowstate_msg.motor_state
    # Unitree motor order: Front Right hip (haa), FR thigh (hfe), FR calf (kfe),
    # Front Left ... Rear Right ... Rear Left
    dof_pos_unitree_convention = torch.zeros(12)
    for i in range(12):
        dof_pos_unitree_convention[i] = motor_states[i].q
    # QGym motor order: FL hip, FL thigh, FL calf, FR ... RL ... RR
    unitree_to_qgym_joint_idx = torch.tensor([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])
    return dof_pos_unitree_convention[unitree_to_qgym_joint_idx]


def _get_obs_dof_vel(main_controller, lowstate_msg):
    motor_states = lowstate_msg.motor_state
    dof_vel_unitree_convention = torch.zeros(12)
    for i in range(12):
        dof_vel_unitree_convention[i] = motor_states[i].dq
    unitree_to_qgym_joint_idx = torch.tensor([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])
    return dof_vel_unitree_convention[unitree_to_qgym_joint_idx]


def _get_obs_dof_accel(main_controller, lowstate_msg):
    motor_states = lowstate_msg.motor_state
    dof_accel_unitree_convention = torch.zeros(12)
    for i in range(12):
        dof_accel_unitree_convention[i] = motor_states[i].ddq
    unitree_to_qgym_joint_idx = torch.tensor([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])
    return dof_accel_unitree_convention[unitree_to_qgym_joint_idx]


def _get_obs_dof_pos_target(main_controller, lowstate_msg):
    return main_controller.last_action


# Assemble observation vector (torch tensor) from LowState_ msg
def lowstate_to_obs(main_controller, lowstate_msg):
    get_obs_piece = {
        "base_ang_vel": _get_obs_base_ang_vel,
        "projected_gravity": _get_obs_projected_gravity,
        "commands": _get_obs_commands,
        "dof_pos_obs": _get_obs_dof_pos_obs,
        "dof_vel": _get_obs_dof_vel,
        "dof_accel": _get_obs_dof_accel,
        "dof_pos_target": _get_obs_dof_pos_target,
    }

    obs_vector = torch.zeros(main_controller.obs_vec_size)
    i = 0
    for obs in main_controller.cfg.obs_vector:
        obs_size = main_controller.cfg.obs_sizes[obs]
        obs_vector[i : i + obs_size] = get_obs_piece[obs](main_controller, lowstate_msg)
        i += obs_size

    return obs_vector


# Assemble LowCmd_ msg from action
def action_to_lowcmd(main_controller, action):
    lowcmd = main_controller.default_lowcmd
    for i in range(12):
        lowcmd.motor_cmd[i].q = action[i]

    return lowcmd


def default_lowcmd():
    lowcmd = unitree_go_msg_dds__LowCmd_()
    crc = CRC()
    lowcmd.head[0] = 0xFE
    lowcmd.head[1] = 0xEF
    lowcmd.level_flag = 0xFF
    lowcmd.gpio = 0
    cfg = DeployConfig()

    for i in range(20):
        lowcmd.motor_cmd[i].mode = 0x01
        lowcmd.motor_cmd[i].q = 0
        lowcmd.motor_cmd[i].kp = 0
        lowcmd.motor_cmd[i].dq = 0
        lowcmd.motor_cmd[i].kd = 0
        lowcmd.motor_cmd[i].tau = 0

    # the motors that are actually used
    for i in range(12):
        lowcmd.motor_cmd[i].kp = cfg.kp
        lowcmd.motor_cmd[i].kd = cfg.kd

    lowcmd.crc = crc.Crc(lowcmd)

    return lowcmd


def emergency_lowcmd():
    lowcmd = default_lowcmd()
    crc = CRC()

    for i in range(12):
        lowcmd.motor_cmd[i].kp = 0
        lowcmd.motor_cmd[i].kd = 5.0

    lowcmd.crc = crc.Crc(lowcmd)

    return lowcmd
