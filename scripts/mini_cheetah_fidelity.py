"""Legged cross-engine fidelity probes for mini_cheetah_ref (Phase 4 parity).

Policy-free, deterministic probes driven identically on every backend:

  drop  — free base, PD-hold the default pose, released from a height onto the
          plane.  The CONTACT / impact-model comparison (feet <-> ground).
  step  — fix_base_link=True, spawned clear of the ground: step the desired
          joint angles and record the joint response.  Pure PD + limb dynamics,
          NO contact — isolates actuator / integration differences.
  torque — fixed base, zero gravity: apply a one-step one-hot joint torque and
           record the full joint-acceleration response matrix.
  damping — fixed base, zero gravity: release each joint from a one-hot initial
            velocity with zero applied torque and record passive decay.
  kinematics — fixed base, zero gravity: prescribe the default pose plus one
               offset per joint and compare body transforms.
  reaction — floating base, zero gravity, no ground: apply one-hot joint torques
             and compare the equal-and-opposite base response.
  impact — floating base, 500 Hz: drop onto the plane and record per-foot
           positions and force vectors through the first impact and settling.
  slide — settle on the plane, inject horizontal base velocity, and record
          tangential deceleration and foot slip at 500 Hz.

`dof_pos_target` is a residual on the default pose (torque =
p_gains*(dof_pos_target + default - dof_pos)), so 0 holds default and a nonzero
entry steps that joint.  reset_to_basic gives a deterministic IC identical on
every backend; contact termination is disabled so trajectories run full length.

Let PF="scripts/mini_cheetah_fidelity.py", F=logs/mc_fid (for vsim, prefix
`uv run --env-file .env.vsim` and pass `--backend vsim`):

    uv run $PF run --probe drop --backend mujoco --device cpu    --out $F/drop_cpu.npz
    uv run $PF run --probe step --backend mujoco --device cuda:0 --out $F/step_warp.npz
    uv run $PF compare $F/drop_*.npz

Expectation: cpu ~= warp tight (same MuJoCo model/solver). Differences in the
four non-contact probes localize vsim discrepancies to inertia/torque semantics,
passive joint dynamics, kinematic frames, or floating-base coupling.
"""

import argparse
import os
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import torch

from gym import LEGGED_GYM_ROOT_DIR
from gym.envs.mini_cheetah.mini_cheetah_config import MINI_CHEETAH_DOF_NAMES
from gym.utils.helpers import set_seed
from gym.utils.task_registry import task_registry

TASK = "mini_cheetah_ref"
NONCONTACT_PROBES = ("torque", "damping", "kinematics", "reaction")
CONTACT_PROBES = ("impact", "slide")
# mini_cheetah_simple.urdf foot collision sphere in the local foot frame.
FOOT_SPHERE_LOCAL_CENTER = np.asarray([0.0, 0.0, 0.024])
FOOT_SPHERE_RADIUS = 0.0202


def reset_probe_state(env):
    """Restore the configured deterministic IC after task construction.

    TaskSkeleton.reset() performs one implicit control step during construction.
    That step is useful for normal task initialization but would make a fidelity
    probe start from an already-evolved, backend-dependent state.
    """
    env_ids = torch.arange(env.num_envs, device=env.device)
    env.dof_pos_target.zero_()
    env._reset_system(env_ids)
    env.dof_pos_target.zero_()


def build_env(
    backend,
    device,
    num_envs,
    fixed_base,
    base_z,
    t_end,
    *,
    disable_gravity=False,
    with_ground=True,
    terrain_properties=None,
    vsim_properties=None,
):
    import gym.envs  # noqa: F401 — registers tasks

    env_cfg, train_cfg = task_registry.get_cfgs(TASK)
    env_cfg.env.num_envs = num_envs
    env_cfg.asset.fix_base_link = fixed_base
    env_cfg.asset.disable_gravity = disable_gravity
    if not with_ground:
        env_cfg.terrain.mesh_type = None
        env_cfg.terrain.measure_heights = False
    for name, value in (terrain_properties or {}).items():
        if value is not None:
            setattr(env_cfg.terrain, name, value)
    if vsim_properties:
        knobs = getattr(env_cfg, "vsim_attributes", SimpleNamespace())
        for name, value in vsim_properties.items():
            if value is not None:
                setattr(knobs, name, value)
        env_cfg.vsim_attributes = knobs
    # Full-length trajectories: no timeout- or contact-driven resets mid-probe.
    env_cfg.asset.terminate_after_contacts_on = []
    env_cfg.env.episode_length_s = t_end + 10.0
    env_cfg.init_state.reset_mode = "reset_to_basic"
    env_cfg.init_state.pos = [0.0, 0.0, base_z]
    env_cfg.seed = 0
    train_cfg.seed = 0
    set_seed(0)

    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    return task_registry.make_env_mujoco(
        TASK, env_cfg, device=device, headless=True, backend=backend
    )


def run_drop(env, n_steps):
    """Free base falls from height, PD holds the default pose."""
    nfeet = len(env.feet_indices)
    base_pos = np.empty((n_steps, env.num_envs, 3), dtype=np.float32)
    base_quat = np.empty((n_steps, env.num_envs, 4), dtype=np.float32)
    base_lin_vel = np.empty((n_steps, env.num_envs, 3), dtype=np.float32)
    base_ang_vel = np.empty((n_steps, env.num_envs, 3), dtype=np.float32)
    dof_pos = np.empty((n_steps, env.num_envs, env.num_dof), dtype=np.float32)
    dof_vel = np.empty((n_steps, env.num_envs, env.num_dof), dtype=np.float32)
    grf = np.empty((n_steps, env.num_envs, nfeet), dtype=np.float32)
    with torch.no_grad():
        for k in range(n_steps):
            env.dof_pos_target[:] = 0.0  # hold default pose
            env.step()
            base_pos[k] = env.root_states[:, :3].detach().cpu().numpy()
            base_quat[k] = env.root_states[:, 3:7].detach().cpu().numpy()
            base_lin_vel[k] = env.root_states[:, 7:10].detach().cpu().numpy()
            base_ang_vel[k] = env.root_states[:, 10:13].detach().cpu().numpy()
            dof_pos[k] = env.dof_pos.detach().cpu().numpy()
            dof_vel[k] = env.dof_vel.detach().cpu().numpy()
            f = torch.norm(env.contact_forces[:, env.feet_indices, :], dim=-1)
            grf[k] = f.detach().cpu().numpy()
    feet = env.feet_indices.detach().cpu().tolist()
    foot_names = [env._backend.body_names[i] for i in feet]
    return {
        "base_pos": base_pos,
        "base_z": base_pos[..., 2],
        "base_quat": base_quat,
        "base_lin_vel": base_lin_vel,
        "base_ang_vel": base_ang_vel,
        "dof_pos": dof_pos,
        "dof_vel": dof_vel,
        "dof_names": np.asarray(env.dof_names),
        "grf": grf,
        "foot_names": np.asarray(foot_names),
        "grf_names": np.asarray(foot_names),
    }


def run_step(env, n_steps, settle, deltas):
    """Fixed base: hold default, then step every joint by a per-env delta."""
    nd = env.num_dof
    dof_pos = np.empty((n_steps, env.num_envs, nd), dtype=np.float32)
    d = torch.tensor(deltas, dtype=torch.float, device=env.device)
    with torch.no_grad():
        for k in range(n_steps):
            if k < settle:
                env.dof_pos_target[:] = 0.0
            else:
                env.dof_pos_target[:] = d[:, None]
            env.step()
            dof_pos[k] = env.dof_pos.detach().cpu().numpy()
    return {
        "dof_pos": dof_pos,
        "deltas": np.asarray(deltas, dtype=np.float32),
        "settle": settle,
        "default_dof_pos": env.default_dof_pos.detach().cpu().numpy().ravel(),
        "dof_names": np.array(env.dof_names),
    }


def _assert_one_env_per_dof(env):
    if env.num_envs != env.num_dof:
        raise ValueError(
            "this probe requires one environment per canonical DOF: "
            f"got {env.num_envs} environments and {env.num_dof} DOFs"
        )


def _set_dof_state(env, dof_pos, dof_vel):
    """Write a canonical joint state and commit it through the backend API."""
    env_ids = torch.arange(env.num_envs, device=env.device)
    env.dof_pos.copy_(dof_pos)
    env.dof_vel.copy_(dof_vel)
    env._backend.reset_dof_state(env_ids)


def _quat_multiply_xyzw(left, right):
    """Hamilton product for scalar-last quaternions."""
    left_xyz, left_w = left[..., :3], left[..., 3:4]
    right_xyz, right_w = right[..., :3], right[..., 3:4]
    xyz = (
        left_w * right_xyz
        + right_w * left_xyz
        + torch.cross(left_xyz, right_xyz, dim=-1)
    )
    w = left_w * right_w - torch.sum(left_xyz * right_xyz, dim=-1, keepdim=True)
    return torch.cat((xyz, w), dim=-1)


def _quat_rotate_xyzw(quat, vector):
    """Rotate vectors by scalar-last quaternions."""
    q_xyz = quat[..., :3]
    twice_cross = 2.0 * torch.cross(q_xyz, vector, dim=-1)
    return (
        vector + quat[..., 3:4] * twice_cross + torch.cross(q_xyz, twice_cross, dim=-1)
    )


def run_torque_response(env, torque_nm):
    """One physics step per one-hot torque; rows identify excited joints."""
    _assert_one_env_per_dof(env)
    nd = env.num_dof
    default = env.default_dof_pos.expand(nd, -1)
    zeros = torch.zeros_like(default)
    _set_dof_state(env, default, zeros)

    initial_pos = env.dof_pos.clone()
    initial_vel = env.dof_vel.clone()
    torques = torch.eye(nd, device=env.device) * torque_nm
    with torch.no_grad():
        env._backend.step(torques)

    dt = float(env.cfg.sim_dt)
    return {
        "torque_nm": np.float64(torque_nm),
        "sim_dt": np.float64(dt),
        "joint_accel": ((env.dof_vel - initial_vel) / dt).cpu().numpy(),
        "joint_pos_delta": (env.dof_pos - initial_pos).cpu().numpy(),
        "dof_names": np.asarray(env.dof_names),
    }


def run_damping_decay(env, n_steps, initial_speed):
    """Zero-torque decay from one-hot canonical joint velocities."""
    _assert_one_env_per_dof(env)
    nd = env.num_dof
    default = env.default_dof_pos.expand(nd, -1)
    initial_vel = torch.eye(nd, device=env.device) * initial_speed
    _set_dof_state(env, default, initial_vel)

    dof_pos = np.empty((n_steps + 1, nd, nd), dtype=np.float32)
    dof_vel = np.empty_like(dof_pos)
    dof_pos[0] = env.dof_pos.cpu().numpy()
    dof_vel[0] = env.dof_vel.cpu().numpy()
    zero_torque = torch.zeros(nd, nd, device=env.device)
    with torch.no_grad():
        for k in range(1, n_steps + 1):
            env._backend.step(zero_torque)
            dof_pos[k] = env.dof_pos.cpu().numpy()
            dof_vel[k] = env.dof_vel.cpu().numpy()

    return {
        "time": np.arange(n_steps + 1, dtype=np.float64) * float(env.cfg.sim_dt),
        "initial_speed": np.float64(initial_speed),
        "dof_pos": dof_pos,
        "dof_vel": dof_vel,
        "dof_names": np.asarray(env.dof_names),
    }


def run_kinematics(env, pose_offset):
    """Compare canonical body transforms at a small basis of joint poses."""
    expected_envs = env.num_dof + 1
    if env.num_envs != expected_envs:
        raise ValueError(
            "kinematics requires the default pose plus one pose per DOF: "
            f"got {env.num_envs} environments, expected {expected_envs}"
        )

    nd = env.num_dof
    targets = env.default_dof_pos.expand(expected_envs, -1).clone()
    targets[1:] += torch.eye(nd, device=env.device) * pose_offset
    _set_dof_state(env, targets, torch.zeros_like(targets))

    # A zero-force step gives every backend the same public refresh boundary.
    with torch.no_grad():
        env._backend.step(torch.zeros(expected_envs, nd, device=env.device))

    rbs = env._backend.rigid_body_states.view(
        expected_envs, env._backend.num_bodies, 13
    )
    base_index = env._backend.find_body_index(env._backend.body_names[0])
    base_pos = rbs[:, base_index, 0:3]
    base_quat = rbs[:, base_index, 3:7]
    base_quat_inv = base_quat.clone()
    base_quat_inv[:, :3] *= -1.0

    inverse_for_bodies = base_quat_inv[:, None, :].expand(
        -1, env._backend.num_bodies, -1
    )
    relative_pos = _quat_rotate_xyzw(
        inverse_for_bodies,
        rbs[..., 0:3] - base_pos[:, None, :],
    )
    relative_quat = _quat_multiply_xyzw(inverse_for_bodies, rbs[..., 3:7])

    return {
        "pose_offset": np.float64(pose_offset),
        "target_dof_pos": targets.cpu().numpy(),
        "actual_dof_pos": env.dof_pos.cpu().numpy(),
        "body_pos_relative": relative_pos.cpu().numpy(),
        "body_quat_relative": relative_quat.cpu().numpy(),
        "dof_names": np.asarray(env.dof_names),
        "body_names": np.asarray(env._backend.body_names),
    }


def run_floating_reaction(env, n_steps, torque_nm):
    """Floating-base response to a constant one-hot torque on each joint."""
    _assert_one_env_per_dof(env)
    nd = env.num_dof
    default = env.default_dof_pos.expand(nd, -1)
    _set_dof_state(env, default, torch.zeros_like(default))

    base_pos = np.empty((n_steps + 1, nd, 3), dtype=np.float32)
    base_quat = np.empty((n_steps + 1, nd, 4), dtype=np.float32)
    base_lin_vel = np.empty_like(base_pos)
    base_ang_vel = np.empty_like(base_pos)
    dof_pos = np.empty((n_steps + 1, nd, nd), dtype=np.float32)
    dof_vel = np.empty_like(dof_pos)

    def record(k):
        base_pos[k] = env.root_states[:, 0:3].cpu().numpy()
        base_quat[k] = env.root_states[:, 3:7].cpu().numpy()
        base_lin_vel[k] = env.root_states[:, 7:10].cpu().numpy()
        base_ang_vel[k] = env.root_states[:, 10:13].cpu().numpy()
        dof_pos[k] = env.dof_pos.cpu().numpy()
        dof_vel[k] = env.dof_vel.cpu().numpy()

    record(0)
    torques = torch.eye(nd, device=env.device) * torque_nm
    with torch.no_grad():
        for k in range(1, n_steps + 1):
            env._backend.step(torques)
            record(k)

    return {
        "time": np.arange(n_steps + 1, dtype=np.float64) * float(env.cfg.sim_dt),
        "torque_nm": np.float64(torque_nm),
        "base_pos_delta": base_pos - base_pos[0:1],
        "base_quat": base_quat,
        "base_lin_vel": base_lin_vel,
        "base_ang_vel": base_ang_vel,
        "dof_pos": dof_pos,
        "dof_vel": dof_vel,
        "dof_names": np.asarray(env.dof_names),
    }


def _default_pose_pd_torques(env):
    """Full canonical-DOF torque vector for a default-pose PD hold."""
    actuated = env.actuated_dof_indices
    pos = env.dof_pos.index_select(1, actuated)
    vel = env.dof_vel.index_select(1, actuated)
    default = env.default_dof_pos.index_select(1, actuated)
    actuator_torques = env.p_gains * (default - pos) - env.d_gains * vel
    actuator_torques = torch.clip(
        actuator_torques,
        -env.actuated_torque_limits,
        env.actuated_torque_limits,
    )
    torques = torch.zeros(env.num_envs, env.num_dof, device=env.device)
    torques[:, actuated] = actuator_torques
    return torques


def _canonical_body_inertial_properties(env):
    """Return canonical body masses and body-frame COM offsets from the URDF."""
    asset_path = env.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    links = {
        link.get("name"): link
        for link in ET.parse(asset_path).getroot().findall("link")
    }
    masses = []
    local_com = []
    for name in env._backend.body_names:
        inertial = links[name].find("inertial")
        if inertial is None:
            masses.append(0.0)
            local_com.append([0.0, 0.0, 0.0])
            continue
        mass = inertial.find("mass")
        origin = inertial.find("origin")
        masses.append(float(mass.get("value")) if mass is not None else 0.0)
        xyz = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
        local_com.append([float(value) for value in xyz.split()])
    return (
        torch.tensor(masses, device=env.device),
        torch.tensor(local_com, device=env.device),
    )


def _contact_trace_buffers(env, n_steps):
    nfeet = len(env.feet_indices)
    body_mass, body_local_com = _canonical_body_inertial_properties(env)
    return {
        "_body_mass": body_mass,
        "_body_local_com": body_local_com,
        "base_pos": np.empty((n_steps + 1, env.num_envs, 3), dtype=np.float32),
        "base_quat": np.empty((n_steps + 1, env.num_envs, 4), dtype=np.float32),
        "base_lin_vel": np.empty((n_steps + 1, env.num_envs, 3), dtype=np.float32),
        "base_ang_vel": np.empty((n_steps + 1, env.num_envs, 3), dtype=np.float32),
        "dof_pos": np.empty((n_steps + 1, env.num_envs, env.num_dof), dtype=np.float32),
        "dof_vel": np.empty((n_steps + 1, env.num_envs, env.num_dof), dtype=np.float32),
        "foot_pos": np.empty((n_steps + 1, env.num_envs, nfeet, 3), dtype=np.float32),
        "foot_quat": np.empty((n_steps + 1, env.num_envs, nfeet, 4), dtype=np.float32),
        "system_com_pos": np.empty((n_steps + 1, env.num_envs, 3), dtype=np.float32),
        "contact_force": np.empty(
            (n_steps + 1, env.num_envs, nfeet, 3), dtype=np.float32
        ),
    }


def _record_contact_trace(env, buffers, index):
    rbs = env._backend.rigid_body_states.view(env.num_envs, env._backend.num_bodies, 13)
    buffers["base_pos"][index] = env.root_states[:, 0:3].cpu().numpy()
    buffers["base_quat"][index] = env.root_states[:, 3:7].cpu().numpy()
    buffers["base_lin_vel"][index] = env.root_states[:, 7:10].cpu().numpy()
    buffers["base_ang_vel"][index] = env.root_states[:, 10:13].cpu().numpy()
    buffers["dof_pos"][index] = env.dof_pos.cpu().numpy()
    buffers["dof_vel"][index] = env.dof_vel.cpu().numpy()
    buffers["foot_pos"][index] = rbs[:, env.feet_indices, 0:3].cpu().numpy()
    buffers["foot_quat"][index] = rbs[:, env.feet_indices, 3:7].cpu().numpy()
    local_com = buffers["_body_local_com"].unsqueeze(0).expand(env.num_envs, -1, -1)
    body_com = rbs[..., 0:3] + _quat_rotate_xyzw(rbs[..., 3:7], local_com)
    body_mass = buffers["_body_mass"]
    system_com = torch.sum(body_com * body_mass[None, :, None], dim=1) / body_mass.sum()
    buffers["system_com_pos"][index] = system_com.cpu().numpy()
    buffers["contact_force"][index] = (
        env.contact_forces[:, env.feet_indices, :].cpu().numpy()
    )


def _finish_contact_trace(env, buffers, n_steps):
    foot_names = np.asarray(
        [env._backend.body_names[index] for index in env.feet_indices.cpu().tolist()]
    )
    buffers.pop("_body_mass")
    buffers.pop("_body_local_com")
    buffers.update(
        {
            "time": np.arange(n_steps + 1, dtype=np.float64) * float(env.cfg.sim_dt),
            "sim_dt": np.float64(env.cfg.sim_dt),
            "dof_names": np.asarray(env.dof_names),
            "foot_names": foot_names,
        }
    )
    return buffers


def run_impact(env, n_steps):
    """High-rate vertical drop through impact and early settling."""
    buffers = _contact_trace_buffers(env, n_steps)
    _record_contact_trace(env, buffers, 0)
    with torch.no_grad():
        for k in range(1, n_steps + 1):
            env._backend.step(_default_pose_pd_torques(env))
            _record_contact_trace(env, buffers, k)
    return _finish_contact_trace(env, buffers, n_steps)


def run_slide(env, n_steps, settle_steps, initial_speed):
    """Settle, inject horizontal velocity, then record frictional slowdown."""
    with torch.no_grad():
        for _ in range(settle_steps):
            env._backend.step(_default_pose_pd_torques(env))

    env_ids = torch.arange(env.num_envs, device=env.device)
    env.dof_vel.zero_()
    env.root_states[:, 7:13] = 0.0
    env.root_states[:, 7] = initial_speed
    env._backend.reset_dof_state(env_ids)
    env._backend.reset_root_state(env_ids)

    buffers = _contact_trace_buffers(env, n_steps)
    _record_contact_trace(env, buffers, 0)
    with torch.no_grad():
        for k in range(1, n_steps + 1):
            env._backend.step(_default_pose_pd_torques(env))
            _record_contact_trace(env, buffers, k)
    buffers["initial_speed"] = np.float64(initial_speed)
    buffers["settle_time"] = np.float64(settle_steps * float(env.cfg.sim_dt))
    return _finish_contact_trace(env, buffers, n_steps)


def run(args):
    hz_key = "ctrl_frequency"
    environment_overrides = {
        "terrain_properties": {
            "static_friction": args.static_friction,
            "dynamic_friction": args.dynamic_friction,
            "restitution": args.restitution,
        },
        "vsim_properties": {
            "solver_iterations": args.vsim_solver_iterations,
            "contact_offset": args.vsim_contact_offset,
            "rest_offset": args.vsim_rest_offset,
            "contact_stiffness": args.vsim_contact_stiffness,
            "contact_damping": args.vsim_contact_damping,
        },
    }
    duration = args.t_end
    if duration is None:
        duration = {
            "damping": 0.25,
            "reaction": 0.10,
            "torque": 0.01,
            "kinematics": 0.01,
            "impact": 1.00,
            "slide": 1.00,
        }.get(args.probe, 3.0)

    if args.probe == "drop":
        env = build_env(
            args.backend,
            args.device,
            args.num_envs,
            fixed_base=False,
            base_z=0.5,
            t_end=duration,
            **environment_overrides,
        )
        n_steps = int(duration * float(getattr(env.cfg.control, hz_key)))
        reset_probe_state(env)
        data = run_drop(env, n_steps)
    elif args.probe == "step":
        env = build_env(
            args.backend,
            args.device,
            args.num_envs,
            fixed_base=True,
            base_z=1.0,
            t_end=duration,
            **environment_overrides,
        )
        n_steps = int(duration * float(getattr(env.cfg.control, hz_key)))
        settle = int(0.5 * float(getattr(env.cfg.control, hz_key)))
        deltas = np.linspace(-0.3, 0.3, args.num_envs)
        reset_probe_state(env)
        data = run_step(env, n_steps, settle, deltas)
    elif args.probe in NONCONTACT_PROBES:
        nd = len(MINI_CHEETAH_DOF_NAMES)
        num_envs = nd + 1 if args.probe == "kinematics" else nd
        env = build_env(
            args.backend,
            args.device,
            num_envs,
            fixed_base=args.probe != "reaction",
            base_z=1.0,
            t_end=duration,
            disable_gravity=True,
            with_ground=False,
            **environment_overrides,
        )
        reset_probe_state(env)
        if args.probe == "torque":
            data = run_torque_response(env, args.torque_nm)
        elif args.probe == "damping":
            n_steps = max(1, int(round(duration / float(env.cfg.sim_dt))))
            data = run_damping_decay(env, n_steps, args.initial_speed)
        elif args.probe == "kinematics":
            data = run_kinematics(env, args.pose_offset)
        elif args.probe == "reaction":
            n_steps = max(1, int(round(duration / float(env.cfg.sim_dt))))
            data = run_floating_reaction(env, n_steps, args.torque_nm)
        else:
            raise ValueError(f"unknown probe {args.probe!r}")
    elif args.probe in CONTACT_PROBES:
        env = build_env(
            args.backend,
            args.device,
            args.num_envs,
            fixed_base=False,
            base_z=0.5,
            t_end=duration + args.settle_time,
            **environment_overrides,
        )
        reset_probe_state(env)
        n_steps = max(1, int(round(duration / float(env.cfg.sim_dt))))
        if args.probe == "impact":
            data = run_impact(env, n_steps)
        else:
            settle_steps = max(1, int(round(args.settle_time / float(env.cfg.sim_dt))))
            data = run_slide(env, n_steps, settle_steps, args.initial_speed)
    else:
        raise ValueError(f"unknown probe {args.probe!r}")

    default_label = args.backend if args.backend == "vsim" else f"mujoco-{args.device}"
    label = args.label or default_label
    env._backend.close()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        probe=args.probe,
        label=label,
        ctrl_hz=float(getattr(env.cfg.control, hz_key)),
        duration=np.float64(duration),
        **data,
    )
    if args.probe == "drop":
        print(
            f"[{label}] drop: final base_z mean {data['base_z'][-1].mean():.4f} "
            f"| peak grf {data['grf'].max():.1f} N | wrote {args.out}"
        )
    elif args.probe == "step":
        settled = data["dof_pos"][-1]  # [N, nd]
        print(
            f"[{label}] step: final joint spread "
            f"{settled.std(0).mean():.4f} rad | wrote {args.out}"
        )
    elif args.probe == "torque":
        diagonal = np.diag(data["joint_accel"])
        print(
            f"[{label}] torque: mean direct acceleration "
            f"{np.mean(np.abs(diagonal)):.2f} rad/s² | wrote {args.out}"
        )
    elif args.probe == "damping":
        diagonal = np.diag(data["dof_vel"][-1])
        print(
            f"[{label}] damping: mean final direct speed "
            f"{np.mean(np.abs(diagonal)):.4f} rad/s | wrote {args.out}"
        )
    elif args.probe == "kinematics":
        target_error = np.max(np.abs(data["actual_dof_pos"] - data["target_dof_pos"]))
        print(
            f"[{label}] kinematics: max target drift "
            f"{target_error:.2e} rad | wrote {args.out}"
        )
    elif args.probe == "reaction":
        final_speed = np.linalg.norm(data["base_ang_vel"][-1], axis=-1).mean()
        print(
            f"[{label}] reaction: mean final base angular speed "
            f"{final_speed:.4f} rad/s | wrote {args.out}"
        )
    elif args.probe == "impact":
        force = np.sum(data["contact_force"][..., 2], axis=-1).mean(axis=1)
        contacts = np.flatnonzero(force > 1.0)
        impact_time = data["time"][contacts[0]] if len(contacts) else np.nan
        print(
            f"[{label}] impact: first force at {impact_time:.4f} s "
            f"| peak normal force {force.max():.1f} N | wrote {args.out}"
        )
    elif args.probe == "slide":
        com_x = data["system_com_pos"][..., 0].mean(axis=1)
        speed = np.gradient(com_x.astype(np.float64), data["time"])
        stopped = np.flatnonzero(speed[2:] <= 0.0)
        stop_time = data["time"][stopped[0] + 2] if len(stopped) else np.nan
        print(f"[{label}] slide: first stop at {stop_time:.4f} s | wrote {args.out}")


def compare(args):
    data = {}
    for path in args.files:
        d = np.load(path, allow_pickle=True)
        data[str(d["label"])] = d
    ref_label = next((k for k in data if k.startswith("mujoco-cpu")), list(data)[0])
    ref = data[ref_label]
    probe = str(ref["probe"])
    mismatched_probes = {
        label: str(result["probe"])
        for label, result in data.items()
        if str(result["probe"]) != probe
    }
    if mismatched_probes:
        raise ValueError(
            f"cannot compare {probe!r} with other probe types: {mismatched_probes}"
        )

    def quaternion_angle_error(left, right):
        # Float32 dot products near one lose enough precision to report a
        # spurious ~1e-3 rad angle for component differences around 1e-7.
        left = left.astype(np.float64)
        right = right.astype(np.float64)
        left = left / np.linalg.norm(left, axis=-1, keepdims=True)
        right = right / np.linalg.norm(right, axis=-1, keepdims=True)
        dot = np.abs(np.sum(left * right, axis=-1))
        return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))

    def mean_total_normal_force(result):
        return np.sum(result["contact_force"][..., 2], axis=-1).mean(axis=1)

    def first_contact_index(result, threshold=1.0):
        contacts = np.flatnonzero(mean_total_normal_force(result) > threshold)
        return int(contacts[0]) if len(contacts) else None

    def impact_metrics(result):
        force = mean_total_normal_force(result)
        onset = first_contact_index(result)
        if onset is None:
            return (np.nan,) * 6
        dt = float(result["sim_dt"])
        impulse_end = min(len(force), onset + int(round(0.1 / dt)) + 1)
        rebound_end = min(len(force), onset + int(round(0.25 / dt)) + 1)
        foot_quat = result["foot_quat"][onset].astype(np.float64)
        sphere_center = np.broadcast_to(
            FOOT_SPHERE_LOCAL_CENTER, foot_quat.shape[:-1] + (3,)
        )
        q_xyz = foot_quat[..., :3]
        twice_cross = 2.0 * np.cross(q_xyz, sphere_center)
        rotated_center = (
            sphere_center
            + foot_quat[..., 3:4] * twice_cross
            + np.cross(q_xyz, twice_cross)
        )
        sphere_center_z = result["foot_pos"][onset, ..., 2] + rotated_center[..., 2]
        clearance = sphere_center_z - FOOT_SPHERE_RADIUS
        settle_samples = max(1, int(round(0.1 / dt)))
        return (
            float(result["time"][onset]),
            float(clearance.min()),
            float(force.max()),
            float(force[onset:impulse_end].sum() * dt),
            float(result["base_lin_vel"][onset:rebound_end, ..., 2].max()),
            float(result["base_pos"][-settle_samples:, ..., 2].mean()),
        )

    def slide_metrics(result):
        time = result["time"]
        com_x = result["system_com_pos"][..., 0].mean(axis=1)
        vx = np.gradient(com_x.astype(np.float64), time)
        initial_speed = abs(float(result["initial_speed"]))
        # The first two samples avoid MuJoCo's one-refresh position lag at launch.
        measurement_start = 2
        half = np.flatnonzero(np.abs(vx[measurement_start:]) <= 0.5 * initial_speed)
        half_index = int(half[0] + measurement_start) if len(half) else None
        stopped = np.flatnonzero(vx[measurement_start:] <= 0.0)
        stop_index = (
            int(stopped[0] + measurement_start) if len(stopped) else len(time) - 1
        )
        half_time = float(time[half_index]) if half_index is not None else np.nan
        stop_time = float(time[stop_index]) if len(stopped) else np.nan
        com_distance = com_x[stop_index] - com_x[0]
        foot_delta = (
            result["foot_pos"][stop_index, ..., 0:2] - result["foot_pos"][0, ..., 0:2]
        )
        foot_slip = np.linalg.norm(foot_delta, axis=-1).mean()
        normal_force = mean_total_normal_force(result)
        return (
            half_time,
            stop_time,
            float(com_distance),
            float(foot_slip),
            float(normal_force[: stop_index + 1].mean()),
        )

    def aligned_grf(d):
        ref_names = [str(name) for name in ref.get("grf_names", ref["foot_names"])]
        names = [str(name) for name in d.get("grf_names", d["foot_names"])]
        return d["grf"][..., [names.index(name) for name in ref_names]]

    print(f"\nprobe: {probe}  ·  reference: {ref_label}")
    if probe == "drop":
        print(
            f"{'engine':<16}{'base-z RMS':>14}{'quat RMS':>12}"
            f"{'grf-time RMS':>14}{'joint-q RMS':>14}{'joint-qd RMS':>15}"
        )
        for label, d in data.items():
            if label == ref_label:
                print(f"{label:<16}{'—':>14}{'—':>12}{'—':>14}{'—':>14}{'—':>15}")
                continue
            zr = np.sqrt(np.mean((d["base_z"] - ref["base_z"]) ** 2))
            qr = np.sqrt(np.mean((d["base_quat"] - ref["base_quat"]) ** 2))
            gr = np.sqrt(np.mean((aligned_grf(d) - aligned_grf(ref)) ** 2))
            q = np.sqrt(np.mean((d["dof_pos"] - ref["dof_pos"]) ** 2))
            qd = np.sqrt(np.mean((d["dof_vel"] - ref["dof_vel"]) ** 2))
            print(f"{label:<16}{zr:>14.2e}{qr:>12.2e}{gr:>14.2e}{q:>14.2e}{qd:>15.2e}")
    elif probe == "step":
        print(f"{'engine':<16}{'joint-traj RMS':>16}{'final-pos RMS':>15}")
        for label, d in data.items():
            if label == ref_label:
                print(f"{label:<16}{'—':>16}{'—':>15}")
                continue
            tr = np.sqrt(np.mean((d["dof_pos"] - ref["dof_pos"]) ** 2))
            fr = np.sqrt(np.mean((d["dof_pos"][-1] - ref["dof_pos"][-1]) ** 2))
            print(f"{label:<16}{tr:>16.2e}{fr:>15.2e}")
    elif probe == "torque":
        print(
            f"{'engine':<16}{'accel RMS':>16}{'accel rel':>13}"
            f"{'accel max':>14}{'q-step RMS':>14}"
        )
        ref_scale = np.sqrt(np.mean(ref["joint_accel"] ** 2))
        for label, d in data.items():
            if label == ref_label:
                print(f"{label:<16}{'—':>16}{'—':>13}{'—':>14}{'—':>14}")
                continue
            delta_accel = d["joint_accel"] - ref["joint_accel"]
            accel_rms = np.sqrt(np.mean(delta_accel**2))
            accel_max = np.max(np.abs(delta_accel))
            q_rms = np.sqrt(
                np.mean((d["joint_pos_delta"] - ref["joint_pos_delta"]) ** 2)
            )
            print(
                f"{label:<16}{accel_rms:>16.2e}"
                f"{accel_rms / ref_scale:>12.2%}{accel_max:>14.2e}{q_rms:>14.2e}"
            )
    elif probe == "damping":
        print(
            f"{'engine':<16}{'q-traj RMS':>16}{'qd-traj RMS':>16}{'final qd RMS':>16}"
        )
        for label, d in data.items():
            if label == ref_label:
                print(f"{label:<16}{'—':>16}{'—':>16}{'—':>16}")
                continue
            q_rms = np.sqrt(np.mean((d["dof_pos"] - ref["dof_pos"]) ** 2))
            qd_rms = np.sqrt(np.mean((d["dof_vel"] - ref["dof_vel"]) ** 2))
            final_qd_rms = np.sqrt(
                np.mean((d["dof_vel"][-1] - ref["dof_vel"][-1]) ** 2)
            )
            print(f"{label:<16}{q_rms:>16.2e}{qd_rms:>16.2e}{final_qd_rms:>16.2e}")
    elif probe == "kinematics":
        print(
            f"{'engine':<16}{'body-pos RMS':>16}{'body-pos max':>16}"
            f"{'body-angle RMS':>18}{'joint drift':>14}"
        )
        for label, d in data.items():
            if label == ref_label:
                print(f"{label:<16}{'—':>16}{'—':>16}{'—':>18}{'—':>14}")
                continue
            pos_delta = d["body_pos_relative"] - ref["body_pos_relative"]
            angle = quaternion_angle_error(
                d["body_quat_relative"], ref["body_quat_relative"]
            )
            joint_drift = np.max(np.abs(d["actual_dof_pos"] - d["target_dof_pos"]))
            print(
                f"{label:<16}{np.sqrt(np.mean(pos_delta**2)):>16.2e}"
                f"{np.max(np.abs(pos_delta)):>16.2e}"
                f"{np.sqrt(np.mean(angle**2)):>18.2e}{joint_drift:>14.2e}"
            )
    elif probe == "reaction":
        print(
            f"{'engine':<16}{'base-pos RMS':>16}{'base-angle RMS':>18}"
            f"{'base-v RMS':>14}{'base-w RMS':>14}{'joint-qd RMS':>16}"
        )
        for label, d in data.items():
            if label == ref_label:
                print(f"{label:<16}{'—':>16}{'—':>18}{'—':>14}{'—':>14}{'—':>16}")
                continue
            pos = np.sqrt(np.mean((d["base_pos_delta"] - ref["base_pos_delta"]) ** 2))
            angle = quaternion_angle_error(d["base_quat"], ref["base_quat"])
            lin_vel = np.sqrt(np.mean((d["base_lin_vel"] - ref["base_lin_vel"]) ** 2))
            ang_vel = np.sqrt(np.mean((d["base_ang_vel"] - ref["base_ang_vel"]) ** 2))
            dof_vel = np.sqrt(np.mean((d["dof_vel"] - ref["dof_vel"]) ** 2))
            print(
                f"{label:<16}{pos:>16.2e}{np.sqrt(np.mean(angle**2)):>18.2e}"
                f"{lin_vel:>14.2e}{ang_vel:>14.2e}{dof_vel:>16.2e}"
            )
    elif probe == "impact":
        print(
            f"{'engine':<16}{'impact':>10}{'onset gap':>13}{'peak Fz':>12}"
            f"{'100ms impulse':>16}{'rebound vz':>14}{'settle z':>12}"
        )
        print(
            f"{'':<16}{'[s]':>10}{'[mm]':>13}{'[N]':>12}"
            f"{'[N·s]':>16}{'[m/s]':>14}{'[m]':>12}"
        )
        for label, d in data.items():
            onset, gap, peak, impulse, rebound, settle = impact_metrics(d)
            print(
                f"{label:<16}{onset:>10.4f}{gap * 1e3:>13.3f}{peak:>12.1f}"
                f"{impulse:>16.3f}{rebound:>14.3f}{settle:>12.4f}"
            )
    elif probe == "slide":
        print(
            f"{'engine':<16}{'half-time':>12}{'stop-time':>12}"
            f"{'stop dx':>12}{'foot slip':>13}{'mean Fz':>12}"
        )
        print(f"{'':<16}{'[s]':>12}{'[s]':>12}{'[m]':>12}{'[m]':>13}{'[N]':>12}")
        for label, d in data.items():
            half_time, stop_time, stop_dx, foot_slip, normal_force = slide_metrics(d)
            print(
                f"{label:<16}{half_time:>12.4f}{stop_time:>12.4f}"
                f"{stop_dx:>12.4f}{foot_slip:>13.4f}{normal_force:>12.1f}"
            )
    else:
        raise ValueError(f"unsupported probe {probe!r}")

    if probe in ("drop", "step"):
        print(
            "\ndrop RMS is contact-model divergence (expect vsim >> warp); step RMS "
            "is contact-free PD/limb divergence."
        )
    elif probe in NONCONTACT_PROBES:
        print(
            "\nAll values are differences from MuJoCo CPU. These probes bypass task "
            "PD control and contain no gravity or contact."
        )
    else:
        print("\nImpact and slide report absolute behavior at the 500 Hz physics rate.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument(
        "--probe",
        choices=["drop", "step", *NONCONTACT_PROBES, *CONTACT_PROBES],
        required=True,
    )
    r.add_argument("--backend", choices=["mujoco", "vsim"], default="mujoco")
    r.add_argument("--device", default="cpu")
    r.add_argument(
        "--num_envs",
        type=int,
        default=32,
        help="environment count for drop/step (basis probes choose 12 or 13)",
    )
    r.add_argument(
        "--t_end",
        type=float,
        default=None,
        help="probe duration (defaults: 3 s, damping 0.25 s, reaction 0.10 s)",
    )
    r.add_argument("--torque_nm", type=float, default=1.0)
    r.add_argument("--initial_speed", type=float, default=1.0)
    r.add_argument("--pose_offset", type=float, default=0.2)
    r.add_argument("--settle_time", type=float, default=1.0)
    r.add_argument("--static_friction", type=float)
    r.add_argument("--dynamic_friction", type=float)
    r.add_argument("--restitution", type=float)
    r.add_argument("--vsim_solver_iterations", type=int)
    r.add_argument("--vsim_contact_offset", type=float)
    r.add_argument("--vsim_rest_offset", type=float)
    r.add_argument("--vsim_contact_stiffness", type=float)
    r.add_argument("--vsim_contact_damping", type=float)
    r.add_argument("--label")
    r.add_argument("--out", required=True)
    r.set_defaults(func=run)
    c = sub.add_parser("compare")
    c.add_argument("files", nargs="+")
    c.set_defaults(func=compare)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
