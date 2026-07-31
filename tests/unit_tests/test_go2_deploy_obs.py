"""Parity test for the Go2 hardware observation builder.

go2_deploy/utility.py must produce exactly the vector the policy was trained on.
The load-bearing test drives the go2 env, synthesises the LowState_ message the
robot would have sent for that state, and asserts the deploy pipeline reproduces
env.get_states() to float precision.  A wrong joint permutation, a swapped
quaternion, or a missed scale factor all fail here.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "go2_deploy"))

import utility  # noqa: E402


def _build_env():
    pytest.importorskip("mujoco")

    import gym.envs  # noqa: F401  — registers tasks
    from gym.utils.task_registry import task_registry

    env_cfg, train_cfg = task_registry.get_cfgs("go2")
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 9999
    env_cfg.push_robots.toggle = False
    env_cfg.seed = 0
    train_cfg.seed = 0
    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)

    return task_registry.make_env_mujoco("go2", env_cfg, device="cpu", headless=True)


def _synth_lowstate(env) -> types.SimpleNamespace:
    """Build the LowState_ the robot would send for the env's current state.

    The sim -> SDK reordering here is derived from joint *names* against
    env.dof_names, deliberately not from utility.LEG_PERM: reusing the module's
    own permutation would cancel a wrong permutation out and let the parity
    assertion pass on broken code.
    """
    sim_of_sdk = [list(env.dof_names).index(n) for n in utility.SDK_JOINT_NAMES]
    dof_pos = env.dof_pos[0].numpy()[sim_of_sdk]
    dof_vel = env.dof_vel[0].numpy()[sim_of_sdk]

    motor_state = [
        types.SimpleNamespace(q=dof_pos[i], dq=dof_vel[i], ddq=0.0, tau_est=0.0)
        for i in range(utility.NUM_DOF)
    ] + [types.SimpleNamespace(q=0.0, dq=0.0, ddq=0.0, tau_est=0.0)] * 8

    quat_xyzw = env.base_quat[0].numpy()
    imu_state = types.SimpleNamespace(
        quaternion=quat_xyzw[[3, 0, 1, 2]],  # the IMU reports (w, x, y, z)
        # MuJoCo's free-joint qvel[3:6] is body-frame, which is what the backend
        # stores in root_states[:, 10:13] and what a real gyro measures.
        gyroscope=env.root_states[0, 10:13].numpy(),
        accelerometer=np.zeros(3, dtype=np.float32),
        rpy=np.zeros(3, dtype=np.float32),
    )

    return types.SimpleNamespace(
        motor_state=motor_state,
        imu_state=imu_state,
        foot_force=np.zeros(4, dtype=np.float32),
        wireless_remote=bytes(40),
        tick=1234,
    )


def test_permutations_are_self_inverse():
    assert (utility.LEG_PERM[utility.LEG_PERM] == np.arange(utility.NUM_DOF)).all()
    assert (utility.FOOT_PERM[utility.FOOT_PERM] == np.arange(utility.NUM_FEET)).all()
    # LegID order: FR, FL, RR, RL
    assert [utility.JOINT_NAMES[i] for i in utility.LEG_PERM] == [
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    ]


def test_decode_lowstate_permutes_and_reorders_quaternion():
    """No env, no DDS stack — pure layout check with distinct per-joint values."""
    sdk_values = np.arange(utility.NUM_DOF, dtype=np.float32)
    msg = types.SimpleNamespace(
        motor_state=[
            types.SimpleNamespace(
                q=sdk_values[i], dq=-sdk_values[i], ddq=2 * sdk_values[i], tau_est=0.0
            )
            for i in range(utility.NUM_DOF)
        ],
        imu_state=types.SimpleNamespace(
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),  # w, x, y, z
            gyroscope=np.array([0.1, 0.2, 0.3]),
        ),
        foot_force=np.array([10.0, 20.0, 30.0, 40.0]),
        wireless_remote=bytes(40),
        tick=5000,
    )
    frame = utility.decode_lowstate(msg)

    # spelled out rather than expressed via LEG_PERM, so a wrong permutation
    # cannot cancel itself out
    expected = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.float32)
    assert (frame.dof_pos == expected).all()
    assert (frame.dof_vel == -expected).all()
    assert (frame.dof_acc == 2 * expected).all()
    assert (frame.quat_xyzw == np.array([0.0, 0.0, 0.0, 1.0])).all()
    assert (frame.foot_force == np.array([20.0, 10.0, 40.0, 30.0])).all()
    assert frame.tick_s == pytest.approx(5.0)


def test_constants_match_env():
    env = _build_env()
    assert utility.JOINT_NAMES == list(env.dof_names)
    assert np.allclose(utility.DEFAULT_DOF_POS, env.default_dof_pos[0].numpy())
    assert np.allclose(utility.P_GAINS, env.p_gains[0].numpy())
    assert np.allclose(utility.D_GAINS, env.d_gains[0].numpy())


def test_deploy_obs_matches_env_get_states():
    env = _build_env()
    builder = utility.ObsBuilder()
    scale = np.asarray(utility.Go2Cfg.scaling.dof_pos_target, dtype=np.float32)

    torch.manual_seed(0)
    for _ in range(50):
        action = 0.5 * torch.randn(1, utility.NUM_DOF)
        env.set_states(utility.ACTION_LIST, action)
        env.step()

        # The two quantities no message carries; the action round-trip is
        # covered separately in test_action_roundtrip.
        builder.last_action = env.dof_pos_target[0].numpy() / scale
        builder.commands = env.commands[0].numpy()

        deploy_obs = builder.get_obs(_synth_lowstate(env))
        sim_obs = env.get_states(utility.OBS_LIST)[0].numpy()

        assert deploy_obs.shape == sim_obs.shape == (45,)
        np.testing.assert_allclose(deploy_obs, sim_obs, atol=1e-6)


def test_action_roundtrip():
    builder = utility.ObsBuilder()
    action = np.linspace(-1.0, 1.0, utility.NUM_DOF).astype(np.float32)
    builder.commit_action(action)

    frame = utility.decode_lowstate(_zero_msg())
    observed = builder.get_state("dof_pos_target", frame)
    np.testing.assert_allclose(observed, action, atol=1e-7)

    # zero action must command the nominal pose, and the SDK permutation must
    # be undone by the same array
    q_des = utility.action_to_dof_pos_target(np.zeros(utility.NUM_DOF))
    np.testing.assert_allclose(q_des, utility.DEFAULT_DOF_POS, atol=1e-7)
    np.testing.assert_allclose(
        utility.to_sdk_order(utility.to_sdk_order(q_des)), q_des, atol=0
    )


def test_obs_sizes_cover_every_generator():
    builder = utility.ObsBuilder(obs_list=list(utility.ObsBuilder.OBS_SIZE))
    frame = utility.decode_lowstate(_zero_msg())
    for name, size in utility.ObsBuilder.OBS_SIZE.items():
        assert builder.get_state(name, frame).shape == (size,)
    assert builder.get_obs_from_frame(frame).shape == (builder.num_obs,)


def _zero_msg() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        motor_state=[
            types.SimpleNamespace(q=0.0, dq=0.0, ddq=0.0, tau_est=0.0)
            for _ in range(utility.NUM_DOF)
        ],
        imu_state=types.SimpleNamespace(
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            gyroscope=np.zeros(3),
        ),
        foot_force=np.zeros(4),
        wireless_remote=bytes(40),
        tick=0,
    )
