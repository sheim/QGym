"""Diagnostic tests for the canonical robot-layout boundary."""

from types import SimpleNamespace

import pytest
import torch

from gym.envs.base.robot_layout import RobotLayout
from gym.envs.base.vsim_backend import (
    _canonical_body_sensor_indices,
    _motor_sources,
)
from gym.envs.mini_cheetah.mini_cheetah_config import (
    MINI_CHEETAH_BODY_NAMES,
    MINI_CHEETAH_DOF_NAMES,
    MiniCheetahCfg,
)
from gym.envs.mini_cheetah.mini_cheetah_ref import MiniCheetahRef


def test_mini_cheetah_layout_v1_is_explicit():
    layout = RobotLayout.from_cfg(MiniCheetahCfg())

    assert layout.version == "mini_cheetah_v1"
    assert list(layout.dof_names) == MINI_CHEETAH_DOF_NAMES
    assert list(layout.body_names) == MINI_CHEETAH_BODY_NAMES
    assert layout.dof_group_indices("rf_leg") == (0, 1, 2)
    assert layout.dof_group_indices("abad") == (0, 3, 6, 9)
    assert layout.body_group_indices("feet") == (4, 8, 12, 16)


def test_backend_layout_is_non_optional(mock_backend):
    assert tuple(mock_backend.dof_names) == mock_backend.robot_layout.dof_names
    assert tuple(mock_backend.body_names) == mock_backend.robot_layout.body_names


def test_permuted_native_dof_round_trip():
    layout = RobotLayout.from_cfg(MiniCheetahCfg())
    native_names = list(reversed(layout.dof_names))
    canonical_to_native = layout.canonical_to_native_dof(native_names)
    native_to_canonical = layout.native_to_canonical_dof(native_names)

    native_read = torch.tensor(
        [[layout.dof_names.index(name) for name in native_names]]
    )
    canonical_read = native_read[:, canonical_to_native]
    assert canonical_read.tolist() == [list(range(len(layout.dof_names)))]

    canonical_write = torch.arange(len(layout.dof_names)).unsqueeze(0)
    native_write = canonical_write[:, native_to_canonical]
    assert native_write.tolist() == [list(reversed(range(len(layout.dof_names))))]


def test_layout_rejects_missing_native_name():
    layout = RobotLayout.from_cfg(MiniCheetahCfg())
    with pytest.raises(ValueError, match="native/canonical DOF mismatch"):
        layout.validate_native(layout.dof_names[:-1], layout.body_names)


def test_vsim_motor_and_sensor_routes_are_not_ordinal():
    # Canonical order: RF, LF, RH, LH. Native articulation order is reversed;
    # motors retain XML/canonical order.
    native_to_canonical_dof = [3, 2, 1, 0]
    motor_native_dofs = [3, 2, 1, 0]
    assert _motor_sources(native_to_canonical_dof, motor_native_dofs) == [
        0,
        1,
        2,
        3,
    ]

    # Canonical bodies map to native links [base, RF, LF, RH, LH]. Sensors may
    # have any declaration order; the result is canonical body -> sensor slot.
    canonical_to_native_body = [0, 4, 3, 2, 1]
    sensor_native_links = [4, 0, 2, 1, 3]
    assert _canonical_body_sensor_indices(
        canonical_to_native_body, sensor_native_links
    ) == [1, 0, 4, 2, 3]


def test_mujoco_exposes_robot_only_canonical_layout(legged_cpu_backend):
    backend = legged_cpu_backend
    assert backend.dof_names == MINI_CHEETAH_DOF_NAMES
    assert backend.body_names == MINI_CHEETAH_BODY_NAMES
    assert "world" not in backend.body_names
    assert backend.num_bodies == 17


def test_mujoco_named_torque_routing(legged_cpu_backend):
    backend = legged_cpu_backend
    canonical_torques = torch.arange(1, 13, dtype=torch.float).unsqueeze(0).repeat(4, 1)
    backend.step(canonical_torques)

    native_expected = canonical_torques[0].numpy()[backend._native_to_canonical_dof_np]
    for data in backend._datas:
        assert data.qfrc_applied[backend._qvel_offset :].tolist() == pytest.approx(
            native_expected.tolist()
        )


@pytest.mark.vsim
def test_vsim_exposes_canonical_layout(legged_vsim_backend):
    backend = legged_vsim_backend
    assert backend.dof_names == MINI_CHEETAH_DOF_NAMES
    assert backend.body_names == MINI_CHEETAH_BODY_NAMES
    assert backend._native_dof_names != backend.dof_names
    assert backend._native_body_names != backend.body_names


def test_reference_trajectory_uses_named_leg_groups():
    task = object.__new__(MiniCheetahRef)
    task.torques = torch.zeros(2, 12)
    task.phase = torch.tensor([[0.0], [torch.pi]])
    task.leg_ref = torch.stack(
        [
            torch.tensor([float(index), float(index + 1), float(index + 2)])
            for index in range(8)
        ]
    )
    task._reference_leg_indices = [
        torch.tensor([0, 1, 2]),
        torch.tensor([3, 4, 5]),
        torch.tensor([6, 7, 8]),
        torch.tensor([9, 10, 11]),
    ]
    task._gait_phase_offsets = torch.tensor([0.0, torch.pi, torch.pi, 0.0])

    reference = task._get_ref()

    assert torch.equal(reference[:, 0:3], reference[:, 9:12])
    assert torch.equal(reference[:, 3:6], reference[:, 6:9])
    assert not torch.equal(reference[:, 0:3], reference[:, 3:6])


def test_reference_trajectory_reaches_last_sample_and_wraps():
    task = object.__new__(MiniCheetahRef)
    task.torques = torch.zeros(2, 3)
    task.phase = torch.tensor([[2 * torch.pi * 7 / 8], [2 * torch.pi]])
    task.leg_ref = torch.arange(8, dtype=torch.float).unsqueeze(1).repeat(1, 3)
    task._reference_leg_indices = [torch.tensor([0, 1, 2])]
    task._gait_phase_offsets = torch.tensor([0.0])

    reference = task._get_ref()

    assert torch.equal(reference[0], torch.full((3,), 7.0))
    assert torch.equal(reference[1], torch.zeros(3))


def test_implicit_layout_uses_urdf_not_native_order():
    cfg = SimpleNamespace(
        asset=SimpleNamespace(
            file=MiniCheetahCfg().asset.file,
            foot_name="foot",
        )
    )
    layout = RobotLayout.from_cfg(cfg)
    assert list(layout.dof_names) == MINI_CHEETAH_DOF_NAMES
    assert layout.body_groups["feet"] == (
        "rf_foot",
        "lf_foot",
        "rh_foot",
        "lh_foot",
    )
