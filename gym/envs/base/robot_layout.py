"""Canonical robot metadata shared by every physics backend.

Physics engines are free to compile links, joints, motors, and sensors in
different orders.  ``RobotLayout`` defines the task-facing order from the
source asset and resolves semantic groups to cached tensor indices.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence
import xml.etree.ElementTree as ET

from gym import LEGGED_GYM_ROOT_DIR


def _as_tuple(values: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(values)
    duplicates = sorted({name for name in result if result.count(name) > 1})
    if duplicates:
        raise ValueError(f"{label} contains duplicate names: {duplicates}")
    return result


def _config_mapping(config, name: str) -> dict[str, tuple[str, ...]]:
    values = getattr(config, name, {})
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        values = {
            key: getattr(values, key) for key in dir(values) if not key.startswith("_")
        }
    return {
        group_name: _as_tuple(group_values, f"{name}.{group_name}")
        for group_name, group_values in values.items()
    }


def _urdf_names(asset_path: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    root = ET.parse(asset_path).getroot()
    body_names = _as_tuple(
        [link.get("name") for link in root.findall("link")],
        "URDF link names",
    )
    dof_names = _as_tuple(
        [
            joint.get("name")
            for joint in root.findall("joint")
            if joint.get("type") != "fixed"
        ],
        "URDF movable-joint names",
    )
    return dof_names, body_names


def _validate_members(
    groups: Mapping[str, tuple[str, ...]],
    valid_names: tuple[str, ...],
    label: str,
) -> None:
    valid = set(valid_names)
    for group_name, members in groups.items():
        missing = sorted(set(members) - valid)
        if missing:
            raise ValueError(
                f"{label} group {group_name!r} has unknown names: {missing}"
            )


@dataclass(frozen=True)
class RobotLayout:
    """Stable task-facing robot order and named semantic groups."""

    version: str
    dof_names: tuple[str, ...]
    actuated_dof_names: tuple[str, ...]
    body_names: tuple[str, ...]
    dof_groups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    body_groups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def from_cfg(cls, cfg) -> "RobotLayout":
        asset_path = cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_path = str(Path(asset_path).resolve())
        urdf_dofs, urdf_bodies = _urdf_names(asset_path)
        config = getattr(cfg.asset, "robot_layout", None)
        control = getattr(cfg, "control", None)
        control_actuators = tuple(
            getattr(control, "actuated_joint_names", []) if control is not None else []
        )

        if config is None:
            version = f"urdf:{Path(asset_path).name}"
            dof_names = urdf_dofs
            body_names = urdf_bodies
            actuated_dof_names = control_actuators or dof_names
            dof_groups = {}
            body_groups = {}
        else:
            version = str(getattr(config, "version"))
            dof_names = _as_tuple(
                getattr(config, "dof_names", urdf_dofs),
                "canonical DOF names",
            )
            body_names = _as_tuple(
                getattr(config, "body_names", urdf_bodies),
                "canonical body names",
            )
            actuated_dof_names = _as_tuple(
                getattr(config, "actuated_dof_names", dof_names),
                "canonical actuated DOF names",
            )
            dof_groups = _config_mapping(config, "dof_groups")
            body_groups = _config_mapping(config, "body_groups")

        foot_pattern = getattr(cfg.asset, "foot_name", None)
        if foot_pattern and "feet" not in body_groups:
            feet = tuple(name for name in body_names if foot_pattern in name)
            if not feet:
                raise ValueError(
                    f"foot_name={foot_pattern!r} matched no canonical bodies"
                )
            body_groups["feet"] = feet

        if set(dof_names) != set(urdf_dofs):
            raise ValueError(
                "canonical DOFs must exactly match movable URDF joints: "
                f"canonical={list(dof_names)}, urdf={list(urdf_dofs)}"
            )
        if set(body_names) != set(urdf_bodies):
            raise ValueError(
                "canonical bodies must exactly match URDF links: "
                f"canonical={list(body_names)}, urdf={list(urdf_bodies)}"
            )

        missing_actuators = sorted(set(actuated_dof_names) - set(dof_names))
        if missing_actuators:
            raise ValueError(
                f"actuated DOFs are not in the canonical DOF list: {missing_actuators}"
            )
        _validate_members(dof_groups, dof_names, "DOF")
        _validate_members(body_groups, body_names, "body")

        return cls(
            version=version,
            dof_names=dof_names,
            actuated_dof_names=actuated_dof_names,
            body_names=body_names,
            dof_groups=dof_groups,
            body_groups=body_groups,
        )

    def validate_native(
        self,
        native_dof_names: Sequence[str],
        native_body_names: Sequence[str],
        allowed_extra_body_names: Sequence[str] = (),
    ) -> None:
        native_dofs = _as_tuple(native_dof_names, "native DOF names")
        native_bodies = _as_tuple(native_body_names, "native body names")
        if set(native_dofs) != set(self.dof_names):
            raise ValueError(
                "native/canonical DOF mismatch: "
                f"native={list(native_dofs)}, canonical={list(self.dof_names)}"
            )

        allowed_extra = set(allowed_extra_body_names)
        missing = sorted(set(self.body_names) - set(native_bodies))
        extra = sorted(set(native_bodies) - set(self.body_names) - allowed_extra)
        if missing or extra:
            raise ValueError(
                "native/canonical body mismatch: "
                f"missing={missing}, unexpected={extra}, "
                f"allowed_extra={sorted(allowed_extra)}"
            )

    def canonical_to_native_dof(self, native_names: Sequence[str]) -> list[int]:
        return [list(native_names).index(name) for name in self.dof_names]

    def native_to_canonical_dof(self, native_names: Sequence[str]) -> list[int]:
        return [self.dof_names.index(name) for name in native_names]

    def canonical_to_native_body(self, native_names: Sequence[str]) -> list[int]:
        return [list(native_names).index(name) for name in self.body_names]

    def dof_indices(self, names: Sequence[str]) -> tuple[int, ...]:
        return tuple(self.dof_names.index(name) for name in names)

    def body_indices(self, names: Sequence[str]) -> tuple[int, ...]:
        return tuple(self.body_names.index(name) for name in names)

    def dof_group_indices(self, group_name: str) -> tuple[int, ...]:
        if group_name not in self.dof_groups:
            raise KeyError(f"unknown DOF group {group_name!r}")
        return self.dof_indices(self.dof_groups[group_name])

    def body_group_indices(self, group_name: str) -> tuple[int, ...]:
        if group_name not in self.body_groups:
            raise KeyError(f"unknown body group {group_name!r}")
        return self.body_indices(self.body_groups[group_name])
