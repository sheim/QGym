"""URDF → .vsim asset pipeline for the vsim (vlearn) backend.

vsim's converter preserves joint names, types, and <limit> attributes but
emits NO <motor> elements, and its importer fuses fixed joints by default —
both verified 2026-07-12 (see q2-backend-integration skill).  This module
converts, then post-processes the .vsim XML:

- inject one gear=1.0 motor per movable joint (raw-Nm semantics, verified);
- inject a `flags="contact"` force sensor on EVERY link (the backend's
  contact_forces source; force is in sensor slots [0:3]);
- write joint damping / per-dof armature;
- fixed-base robots: strip <collision> blocks (contacts-disabled semantics);
- verify the result against the source URDF (names + effort/velocity) and
  fail fast on drift.

Assets are regenerated on every setup (conversion is sub-second and a cache
key would have to include cfg params); output lives in
resources/robots/<name>/vsim/ (gitignored) for inspection.
"""

import os
import xml.etree.ElementTree as ET

from gym import GYM_ROOT_DIR
from gym.envs.base.urdf_limits import parse_urdf_limits

UNLIMITED_EFFORT = 1.0e6


def movable_joints(root: ET.Element) -> list:
    return [j for j in root.findall("joint") if j.get("type") != "fixed"]


def inject_motors(root: ET.Element) -> None:
    """One gear=1.0 motor per movable joint; limits = ±effort from <limit>."""
    act = ET.SubElement(root, "actuator")
    for j in movable_joints(root):
        limit = j.find("limit")
        effort = float(limit.get("effort")) if limit is not None else 0.0
        if effort <= 0.0:
            effort = UNLIMITED_EFFORT
        ET.SubElement(
            act,
            "motor",
            name=f"{j.get('name')}_motor",
            joint=j.get("name"),
            gear="1.0",
            lowLimit=str(-effort),
            highLimit=str(effort),
        )


def inject_contact_sensors(root: ET.Element) -> None:
    """A contact force sensor on every link, named '<link>__cf'."""
    fs = ET.SubElement(root, "forceSensor")
    for link in root.findall("link"):
        ET.SubElement(
            fs,
            "sensor",
            name=f"{link.get('name')}__cf",
            link=link.get("name"),
            offset="0 0 0",
            flags="contact",
        )


def set_joint_dynamics(root: ET.Element, damping: float, armature) -> None:
    """Write damping and per-dof armature into each movable joint.

    armature may be a scalar or a per-dof sequence (XML joint order).
    """
    joints = movable_joints(root)
    for i, j in enumerate(joints):
        arm = armature[i] if hasattr(armature, "__len__") else armature
        dyn = j.find("dynamics")
        if dyn is None:
            dyn = ET.SubElement(j, "dynamics")
        dyn.set("damping", str(damping))
        dyn.set("armature", str(arm))


def absolutize_mesh_paths(root: ET.Element, urdf_dir: str) -> None:
    """Rewrite relative <mesh filename=...> refs to absolute paths.

    The converter copies the URDF's relative refs verbatim, but the .vsim
    lives in a sibling vsim/ directory, so they dangle at import time
    ("Failed to resolve resource" warnings; visuals fall back to collision
    shapes)."""
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename")
        if fn and not os.path.isabs(fn):
            mesh.set("filename", os.path.abspath(os.path.join(urdf_dir, fn)))


def verify_vsim_against_urdf(root: ET.Element, urdf_path: str) -> None:
    """Fail fast if the converter dropped or mutated joints/limits."""
    urdf_limits = parse_urdf_limits(urdf_path)
    urdf_root = ET.parse(urdf_path).getroot()
    urdf_movable = [
        j.get("name") for j in urdf_root.findall("joint") if j.get("type") != "fixed"
    ]
    vsim_movable = [j.get("name") for j in movable_joints(root)]
    if urdf_movable != vsim_movable:
        raise ValueError(
            f"vsim conversion changed movable joints:\n"
            f"  urdf: {urdf_movable}\n  vsim: {vsim_movable}"
        )
    for j in movable_joints(root):
        name = j.get("name")
        if name not in urdf_limits:
            continue  # URDF had no effort/velocity for this joint
        eff, vel = urdf_limits[name]
        limit = j.find("limit")
        if limit is None:
            raise ValueError(f"vsim conversion dropped <limit> of joint {name}")
        got = (float(limit.get("effort")), float(limit.get("velocity")))
        if got != (eff, vel):
            raise ValueError(
                f"vsim conversion mutated limits of joint {name}: "
                f"urdf effort/velocity {(eff, vel)} vs vsim {got}"
            )


def postprocess_vsim(
    tree: ET.ElementTree,
    urdf_path: str,
    fix_base_link: bool,
    joint_damping: float,
    rotor_inertia,
) -> ET.ElementTree:
    """Pure-XML post-processing; separable from conversion for testing.

    NB: collision geometry is deliberately KEPT for fixed-base robots
    (vsim imports geometry from collision shapes — stripping them made the
    pendulum invisible).  Contacts-disabled semantics hold anyway: fixed
    robots get no ground plane, and adjacent-link pairs don't collide.
    Validated by the pendulum energy/period tests."""
    root = tree.getroot()
    verify_vsim_against_urdf(root, urdf_path)
    inject_motors(root)
    inject_contact_sensors(root)
    set_joint_dynamics(root, joint_damping, rotor_inertia)
    absolutize_mesh_paths(root, os.path.dirname(os.path.abspath(urdf_path)))
    return tree


def ensure_vsim_asset(cfg, vgym) -> str:
    """Convert cfg.asset.file to a post-processed .vsim; return its path.

    Requires a live vlearn gym (the converter runs on the singleton), so the
    backend calls this after create_gym and before world-building.
    """
    urdf_path = cfg.asset.file.format(GYM_ROOT_DIR=GYM_ROOT_DIR)
    stem = os.path.splitext(os.path.basename(urdf_path))[0]
    out_dir = os.path.join(os.path.dirname(urdf_path), os.pardir, "vsim")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.abspath(os.path.join(out_dir, f"{stem}.vsim"))

    vgym.convert_urdf_to_vsim(urdf_path, out_path)
    tree = ET.parse(out_path)
    postprocess_vsim(
        tree,
        urdf_path,
        fix_base_link=getattr(cfg.asset, "fix_base_link", True),
        joint_damping=cfg.asset.joint_damping,
        rotor_inertia=getattr(cfg.asset, "rotor_inertia", 0.0),
    )
    tree.write(out_path)
    return out_path
