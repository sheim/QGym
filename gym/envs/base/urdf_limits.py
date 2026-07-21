"""URDF <limit> parsing shared by physics backends.

MuJoCo's URDF importer discards effort/velocity limits (it expects them on
actuators), and vsim's converter has its own quirks — so backends re-parse
the limits straight from the URDF text.  Stdlib-only: backends that don't
ship a given engine must still be importable.
"""

import xml.etree.ElementTree as ET


def parse_urdf_limits(urdf_path: str) -> dict:
    """Read <joint><limit effort=... velocity=.../></joint> from a URDF.

    Returns {joint_name: (effort, velocity)}.  Joints without a <limit>
    tag or without both attributes are absent — caller decides the default.
    """
    out: dict = {}
    root = ET.parse(urdf_path).getroot()
    for joint in root.findall("joint"):
        name = joint.get("name")
        limit = joint.find("limit")
        if name is None or limit is None:
            continue
        eff = limit.get("effort")
        vel = limit.get("velocity")
        if eff is None or vel is None:
            continue
        out[name] = (float(eff), float(vel))
    return out
