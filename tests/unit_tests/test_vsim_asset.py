"""Pure-XML tests for the vsim asset post-processor.

The only vsim test that needs neither GPU, license, nor the vlearn package:
it feeds a converter-shaped .vsim document (structure taken from a real
conversion, 2026-07-12) through postprocess_vsim and asserts the invariants
the backend depends on.
"""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from gym import GYM_ROOT_DIR
from gym.envs.base.vsim_asset import (
    absolutize_mesh_paths,
    ensure_vsim_asset,
    postprocess_vsim,
    replace_visual_meshes,
    verify_vsim_against_urdf,
)

URDF = """<robot name="pendulum">
  <link name="base"/>
  <link name="pole"/>
  <joint name="theta" type="continuous">
    <parent link="base"/>
    <child link="pole"/>
    <limit effort="5" velocity="100"/>
  </joint>
</robot>"""

# Shape mirrors convert_urdf_to_vsim output: limits kept, lower>upper =
# unlimited convention, fixed joints preserved, and NO motors.
VSIM = """<robot name="pendulum">
  <link name="base">
    <collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>
  </link>
  <link name="pole">
    <collision><geometry><box size="0.02 0.02 1.0"/></geometry></collision>
  </link>
  <joint name="theta" type="revolute">
    <child link="pole"/><parent link="base"/>
    <limit lower="1" upper="-1" effort="5" velocity="100"/>
  </joint>
</robot>"""


@pytest.fixture
def urdf_path(tmp_path):
    p = tmp_path / "pendulum.urdf"
    p.write_text(URDF)
    return str(p)


def _tree():
    return ET.ElementTree(ET.fromstring(VSIM))


def test_postprocess_injects_motors_sensors_dynamics(urdf_path):
    tree = postprocess_vsim(
        _tree(), urdf_path, fix_base_link=False, joint_damping=0.01, rotor_inertia=0.5
    )
    root = tree.getroot()

    motors = root.findall("./actuator/motor")
    assert [m.get("joint") for m in motors] == ["theta"]
    assert motors[0].get("gear") == "1.0"
    assert float(motors[0].get("highLimit")) == 5.0  # ±effort from URDF
    assert float(motors[0].get("lowLimit")) == -5.0

    sensors = root.findall("./forceSensor/sensor")
    assert {s.get("link") for s in sensors} == {"base", "pole"}
    assert all(s.get("flags") == "contact" for s in sensors)

    dyn = root.find("./joint/dynamics")
    assert dyn.get("damping") == "0.01"
    assert dyn.get("armature") == "0.5"

    # floating base keeps collisions
    assert root.find("./link/collision") is not None


def test_postprocess_keeps_collisions_for_fixed_base(urdf_path):
    """vsim builds geometry from collision shapes — stripping them made the
    pendulum invisible. Contacts-disabled semantics hold via no-plane +
    adjacent-pair exclusion (validated by the pendulum physics tests)."""
    tree = postprocess_vsim(
        _tree(), urdf_path, fix_base_link=True, joint_damping=0.0, rotor_inertia=0.0
    )
    assert tree.getroot().find("./link/collision") is not None


def test_mesh_paths_absolutized(urdf_path, tmp_path):
    tree = _tree()
    geom = tree.getroot().find("./link/collision/geometry")
    mesh = ET.SubElement(geom, "mesh", filename="meshes/part.dae")
    postprocess_vsim(
        tree, urdf_path, fix_base_link=False, joint_damping=0.0, rotor_inertia=0.0
    )
    import os

    assert os.path.isabs(mesh.get("filename"))
    assert mesh.get("filename").endswith("meshes/part.dae")
    assert mesh.get("filename").startswith(str(tmp_path))  # anchored at URDF dir


def test_per_dof_armature_list(urdf_path):
    tree = postprocess_vsim(
        _tree(),
        urdf_path,
        fix_base_link=False,
        joint_damping=0.0,
        rotor_inertia=[0.123],
    )
    assert tree.getroot().find("./joint/dynamics").get("armature") == "0.123"


def test_verify_catches_mutated_effort(urdf_path):
    bad = _tree()
    bad.getroot().find("joint/limit").set("effort", "999")
    with pytest.raises(ValueError, match="mutated limits"):
        verify_vsim_against_urdf(bad.getroot(), urdf_path)


def test_verify_catches_dropped_joint(urdf_path):
    bad = _tree()
    root = bad.getroot()
    root.remove(root.find("joint"))
    with pytest.raises(ValueError, match="changed movable joints"):
        verify_vsim_against_urdf(root, urdf_path)


def test_go2_visual_replacements_preserve_physics_and_link_frames(tmp_path):
    source = ET.parse(
        Path(GYM_ROOT_DIR) / "resources/robots/go2/urdf/go2.urdf"
    ).getroot()
    root = deepcopy(source)
    part_counts = {
        "base": 5,
        "hip": 2,
        "thigh": 2,
        "thigh_mirror": 2,
        "calf": 2,
        "calf_mirror": 2,
        "foot": 1,
    }
    for stem, count in part_counts.items():
        names = (
            [f"{stem}.obj"]
            if count == 1
            else [f"{stem}_{index}.obj" for index in range(count)]
        )
        for name in names:
            (tmp_path / name).touch()

    replace_visual_meshes(root, str(tmp_path))

    for source_link, link in zip(source.findall("link"), root.findall("link")):
        original = source_link.find("visual")
        if original is not None:
            stem = Path(original.find("geometry/mesh").get("filename")).stem
            visuals = link.findall("visual")
            assert len(visuals) == part_counts[stem]
            for visual in visuals:
                mesh_path = Path(visual.find("geometry/mesh").get("filename"))
                assert mesh_path.is_absolute() and mesh_path.is_file()
                assert mesh_path.parent == tmp_path
                # Includes reflected hips and feet in their own fixed-link frames.
                assert visual.find("origin").attrib == original.find("origin").attrib
                assert [ET.tostring(m) for m in visual.findall("material")] == [
                    ET.tostring(m) for m in original.findall("material")
                ]
        for visual in list(source_link.findall("visual")):
            source_link.remove(visual)
        for visual in list(link.findall("visual")):
            link.remove(visual)
    assert ET.tostring(root) == ET.tostring(source)


def test_missing_visual_replacement_fails(tmp_path):
    root = ET.fromstring(
        '<robot><link><visual><geometry><mesh filename="package://robot/foot.dae"/>'
        "</geometry></visual></link></robot>"
    )
    with pytest.raises(FileNotFoundError):
        replace_visual_meshes(root, str(tmp_path))


def test_conversion_receives_replacements_without_changing_source(tmp_path):
    urdf_dir = tmp_path / "urdf"
    urdf_dir.mkdir()
    source_path = urdf_dir / "pendulum.urdf"
    source_root = ET.fromstring(URDF)
    link = source_root.find("link")
    visual = ET.SubElement(link, "visual")
    ET.SubElement(
        ET.SubElement(visual, "geometry"),
        "mesh",
        filename="package://robot/foot.dae",
    )
    collision = ET.SubElement(link, "collision")
    ET.SubElement(
        ET.SubElement(collision, "geometry"), "mesh", filename="collision.stl"
    )
    ET.ElementTree(source_root).write(source_path)
    original = source_path.read_bytes()
    mesh_path = tmp_path / "foot.obj"
    mesh_path.touch()
    calls = []

    def convert_urdf_to_vsim(input_path, output_path):
        calls.append(Path(input_path))
        tree = ET.parse(input_path)
        assert tree.find("./link/visual/geometry/mesh").get("filename") == str(
            mesh_path
        )
        assert tree.find("./link/collision/geometry/mesh").get("filename") == str(
            urdf_dir / "collision.stl"
        )
        tree.write(output_path)

    cfg = SimpleNamespace(
        asset=SimpleNamespace(
            file=str(source_path),
            vsim_visual_mesh_dir=str(tmp_path),
            fix_base_link=True,
            joint_damping=0.0,
            rotor_inertia=0.0,
        )
    )
    output_path = ensure_vsim_asset(
        cfg, SimpleNamespace(convert_urdf_to_vsim=convert_urdf_to_vsim)
    )

    assert len(calls) == 1 and calls[0] != source_path
    assert source_path.read_bytes() == original
    assert ET.parse(output_path).find("./actuator/motor").get("joint") == "theta"


def test_mesh_path_resolution_preserves_package_uris(tmp_path):
    package_uri = "package://robot/meshes/part.obj"
    root = ET.fromstring(
        f'<robot><mesh filename="{package_uri}"/>'
        '<mesh filename="meshes/part.obj"/></robot>'
    )
    absolutize_mesh_paths(root, str(tmp_path))
    assert [mesh.get("filename") for mesh in root.iter("mesh")] == [
        package_uri,
        str(tmp_path / "meshes/part.obj"),
    ]
