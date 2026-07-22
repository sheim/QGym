"""Pure-XML tests for the vsim asset post-processor.

The only vsim test that needs neither GPU, license, nor the vlearn package:
it feeds a converter-shaped .vsim document (structure taken from a real
conversion, 2026-07-12) through postprocess_vsim and asserts the invariants
the backend depends on.
"""

import xml.etree.ElementTree as ET

import pytest

from gym.envs.base.vsim_asset import postprocess_vsim, verify_vsim_against_urdf

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
