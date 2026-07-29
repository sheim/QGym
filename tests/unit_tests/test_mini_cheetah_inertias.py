"""Physical-validity checks for the Mini Cheetah source inertias."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest


ROBOT_DIR = (
    Path(__file__).resolve().parents[2]
    / "resources"
    / "robots"
    / "mini_cheetah"
    / "urdf"
)
URDFS = (
    ROBOT_DIR / "mini_cheetah_simple.urdf",
    ROBOT_DIR / "mini_cheetah_rotor.urdf",
)


def _inertia_matrix(inertia):
    return np.array(
        [
            [inertia["ixx"], inertia["ixy"], inertia["ixz"]],
            [inertia["ixy"], inertia["iyy"], inertia["iyz"]],
            [inertia["ixz"], inertia["iyz"], inertia["izz"]],
        ],
        dtype=np.float64,
    )


@pytest.mark.parametrize("urdf_path", URDFS, ids=lambda path: path.stem)
def test_all_inertias_are_strictly_physical(urdf_path):
    """Every tensor must have positive moments satisfying the triangle rule."""
    root = ET.parse(urdf_path).getroot()
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        inertia = inertial.find("inertia")
        tensor = _inertia_matrix(
            {name: float(value) for name, value in inertia.attrib.items()}
        )
        moments = np.linalg.eigvalsh(tensor)
        assert np.all(moments > 0.0), (link.get("name"), moments)
        assert moments[0] + moments[1] > moments[2], (
            link.get("name"),
            moments,
        )


@pytest.mark.parametrize("urdf_path", URDFS, ids=lambda path: path.stem)
def test_base_inertia_matches_upstream_dynamics_model(urdf_path):
    base = ET.parse(urdf_path).getroot().find("./link[@name='base']/inertial/inertia")
    np.testing.assert_allclose(
        [float(base.get(name)) for name in ("ixx", "iyy", "izz")],
        [0.011253, 0.036203, 0.042673],
        rtol=0.0,
        atol=0.0,
    )
