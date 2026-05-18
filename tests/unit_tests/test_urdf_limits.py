"""URDF effort/velocity limits round-trip into the dof-props dict.

Regression test for the bug where MuJoCo's URDF importer drops <limit
effort=... velocity=.../> tags (it expects them on actuators, which this
backend doesn't create), and _make_dof_props was returning flat fake
constants instead.
"""

import pytest


# Per-joint values from mini_cheetah_simple.urdf — keyed by joint-name suffix.
EXPECTED_EFFORT = {"_haa": 18.0, "_hfe": 18.0, "_kfe": 28.0}
EXPECTED_VELOCITY = {"_haa": 41.0, "_hfe": 41.0, "_kfe": 26.8}


def _suffix_for(name: str) -> str:
    return next(k for k in EXPECTED_EFFORT if name.endswith(k))


def test_urdf_limits_propagate_to_dof_props(legged_cpu_backend):
    b = legged_cpu_backend
    props = b._make_dof_props(b._mjm)

    assert len(b.dof_names) == 12, "mini_cheetah should have 12 actuated DOFs"
    for i, name in enumerate(b.dof_names):
        suffix = _suffix_for(name)
        assert props["effort"][i] == pytest.approx(EXPECTED_EFFORT[suffix]), (
            f"effort mismatch for joint {name}: got {props['effort'][i]}"
        )
        assert props["velocity"][i] == pytest.approx(EXPECTED_VELOCITY[suffix]), (
            f"velocity mismatch for joint {name}: got {props['velocity'][i]}"
        )


def test_urdf_limits_finite_for_all_actuated_joints(legged_cpu_backend):
    """No actuated joint should be falling through to the 1e6 default."""
    b = legged_cpu_backend
    props = b._make_dof_props(b._mjm)
    assert (props["effort"] < 1e5).all(), "some joint's effort fell back to default"
    assert (props["velocity"] < 1e5).all(), "some joint's velocity fell back to default"
