import numpy as np
import pytest

from gym.utils.legged_signal_analysis import (
    analyze_base_height,
    analyze_foot_clearance_by_phase,
    analyze_gait_and_grf,
    urdf_total_mass,
)


def test_urdf_total_mass_reads_all_links():
    mass = urdf_total_mass(
        "resources/robots/mini_cheetah/urdf/mini_cheetah_simple.urdf"
    )

    assert mass == pytest.approx(8.292)


def test_base_height_metrics_measure_steadiness_independent_of_target():
    sample_rate = 100.0
    time = np.arange(100) / sample_rate
    height = (0.42 + 0.001 * time)[:, None]
    alive = np.ones_like(height, dtype=bool)

    metrics = analyze_base_height(height, alive, sample_rate, settle_steps=0)

    assert metrics["base_height_mean"][0] > 0.42
    assert metrics["base_height_range"][0] == pytest.approx(0.00099)
    assert metrics["base_height_drift_abs"][0] == pytest.approx(0.001)


def test_touchdown_rpd_and_grf_identify_balanced_trot():
    sample_rate = 100.0
    num_steps = 480
    period = 40
    stance_steps = period // 2
    robot_mass = 8.0
    stance_force = 0.5 * robot_mass * 9.81
    force = np.zeros((num_steps, 1, 4), dtype=np.float32)
    touchdown_offsets = [0, period // 2, period // 2, 0]
    for foot, offset in enumerate(touchdown_offsets):
        for touchdown in range(offset, num_steps, period):
            force[touchdown : touchdown + stance_steps, 0, foot] = stance_force

    metrics, artifacts = analyze_gait_and_grf(
        force,
        force,
        np.ones((num_steps, 1), dtype=bool),
        moving=np.ones(1, dtype=bool),
        sample_rate_hz=sample_rate,
        settle_steps=0,
        contact_threshold_n=20.0,
        gait_frequency_hz=2.5,
        robot_mass_kg=robot_mass,
    )

    assert metrics["gait_trot_classified"][0] == 1.0
    assert metrics["gait_rpd_trot_error"][0] == pytest.approx(0.0)
    assert metrics["gait_rpd_cycle_consistency"][0] == pytest.approx(0.0)
    assert metrics["grf_balance_cv"][0] == pytest.approx(0.0)
    assert metrics["grf_total_body_weight"][0] == pytest.approx(1.0, rel=1e-5)
    np.testing.assert_allclose(
        artifacts["grf_body_weight_by_foot"][0],
        np.full(4, 0.25),
        rtol=1e-5,
    )
    assert artifacts["gait_class"][0] == "trot"


def test_clearance_is_measured_relative_to_each_foot_stance_height():
    num_steps = 8
    phase = np.linspace(0.0, 2.0 * np.pi, num_steps, endpoint=False)
    phase = np.broadcast_to(phase[:, None, None], (num_steps, 1, 2))
    stance = phase < np.pi
    height = np.full((num_steps, 1, 2), 0.02, dtype=np.float32)
    height[:, 0, 1] += 0.01
    height[~stance] += 0.05
    force = np.where(stance, 10.0, 0.0).astype(np.float32)
    force_z = 2.0 * force

    metrics, artifacts = analyze_foot_clearance_by_phase(
        height,
        force,
        force_z,
        phase,
        stance,
        np.ones((num_steps, 1), dtype=bool),
        moving=np.ones(1, dtype=bool),
        settle_steps=0,
        contact_threshold_n=5.0,
        num_phase_bins=8,
    )

    assert metrics["swing_clearance_p95_mean"][0] == pytest.approx(0.05)
    assert metrics["swing_clearance_p95_min"][0] == pytest.approx(0.05)
    np.testing.assert_allclose(artifacts["foot_contact_by_phase"][0, :, :4], 1.0)
    np.testing.assert_allclose(
        artifacts["foot_contact_force_z_by_phase"][0, :, :4], 20.0
    )
    np.testing.assert_allclose(
        artifacts["foot_contact_force_z_by_phase"][0, :, 4:], 0.0
    )
    np.testing.assert_allclose(artifacts["foot_clearance_by_phase"][0, 1, 4:], 0.05)
