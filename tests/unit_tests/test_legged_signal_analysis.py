import numpy as np
import pytest

from gym.utils.legged_signal_analysis import (
    analyze_base_height,
    analyze_gait_and_grf,
    analyze_spectra,
    urdf_total_mass,
)


def test_urdf_total_mass_reads_all_links():
    mass = urdf_total_mass(
        "resources/robots/mini_cheetah/urdf/mini_cheetah_simple.urdf"
    )

    assert mass == pytest.approx(8.292)


def test_spectral_metrics_detect_high_frequency_motor_content():
    sample_rate = 100.0
    duration = 10.0
    time = np.arange(int(sample_rate * duration)) / sample_rate
    gait = np.sin(2 * np.pi * 2.5 * time)
    shaky = gait + np.sin(2 * np.pi * 20.0 * time)
    torque = np.stack([gait, shaky], axis=1)[:, :, None]
    torque = np.repeat(torque, 2, axis=2).astype(np.float32)
    velocity = np.repeat(gait[:, None, None], 2, axis=1)
    velocity = np.repeat(velocity, 2, axis=2).astype(np.float32)
    alive = np.ones((len(time), 2), dtype=bool)

    metrics, artifacts = analyze_spectra(
        torque,
        velocity,
        alive,
        sample_rate,
        settle_steps=0,
        high_frequency_hz=10.0,
        gait_frequency_hz=2.5,
        survived=np.ones(2, dtype=bool),
    )

    assert metrics["torque_fft_peak_frequency"][0] == pytest.approx(2.5)
    assert metrics["torque_fft_high_frequency_ratio"][0] < 1e-8
    assert metrics["torque_fft_high_frequency_ratio"][1] > 0.45
    assert metrics["joint_velocity_fft_gait_band_ratio"][0] > 0.99
    assert artifacts["torque_psd_by_joint"].shape == (2, 501)


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
