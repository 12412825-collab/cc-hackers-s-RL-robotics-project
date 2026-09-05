"""Step 3 + 3.5 mismatch tests: gyro-rate (legacy) and fixed heading bias."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from research.adaptation_locus.live_webots.heading_source import GyroHeadingIntegrator
from research.adaptation_locus.live_webots.mismatch import (
    DIAG_FIXED_HEADING_BIAS_RAD,
    DIAG_GYRO_RATE_BIAS_RAD_S,
    DIAG_MOTOR_DELTA,
    MismatchLayer,
    angle_diff_rad,
    diagnostic_suite,
    fixed_heading_diagnostic_suite,
    make_mismatch,
    wrap_angle_rad,
)
from research.adaptation_locus.live_webots.types import ControllerObservation


REPO = Path(__file__).resolve().parents[1]
SUMMARY = (
    REPO / "results" / "adaptation_locus_phase1ar" / "mismatch_validation" / "summary.json"
)
SEM_SUMMARY = (
    REPO / "results" / "adaptation_locus_phase1ar" / "sensor_semantics" / "summary.json"
)


def test_gyro_rate_bias_uses_rad_s():
    layer = MismatchLayer(make_mismatch("gyro_rate_bias", gyro_rate_bias_rad_s=0.1))
    out = layer.apply_gyro_rate_bias(0.25)
    assert abs(out.delta - 0.1) < 1e-15
    assert abs(out.gyro_rate_bias_rad_s - 0.1) < 1e-15


def test_imu_bias_alias_maps_to_gyro_rate_bias():
    cfg = make_mismatch("imu_bias", imu_bias_rad_s=0.07)
    assert cfg.type == "gyro_rate_bias"
    assert abs(cfg.effective_gyro_rate_bias_rad_s - 0.07) < 1e-15


def test_fixed_heading_bias_uses_rad():
    layer = MismatchLayer(
        make_mismatch("fixed_heading_bias", fixed_heading_bias_rad=0.10)
    )
    out = layer.apply_fixed_heading_bias(0.5)
    assert abs(out.fixed_heading_bias_rad - 0.10) < 1e-15
    assert abs(angle_diff_rad(out.observed_heading_rad, out.raw_heading_rad) - 0.10) < 1e-12


def test_fixed_heading_bias_constant_not_accumulating():
    layer = MismatchLayer(
        make_mismatch("fixed_heading_bias", fixed_heading_bias_rad=0.10)
    )
    integ = GyroHeadingIntegrator(dt=0.05, heading0_rad=0.0)
    offsets = []
    for _ in range(80):  # 4 s
        raw = integ.update(0.2)  # some motion
        out = layer.apply_fixed_heading_bias(raw)
        offsets.append(angle_diff_rad(out.observed_heading_rad, out.raw_heading_rad))
    assert all(abs(o - 0.10) < 1e-12 for o in offsets)
    # Contrast: rate bias accumulates in integrated heading
    rate_layer = MismatchLayer(
        make_mismatch("gyro_rate_bias", gyro_rate_bias_rad_s=0.10)
    )
    raw_i = GyroHeadingIntegrator(0.05, 0.0)
    bias_i = GyroHeadingIntegrator(0.05, 0.0)
    acc = []
    for _ in range(80):
        omega = 0.0
        raw_h = raw_i.update(omega)
        obs_h = bias_i.update(
            rate_layer.apply_gyro_rate_bias(omega).observed_imu_yaw_rate_rad_s
        )
        acc.append(angle_diff_rad(obs_h, raw_h))
    # With omega=0 and rate bias 0.1, integrated offset grows as 0.1 * n * dt
    assert abs(acc[-1] - 0.10 * 80 * 0.05) < 1e-9
    assert abs(acc[19] - 0.10 * 20 * 0.05) < 1e-9
    assert abs(acc[-1]) > abs(acc[10]) + 0.2


def test_positive_negative_fixed_heading_bias_signs():
    pos = MismatchLayer(
        make_mismatch("fixed_heading_bias", fixed_heading_bias_rad=+0.1)
    ).apply_fixed_heading_bias(0.0)
    neg = MismatchLayer(
        make_mismatch("fixed_heading_bias", fixed_heading_bias_rad=-0.1)
    ).apply_fixed_heading_bias(0.0)
    assert pos.observed_heading_rad == pytest.approx(0.1)
    assert neg.observed_heading_rad == pytest.approx(-0.1)


def test_fixed_heading_bias_angle_wrap():
    layer = MismatchLayer(
        make_mismatch("fixed_heading_bias", fixed_heading_bias_rad=0.2)
    )
    raw = math.pi - 0.05
    out = layer.apply_fixed_heading_bias(raw)
    assert abs(angle_diff_rad(out.observed_heading_rad, out.raw_heading_rad) - 0.2) < 1e-12
    assert out.observed_heading_rad == pytest.approx(wrap_angle_rad(raw + 0.2))


def test_fixed_heading_bias_does_not_change_motor_gains():
    layer = MismatchLayer(
        make_mismatch("fixed_heading_bias", fixed_heading_bias_rad=0.1)
    )
    layer.assert_no_cross_contamination()
    mot = layer.apply_motor_gains(2.0, 3.0)
    assert mot.motor_gain_left == 1.0 and mot.motor_gain_right == 1.0


def test_fixed_heading_bias_does_not_change_raw_rate_channel():
    layer = MismatchLayer(
        make_mismatch("fixed_heading_bias", fixed_heading_bias_rad=0.1)
    )
    rate = layer.apply_gyro_rate_bias(0.33)
    assert rate.observed_imu_yaw_rate_rad_s == rate.raw_imu_yaw_rate_rad_s


def test_raw_imu_unchanged_by_gyro_rate_bias_injection():
    layer = MismatchLayer(make_mismatch("gyro_rate_bias", gyro_rate_bias_rad_s=-0.2))
    raw = -1.5
    out = layer.apply_gyro_rate_bias(raw)
    assert out.raw_imu_yaw_rate_rad_s == raw


def test_imu_bias_does_not_alter_motor_gain():
    cfg = make_mismatch("gyro_rate_bias", gyro_rate_bias_rad_s=DIAG_GYRO_RATE_BIAS_RAD_S)
    layer = MismatchLayer(cfg)
    layer.assert_no_cross_contamination()
    mot = layer.apply_motor_gains(2.0, 3.0)
    assert abs(mot.applied_left_rad_s - 2.0) < 1e-15


def test_motor_mismatch_changes_only_applied_command():
    layer = MismatchLayer(make_mismatch("motor_asymmetry", motor_delta=0.05))
    imu = layer.apply_gyro_rate_bias(0.3)
    assert abs(imu.gyro_rate_bias_rad_s) < 1e-15
    mot = layer.apply_motor_gains(4.0, 4.0)
    assert abs(mot.applied_left_rad_s - 4.0 * 1.05) < 1e-12


def test_requested_unchanged_before_mismatch_layer():
    layer = MismatchLayer(make_mismatch("motor_asymmetry", motor_delta=-0.05))
    mot = layer.apply_motor_gains(1.2, -0.8)
    assert mot.requested_left_rad_s == 1.2


def test_reverse_motor_asymmetry_swaps_gains():
    p = make_mismatch("motor_asymmetry", motor_delta=+DIAG_MOTOR_DELTA)
    r = make_mismatch("motor_asymmetry", motor_delta=-DIAG_MOTOR_DELTA)
    assert abs(p.left_gain - (1 + DIAG_MOTOR_DELTA)) < 1e-15
    assert abs(r.left_gain - (1 - DIAG_MOTOR_DELTA)) < 1e-15


def test_angle_wrap_helper():
    assert wrap_angle_rad(math.pi + 0.1) == pytest.approx(-math.pi + 0.1, abs=1e-9)


def test_zero_mismatch_is_identity():
    layer = MismatchLayer(make_mismatch("none"))
    assert layer.apply_gyro_rate_bias(0.42).observed_imu_yaw_rate_rad_s == 0.42
    assert layer.apply_fixed_heading_bias(0.3).observed_heading_rad == pytest.approx(0.3)


def test_diagnostic_suite_keys():
    suite = diagnostic_suite(seed=0)
    assert "D1_gyro_rate_bias_pos" in suite
    fh = fixed_heading_diagnostic_suite(0)
    assert "H1_fixed_heading_pos" in fh


def test_gyro_integration_uses_dt():
    g = GyroHeadingIntegrator(dt=0.05, heading0_rad=0.0)
    h = g.update(2.0)
    assert abs(h - 0.1) < 1e-12


def test_gyro_integration_reset():
    g = GyroHeadingIntegrator(dt=0.05, heading0_rad=0.0)
    g.update(1.0)
    g.reset(0.0)
    assert abs(g.heading_rad) < 1e-15


def test_controller_obs_has_no_supervisor_gt():
    ctrl = ControllerObservation(
        sim_time_s=1.0,
        imu_accel_g=[0, 0, 1],
        imu_gyro_deg_s=[0, 0, 0],
        raw_imu_yaw_rate_rad_s=0.1,
        observed_imu_yaw_rate_rad_s=0.1,
        gyro_rate_bias_rad_s=0.0,
        raw_heading_rad=0.1,
        fixed_heading_bias_rad=0.1,
        observed_heading_rad=0.2,
        encoder_heading_rad=0.05,
        encoder_left_rad_s=0.0,
        encoder_right_rad_s=0.0,
        encoder_speed_m_s=0.0,
        distance_cm=None,
        heading_est_rad=0.2,
        heading_source="gyro_integration",
        estimator_params={"heading_bias_hat_rad": 0.0, "fusion_weight": 0.85},
    )
    blob = str(ctrl.to_dict()).lower()
    assert "true_yaw" not in blob
    assert "supervisor" not in blob
    assert ctrl.heading_source == "gyro_integration"


def test_clipping_recorded_when_over_limit():
    layer = MismatchLayer(
        make_mismatch("motor_asymmetry", motor_delta=0.5), max_wheel_speed=5.0
    )
    mot = layer.apply_motor_gains(20.0, 20.0)
    assert mot.clipped_left and layer.clip_fraction == 1.0


@pytest.mark.skipif(not SUMMARY.is_file(), reason="live mismatch summary not yet generated")
def test_live_mismatch_validation_summary_preserved():
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert data["pass"] is True


@pytest.mark.skipif(not SEM_SUMMARY.is_file(), reason="sensor semantics summary not yet")
def test_live_sensor_semantics_summary():
    data = json.loads(SEM_SUMMARY.read_text(encoding="utf-8"))
    assert data["pass"] is True
    assert data["primary_heading_source"] == "gyro_integration"
    assert data["uses_supervisor_for_controller_heading"] is False
    assert data["gates"]["fixed_offset_constant"] is True
