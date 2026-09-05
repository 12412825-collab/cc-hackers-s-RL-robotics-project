"""Step 3 automated tests: mismatch causal separation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from research.adaptation_locus.live_webots.mismatch import (
    DIAG_IMU_BIAS_RAD_S,
    DIAG_MOTOR_DELTA,
    MismatchLayer,
    diagnostic_suite,
    make_mismatch,
    wrap_angle_rad,
)
from research.adaptation_locus.live_webots.types import ControllerObservation


REPO = Path(__file__).resolve().parents[1]
SUMMARY = (
    REPO / "results" / "adaptation_locus_phase1ar" / "mismatch_validation" / "summary.json"
)


def test_imu_bias_changes_observed_by_exact_bias():
    layer = MismatchLayer(make_mismatch("imu_bias", imu_bias_rad_s=0.1))
    raw = 0.25
    out = layer.apply_imu_bias(raw)
    assert out.raw_imu_yaw_rate_rad_s == raw
    assert abs(out.observed_imu_yaw_rate_rad_s - (raw + 0.1)) < 1e-15
    assert abs(out.delta - 0.1) < 1e-15


def test_raw_imu_unchanged_by_bias_injection():
    layer = MismatchLayer(make_mismatch("imu_bias", imu_bias_rad_s=-0.2))
    raw = -1.5
    out = layer.apply_imu_bias(raw)
    assert out.raw_imu_yaw_rate_rad_s == raw
    assert out.raw_imu_yaw_rate_rad_s != out.observed_imu_yaw_rate_rad_s


def test_imu_bias_does_not_alter_motor_gain():
    cfg = make_mismatch("imu_bias", imu_bias_rad_s=DIAG_IMU_BIAS_RAD_S)
    layer = MismatchLayer(cfg)
    layer.assert_no_cross_contamination()
    mot = layer.apply_motor_gains(2.0, 3.0)
    assert abs(mot.motor_gain_left - 1.0) < 1e-15
    assert abs(mot.motor_gain_right - 1.0) < 1e-15
    assert abs(mot.applied_left_rad_s - 2.0) < 1e-15


def test_motor_mismatch_changes_only_applied_command():
    layer = MismatchLayer(make_mismatch("motor_asymmetry", motor_delta=0.05))
    imu = layer.apply_imu_bias(0.3)
    assert abs(imu.mismatch_bias_rad_s) < 1e-15
    assert abs(imu.observed_imu_yaw_rate_rad_s - 0.3) < 1e-15
    mot = layer.apply_motor_gains(4.0, 4.0)
    assert abs(mot.requested_left_rad_s - 4.0) < 1e-15
    assert abs(mot.applied_left_rad_s - 4.0 * 1.05) < 1e-12
    assert abs(mot.applied_right_rad_s - 4.0 * 0.95) < 1e-12


def test_requested_unchanged_before_mismatch_layer():
    layer = MismatchLayer(make_mismatch("motor_asymmetry", motor_delta=-0.05))
    req_l, req_r = 1.2, -0.8
    mot = layer.apply_motor_gains(req_l, req_r)
    assert mot.requested_left_rad_s == req_l
    assert mot.requested_right_rad_s == req_r


def test_motor_mismatch_does_not_alter_raw_imu_path():
    layer = MismatchLayer(make_mismatch("motor_asymmetry", motor_delta=0.05))
    assert abs(layer.config.effective_imu_bias_rad_s) < 1e-15
    out = layer.apply_imu_bias(0.77)
    assert out.observed_imu_yaw_rate_rad_s == out.raw_imu_yaw_rate_rad_s


def test_positive_negative_imu_bias_signs():
    pos = MismatchLayer(make_mismatch("imu_bias", imu_bias_rad_s=+0.1)).apply_imu_bias(0.0)
    neg = MismatchLayer(make_mismatch("imu_bias", imu_bias_rad_s=-0.1)).apply_imu_bias(0.0)
    assert pos.observed_imu_yaw_rate_rad_s == pytest.approx(0.1)
    assert neg.observed_imu_yaw_rate_rad_s == pytest.approx(-0.1)


def test_reverse_motor_asymmetry_swaps_gains():
    p = make_mismatch("motor_asymmetry", motor_delta=+DIAG_MOTOR_DELTA)
    r = make_mismatch("motor_asymmetry", motor_delta=-DIAG_MOTOR_DELTA)
    assert abs(p.left_gain - (1 + DIAG_MOTOR_DELTA)) < 1e-15
    assert abs(p.right_gain - (1 - DIAG_MOTOR_DELTA)) < 1e-15
    assert abs(r.left_gain - (1 - DIAG_MOTOR_DELTA)) < 1e-15
    assert abs(r.right_gain - (1 + DIAG_MOTOR_DELTA)) < 1e-15


def test_angle_wrap_helper():
    assert wrap_angle_rad(math.pi + 0.1) == pytest.approx(-math.pi + 0.1, abs=1e-9)
    assert wrap_angle_rad(-math.pi - 0.2) == pytest.approx(math.pi - 0.2, abs=1e-9)
    assert wrap_angle_rad(0.0) == 0.0


def test_zero_mismatch_is_identity():
    layer = MismatchLayer(make_mismatch("none"))
    imu = layer.apply_imu_bias(0.42)
    mot = layer.apply_motor_gains(1.0, 2.0)
    assert imu.observed_imu_yaw_rate_rad_s == 0.42
    assert (mot.applied_left_rad_s, mot.applied_right_rad_s) == (1.0, 2.0)
    assert layer.config.is_nominal()


def test_diagnostic_suite_keys():
    suite = diagnostic_suite(seed=0)
    assert set(suite) == {
        "D0_nominal",
        "D1_imu_bias_pos",
        "D2_imu_bias_neg",
        "D3_motor_pos",
        "D4_motor_rev",
    }


def test_controller_obs_has_no_supervisor_gt():
    ctrl = ControllerObservation(
        sim_time_s=1.0,
        imu_accel_g=[0, 0, 1],
        imu_gyro_deg_s=[0, 0, 0],
        raw_imu_yaw_rate_rad_s=0.1,
        observed_imu_yaw_rate_rad_s=0.2,
        mismatch_imu_bias_rad_s=0.1,
        encoder_left_rad_s=0.0,
        encoder_right_rad_s=0.0,
        encoder_speed_m_s=0.0,
        distance_cm=None,
        heading_est_rad=0.01,
        estimator_params={"imu_bias_hat_rad_s": 0.0, "fusion_weight": 0.85},
    )
    blob = str(ctrl.to_dict()).lower()
    assert "true_yaw" not in blob
    assert "true_position" not in blob
    assert "supervisor" not in blob


def test_clipping_recorded_when_over_limit():
    layer = MismatchLayer(
        make_mismatch("motor_asymmetry", motor_delta=0.5), max_wheel_speed=5.0
    )
    mot = layer.apply_motor_gains(20.0, 20.0)
    assert mot.clipped_left and mot.clipped_right
    assert layer.clip_fraction == 1.0


@pytest.mark.skipif(not SUMMARY.is_file(), reason="live mismatch summary not yet generated")
def test_live_mismatch_validation_summary():
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert data["pass"] is True
    assert data["gates"]["m1_causal"] is True
    assert data["gates"]["m2_causal"] is True
    assert data["gates"]["adaptation_off"] is True
    assert data["paired_checks"]["M2_motor"]["clip_fraction_pos"] < 0.05
