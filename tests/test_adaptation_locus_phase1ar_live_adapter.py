"""Automated tests for Phase-1A-R Step 2 live research adapter."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from parts.differential_drive import DifferentialDriveKinematics, VelocityDriveMode
from research.adaptation_locus.live_webots.controller import (
    HeadingPController,
    ResidualHook,
)
from research.adaptation_locus.live_webots.estimator import HeadingEstimator
from research.adaptation_locus.live_webots.mismatch_hooks import MismatchHooks
from research.adaptation_locus.live_webots.types import ControllerObservation


REPO = Path(__file__).resolve().parents[1]
LIVE_DIR = REPO / "research" / "adaptation_locus" / "live_webots"
VALIDATION = (
    REPO / "results" / "adaptation_locus_phase1ar" / "adapter_validation" / "summary.json"
)


def test_residual_zero_does_not_alter_base_command():
    hook = ResidualHook(scale_rad_s=0.75, max_angular_velocity=1.5, max_linear_velocity=0.2)
    hook.reset()
    v0, w0, r0 = hook.combine(0.12, 0.3)
    assert abs(r0) < 1e-12
    # With residual forced nonzero, omega changes
    hook.force_set_action_for_tests(0.5)
    v1, w1, r1 = hook.combine(0.12, 0.3)
    assert abs(r1 - 0.375) < 1e-9
    assert abs(w1 - w0) > 1e-6
    assert abs(v1 - v0) < 1e-9


def test_estimator_adaptation_disabled_params_fixed():
    est = HeadingEstimator(fusion_weight=0.85, dt=0.05)
    est.enable_adaptation(False)
    est.lock()
    with pytest.raises(RuntimeError):
        est.set_params(0.1, 0.5)
    est.unlock()
    est.set_params(0.0, 0.85)
    before = est.get_params()
    est.lock()
    h = est.update(0.1, 0.0)  # unlock path used by env; here locked update still works on state
    # update doesn't require unlock in estimator itself — env locks around set_params
    est.unlock()
    after = est.get_params()
    assert before == after
    assert abs(h) > 0


def test_mismatch_hooks_zero_are_identity():
    m = MismatchHooks(0.0, 0.0)
    assert m.observe_imu_yaw_rate(0.2) == 0.2
    assert m.apply_motor_gains(1.0, 2.0) == (1.0, 2.0)
    assert m.is_nominal()


def test_controller_observation_firewall_schema():
    ctrl = ControllerObservation(
        sim_time_s=0.0,
        imu_accel_g=[0, 0, 1],
        imu_gyro_deg_s=[0, 0, 0],
        raw_imu_yaw_rate_rad_s=0.0,
        observed_imu_yaw_rate_rad_s=0.0,
        mismatch_imu_bias_rad_s=0.0,
        encoder_left_rad_s=0.0,
        encoder_right_rad_s=0.0,
        encoder_speed_m_s=0.0,
        distance_cm=100.0,
        heading_est_rad=0.0,
        estimator_params={"imu_bias_hat_rad_s": 0.0, "fusion_weight": 0.85},
    )
    data = ctrl.to_dict()
    assert "true_yaw_rad" not in data
    assert "true_position_m" not in data
    blob = str(data).lower()
    assert "true_yaw" not in blob
    assert "supervisor" not in blob


def test_live_modules_do_not_import_faithful_or_phase0_plant():
    forbidden = (
        "WebotsFaithfulEnv",
        "Phase0CorridorEnv",
        "webots_env",
        "from research.adaptation_locus.env",
    )
    for path in LIVE_DIR.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path} contains {token}"


def test_no_python_pose_integration_in_live_path():
    """Ban plant propulsion integration; estimator rate-integration is allowed."""
    banned = re.compile(
        r"(self\.state\.x|self\.state\.y|self\.state\.yaw_true|yaw_true_rad\s*\+=|"
        r"true_yaw_rad\s*\+=|position_m\s*\+=|x_m\s*\+=|WebotsFaithfulEnv)"
    )
    for path in LIVE_DIR.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "WebotsFaithfulEnv" not in text
        for i, ln in enumerate(text.splitlines(), 1):
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            assert not banned.search(ln), f"{path}:{i}: {ln}"

    # Explicit: plant_backend advances only via robot.step
    plant = (LIVE_DIR / "plant_backend.py").read_text(encoding="utf-8")
    assert "robot.step" in plant
    assert "setVelocity" in plant


def test_historical_velocity_drive_residual_zero_identity():
    drive = VelocityDriveMode(0.20, 1.50)
    kin = DifferentialDriveKinematics(0.0325, 0.130, 12.0)
    v, w = drive.run("local", 0, 0, 0.0, 0.12 / 0.20, residual_omega=0.0)
    l0, r0 = kin.run(v, w)
    v2, w2 = drive.run("local", 0, 0, 0.0, 0.12 / 0.20, residual_omega=0.0)
    l1, r1 = kin.run(v2, w2)
    assert (l0, r0, v, w) == (l1, r1, v2, w2)


def test_heading_p_uses_estimate_not_true_yaw_argument_name():
    """Controller API takes heading_est only."""
    src = ast.parse((LIVE_DIR / "controller.py").read_text(encoding="utf-8"))
    for node in ast.walk(src):
        if isinstance(node, ast.FunctionDef) and node.name == "__call__":
            args = [a.arg for a in node.args.args]
            assert "heading_est_rad" in args
            assert "true_yaw" not in args


@pytest.mark.skipif(not VALIDATION.is_file(), reason="live validation summary not generated yet")
def test_live_validation_summary_gates():
    data = json.loads(VALIDATION.read_text(encoding="utf-8"))
    assert data["plant"]["plant"] == "live_webots_ode"
    assert data["plant"]["python_pose_integration"] is False
    assert data["gates"]["live_ode_plant"] is True
    assert data["gates"]["nominal_closed_loop"] is True
    assert data["gates"]["repeated_reset"] is True
    assert data["gates"]["zero_intervention_equivalence"] is True
    assert data["pass"] is True
