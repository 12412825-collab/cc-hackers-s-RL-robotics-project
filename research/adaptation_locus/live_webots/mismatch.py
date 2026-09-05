"""Phase-1A-R Step 3 — explicit mismatch intervention layer.

M1 enters ONLY the observation path (IMU yaw-rate in rad/s).
M2 enters ONLY the actuator path (left/right wheel command gains).

Adaptation remains OFF; this module never touches φ or residual parameters.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional


MismatchType = Literal["none", "imu_bias", "motor_asymmetry"]

# Diagnostic-only severities for Step 3 (NOT paper-final freezes).
DIAG_IMU_BIAS_RAD_S = 0.10  # rad/s on gyro yaw-rate observation
DIAG_MOTOR_DELTA = 0.05  # gL=1+δ, gR=1-δ


def wrap_angle_rad(angle: float) -> float:
    """Wrap to (-π, π]."""
    a = float(angle)
    return math.atan2(math.sin(a), math.cos(a))


@dataclass(frozen=True)
class MismatchConfig:
    """Immutable research intervention configuration."""

    type: MismatchType = "none"
    severity: str = "0"
    # M1: additive bias on observed IMU yaw-rate [rad/s] (NOT degrees).
    imu_bias_rad_s: float = 0.0
    # M2: symmetric actuator effectiveness; gL=1+δ, gR=1-δ.
    motor_delta: float = 0.0
    seed: int = 0
    active: bool = True
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def left_gain(self) -> float:
        if self.type != "motor_asymmetry" or not self.active:
            return 1.0
        return 1.0 + float(self.motor_delta)

    @property
    def right_gain(self) -> float:
        if self.type != "motor_asymmetry" or not self.active:
            return 1.0
        return 1.0 - float(self.motor_delta)

    @property
    def effective_imu_bias_rad_s(self) -> float:
        if self.type != "imu_bias" or not self.active:
            return 0.0
        return float(self.imu_bias_rad_s)

    def is_nominal(self) -> bool:
        return (
            self.type == "none"
            or not self.active
            or (
                abs(self.effective_imu_bias_rad_s) < 1e-15
                and abs(self.motor_delta) < 1e-15
            )
        )


def make_mismatch(
    type: MismatchType,
    severity: str = "diagnostic",
    *,
    imu_bias_rad_s: float = 0.0,
    motor_delta: float = 0.0,
    seed: int = 0,
    active: bool = True,
    note: str = "",
) -> MismatchConfig:
    if type == "none":
        return MismatchConfig(type="none", severity="0", seed=seed, active=False)
    if type == "imu_bias":
        return MismatchConfig(
            type="imu_bias",
            severity=severity,
            imu_bias_rad_s=float(imu_bias_rad_s),
            motor_delta=0.0,
            seed=seed,
            active=active,
            note=note or "M1 observation-path IMU yaw-rate bias [rad/s]",
        )
    if type == "motor_asymmetry":
        return MismatchConfig(
            type="motor_asymmetry",
            severity=severity,
            imu_bias_rad_s=0.0,
            motor_delta=float(motor_delta),
            seed=seed,
            active=active,
            note=note or "M2 actuator-path motor gain asymmetry",
        )
    raise ValueError(type)


def diagnostic_suite(seed: int = 0) -> dict[str, MismatchConfig]:
    """Minimal Step-3 diagnostic configs (not scientific freezes)."""
    b = DIAG_IMU_BIAS_RAD_S
    d = DIAG_MOTOR_DELTA
    return {
        "D0_nominal": make_mismatch("none", seed=seed),
        "D1_imu_bias_pos": make_mismatch(
            "imu_bias", "diagnostic_pos", imu_bias_rad_s=+b, seed=seed
        ),
        "D2_imu_bias_neg": make_mismatch(
            "imu_bias", "diagnostic_neg", imu_bias_rad_s=-b, seed=seed
        ),
        "D3_motor_pos": make_mismatch(
            "motor_asymmetry", "diagnostic_pos", motor_delta=+d, seed=seed
        ),
        "D4_motor_rev": make_mismatch(
            "motor_asymmetry", "diagnostic_rev", motor_delta=-d, seed=seed
        ),
    }


@dataclass(frozen=True)
class ImuInterventionResult:
    raw_imu_yaw_rate_rad_s: float
    mismatch_bias_rad_s: float
    observed_imu_yaw_rate_rad_s: float

    @property
    def delta(self) -> float:
        return self.observed_imu_yaw_rate_rad_s - self.raw_imu_yaw_rate_rad_s


@dataclass(frozen=True)
class MotorInterventionResult:
    requested_left_rad_s: float
    requested_right_rad_s: float
    motor_gain_left: float
    motor_gain_right: float
    applied_left_rad_s: float
    applied_right_rad_s: float
    clipped_left: bool
    clipped_right: bool
    max_wheel_speed: float


class MismatchLayer:
    """Single intervention layer between sensors/controller and plant I/O."""

    def __init__(self, config: Optional[MismatchConfig] = None, max_wheel_speed: float = 12.0):
        self.config = config or make_mismatch("none")
        self.max_wheel_speed = float(max_wheel_speed)
        self.clip_count = 0
        self.step_count = 0

    def set_config(self, config: MismatchConfig) -> None:
        self.config = config
        self.clip_count = 0
        self.step_count = 0

    def apply_imu_bias(self, raw_imu_yaw_rate_rad_s: float) -> ImuInterventionResult:
        """M1: observation only. Does not modify Webots physical gyro."""
        raw = float(raw_imu_yaw_rate_rad_s)
        bias = self.config.effective_imu_bias_rad_s
        return ImuInterventionResult(
            raw_imu_yaw_rate_rad_s=raw,
            mismatch_bias_rad_s=bias,
            observed_imu_yaw_rate_rad_s=raw + bias,
        )

    def apply_motor_gains(
        self, requested_left_rad_s: float, requested_right_rad_s: float
    ) -> MotorInterventionResult:
        """M2: actuator only. Immediately before Webots Motor.setVelocity."""
        req_l = float(requested_left_rad_s)
        req_r = float(requested_right_rad_s)
        g_l = self.config.left_gain
        g_r = self.config.right_gain
        pre_l = g_l * req_l
        pre_r = g_r * req_r
        lim = self.max_wheel_speed
        app_l = max(-lim, min(lim, pre_l))
        app_r = max(-lim, min(lim, pre_r))
        clipped_l = abs(pre_l) > lim + 1e-12
        clipped_r = abs(pre_r) > lim + 1e-12
        self.step_count += 1
        if clipped_l or clipped_r:
            self.clip_count += 1
        return MotorInterventionResult(
            requested_left_rad_s=req_l,
            requested_right_rad_s=req_r,
            motor_gain_left=g_l,
            motor_gain_right=g_r,
            applied_left_rad_s=app_l,
            applied_right_rad_s=app_r,
            clipped_left=clipped_l,
            clipped_right=clipped_r,
            max_wheel_speed=lim,
        )

    @property
    def clip_fraction(self) -> float:
        if self.step_count <= 0:
            return 0.0
        return float(self.clip_count) / float(self.step_count)

    def assert_no_cross_contamination(self) -> None:
        """Structural invariants of the config object."""
        cfg = self.config
        if cfg.type == "imu_bias":
            assert abs(cfg.motor_delta) < 1e-15
            assert abs(cfg.left_gain - 1.0) < 1e-15
            assert abs(cfg.right_gain - 1.0) < 1e-15
        if cfg.type == "motor_asymmetry":
            assert abs(cfg.imu_bias_rad_s) < 1e-15 or abs(cfg.effective_imu_bias_rad_s) < 1e-15
            assert abs(cfg.effective_imu_bias_rad_s) < 1e-15
        if cfg.type == "none":
            assert cfg.is_nominal()
