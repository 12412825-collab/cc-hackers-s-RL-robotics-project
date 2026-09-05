"""Phase-1A-R mismatch intervention layer (Step 3 + 3.5).

Observation mismatches (mutually exclusive with each other in one config):
  - fixed_heading_bias  [rad]   — PRIMARY Phase-1A-R M1 (Step 3.5)
  - gyro_rate_bias      [rad/s] — SECONDARY / Step-3 diagnostic (preserved)

Dynamics mismatch:
  - motor_asymmetry     δ with gL=1+δ, gR=1-δ

Adaptation never runs inside this module.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional


MismatchType = Literal[
    "none",
    "fixed_heading_bias",
    "gyro_rate_bias",
    "motor_asymmetry",
    # Deprecated alias kept for Step-3 artifact compatibility in loaders only.
    "imu_bias",
]

# Diagnostic-only (NOT paper freezes)
DIAG_FIXED_HEADING_BIAS_RAD = 0.10
DIAG_GYRO_RATE_BIAS_RAD_S = 0.10  # Step-3 legacy diagnostic
DIAG_MOTOR_DELTA = 0.05

# Back-compat alias
DIAG_IMU_BIAS_RAD_S = DIAG_GYRO_RATE_BIAS_RAD_S


def wrap_angle_rad(angle: float) -> float:
    """Wrap to (-π, π]."""
    a = float(angle)
    return math.atan2(math.sin(a), math.cos(a))


def angle_diff_rad(a: float, b: float) -> float:
    """Signed wrapped difference a - b in (-π, π]."""
    return wrap_angle_rad(float(a) - float(b))


@dataclass(frozen=True)
class MismatchConfig:
    """Immutable research intervention configuration."""

    type: MismatchType = "none"
    severity: str = "0"
    # Primary M1 (Step 3.5): fixed heading offset [rad]
    fixed_heading_bias_rad: float = 0.0
    # Secondary M1 (Step 3): gyro yaw-rate bias [rad/s]
    gyro_rate_bias_rad_s: float = 0.0
    # Deprecated field name — mirrored into gyro_rate_bias_rad_s when loading old configs
    imu_bias_rad_s: float = 0.0
    # M2
    motor_delta: float = 0.0
    seed: int = 0
    active: bool = True
    note: str = ""

    def __post_init__(self) -> None:
        # Normalize deprecated alias into gyro_rate_bias_rad_s if needed.
        if self.type == "imu_bias":
            object.__setattr__(self, "type", "gyro_rate_bias")
        if abs(self.gyro_rate_bias_rad_s) < 1e-15 and abs(self.imu_bias_rad_s) > 0:
            object.__setattr__(self, "gyro_rate_bias_rad_s", float(self.imu_bias_rad_s))

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
    def effective_fixed_heading_bias_rad(self) -> float:
        if self.type != "fixed_heading_bias" or not self.active:
            return 0.0
        return float(self.fixed_heading_bias_rad)

    @property
    def effective_gyro_rate_bias_rad_s(self) -> float:
        if self.type not in ("gyro_rate_bias", "imu_bias") or not self.active:
            return 0.0
        return float(self.gyro_rate_bias_rad_s or self.imu_bias_rad_s)

    # Back-compat property used by older Step-3 code paths
    @property
    def effective_imu_bias_rad_s(self) -> float:
        return self.effective_gyro_rate_bias_rad_s

    def is_nominal(self) -> bool:
        return (
            self.type == "none"
            or not self.active
            or (
                abs(self.effective_fixed_heading_bias_rad) < 1e-15
                and abs(self.effective_gyro_rate_bias_rad_s) < 1e-15
                and abs(self.motor_delta) < 1e-15
            )
        )


def make_mismatch(
    type: MismatchType,
    severity: str = "diagnostic",
    *,
    fixed_heading_bias_rad: float = 0.0,
    gyro_rate_bias_rad_s: float = 0.0,
    imu_bias_rad_s: float = 0.0,  # deprecated alias → gyro_rate_bias
    motor_delta: float = 0.0,
    seed: int = 0,
    active: bool = True,
    note: str = "",
) -> MismatchConfig:
    if type == "none":
        return MismatchConfig(type="none", severity="0", seed=seed, active=False)
    if type == "fixed_heading_bias":
        return MismatchConfig(
            type="fixed_heading_bias",
            severity=severity,
            fixed_heading_bias_rad=float(fixed_heading_bias_rad),
            seed=seed,
            active=active,
            note=note or "PRIMARY M1: fixed heading bias [rad]",
        )
    if type in ("gyro_rate_bias", "imu_bias"):
        rate = float(gyro_rate_bias_rad_s if abs(gyro_rate_bias_rad_s) > 0 else imu_bias_rad_s)
        return MismatchConfig(
            type="gyro_rate_bias",
            severity=severity,
            gyro_rate_bias_rad_s=rate,
            imu_bias_rad_s=rate,
            seed=seed,
            active=active,
            note=note or "SECONDARY: gyro yaw-rate bias [rad/s] (Step-3)",
        )
    if type == "motor_asymmetry":
        return MismatchConfig(
            type="motor_asymmetry",
            severity=severity,
            motor_delta=float(motor_delta),
            seed=seed,
            active=active,
            note=note or "M2 actuator-path motor gain asymmetry",
        )
    raise ValueError(type)


def diagnostic_suite(seed: int = 0) -> dict[str, MismatchConfig]:
    """Step-3 legacy suite (gyro-rate + motor). Kept for historical validation."""
    b = DIAG_GYRO_RATE_BIAS_RAD_S
    d = DIAG_MOTOR_DELTA
    return {
        "D0_nominal": make_mismatch("none", seed=seed),
        "D1_gyro_rate_bias_pos": make_mismatch(
            "gyro_rate_bias", "diagnostic_pos", gyro_rate_bias_rad_s=+b, seed=seed
        ),
        "D2_gyro_rate_bias_neg": make_mismatch(
            "gyro_rate_bias", "diagnostic_neg", gyro_rate_bias_rad_s=-b, seed=seed
        ),
        "D3_motor_pos": make_mismatch(
            "motor_asymmetry", "diagnostic_pos", motor_delta=+d, seed=seed
        ),
        "D4_motor_rev": make_mismatch(
            "motor_asymmetry", "diagnostic_rev", motor_delta=-d, seed=seed
        ),
    }


def fixed_heading_diagnostic_suite(seed: int = 0) -> dict[str, MismatchConfig]:
    """Step-3.5 primary observation mismatch diagnostics."""
    b = DIAG_FIXED_HEADING_BIAS_RAD
    return {
        "H0_nominal": make_mismatch("none", seed=seed),
        "H1_fixed_heading_pos": make_mismatch(
            "fixed_heading_bias", "diagnostic_pos", fixed_heading_bias_rad=+b, seed=seed
        ),
        "H2_fixed_heading_neg": make_mismatch(
            "fixed_heading_bias", "diagnostic_neg", fixed_heading_bias_rad=-b, seed=seed
        ),
        "H3_gyro_rate_contrast": make_mismatch(
            "gyro_rate_bias",
            "contrast",
            gyro_rate_bias_rad_s=+DIAG_GYRO_RATE_BIAS_RAD_S,
            seed=seed,
        ),
    }


@dataclass(frozen=True)
class GyroRateInterventionResult:
    raw_imu_yaw_rate_rad_s: float
    gyro_rate_bias_rad_s: float
    observed_imu_yaw_rate_rad_s: float

    @property
    def delta(self) -> float:
        return self.observed_imu_yaw_rate_rad_s - self.raw_imu_yaw_rate_rad_s


# Back-compat alias
ImuInterventionResult = GyroRateInterventionResult


@dataclass(frozen=True)
class FixedHeadingInterventionResult:
    raw_heading_rad: float
    fixed_heading_bias_rad: float
    observed_heading_rad: float

    @property
    def delta(self) -> float:
        return angle_diff_rad(self.observed_heading_rad, self.raw_heading_rad)


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

    def apply_gyro_rate_bias(self, raw_imu_yaw_rate_rad_s: float) -> GyroRateInterventionResult:
        """Secondary observation mismatch (Step 3). Does not modify Webots gyro."""
        raw = float(raw_imu_yaw_rate_rad_s)
        bias = self.config.effective_gyro_rate_bias_rad_s
        return GyroRateInterventionResult(
            raw_imu_yaw_rate_rad_s=raw,
            gyro_rate_bias_rad_s=bias,
            observed_imu_yaw_rate_rad_s=raw + bias,
        )

    # Back-compat name
    def apply_imu_bias(self, raw_imu_yaw_rate_rad_s: float) -> GyroRateInterventionResult:
        return self.apply_gyro_rate_bias(raw_imu_yaw_rate_rad_s)

    def apply_fixed_heading_bias(self, raw_heading_rad: float) -> FixedHeadingInterventionResult:
        """PRIMARY M1 (Step 3.5): constant heading offset [rad]."""
        raw = float(raw_heading_rad)
        bias = self.config.effective_fixed_heading_bias_rad
        return FixedHeadingInterventionResult(
            raw_heading_rad=raw,
            fixed_heading_bias_rad=bias,
            observed_heading_rad=wrap_angle_rad(raw + bias),
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
        cfg = self.config
        if cfg.type == "fixed_heading_bias":
            assert abs(cfg.motor_delta) < 1e-15
            assert abs(cfg.effective_gyro_rate_bias_rad_s) < 1e-15
            assert abs(cfg.left_gain - 1.0) < 1e-15
        if cfg.type in ("gyro_rate_bias", "imu_bias"):
            assert abs(cfg.motor_delta) < 1e-15
            assert abs(cfg.effective_fixed_heading_bias_rad) < 1e-15
            assert abs(cfg.left_gain - 1.0) < 1e-15
        if cfg.type == "motor_asymmetry":
            assert abs(cfg.effective_fixed_heading_bias_rad) < 1e-15
            assert abs(cfg.effective_gyro_rate_bias_rad_s) < 1e-15
        if cfg.type == "none":
            assert cfg.is_nominal()
