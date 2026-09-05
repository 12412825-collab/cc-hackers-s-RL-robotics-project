"""Mismatch injection helpers for Phase-0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .baseline import SEVERITY_IMU_BIAS_DEG_S, SEVERITY_MOTOR_DELTA

MismatchFamily = Literal["imu_bias", "motor_asymmetry", "none"]
SeverityLabel = Literal["0", "small", "medium", "large"]


@dataclass(frozen=True)
class MismatchSpec:
    family: MismatchFamily
    severity: SeverityLabel
    imu_bias_deg_s: float
    motor_delta: float

    @property
    def left_gain(self) -> float:
        return 1.0 + self.motor_delta

    @property
    def right_gain(self) -> float:
        return 1.0 - self.motor_delta


def make_mismatch(family: MismatchFamily, severity: SeverityLabel) -> MismatchSpec:
    if family == "none" or severity == "0":
        return MismatchSpec(
            family=family if family != "none" else "none",
            severity=severity,
            imu_bias_deg_s=0.0,
            motor_delta=0.0,
        )
    if family == "imu_bias":
        return MismatchSpec(
            family=family,
            severity=severity,
            imu_bias_deg_s=SEVERITY_IMU_BIAS_DEG_S[severity],
            motor_delta=0.0,
        )
    if family == "motor_asymmetry":
        return MismatchSpec(
            family=family,
            severity=severity,
            imu_bias_deg_s=0.0,
            motor_delta=SEVERITY_MOTOR_DELTA[severity],
        )
    raise ValueError(f"Unknown mismatch family: {family}")
