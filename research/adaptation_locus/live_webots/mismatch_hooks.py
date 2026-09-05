"""Mismatch injection hooks — Step 2 keeps all mismatches at ZERO.

Hooks exist for Step 3+; they must not alter nominal behavior when zero.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MismatchHooks:
    """Observation / actuator mismatch parameters (default all zero)."""

    imu_bias_rad_s: float = 0.0
    motor_delta: float = 0.0  # gL=1+δ, gR=1-δ

    def observe_imu_yaw_rate(self, true_gyro_yaw_rate_rad_s: float) -> float:
        """Observation-path injection only (does not change plant)."""
        return float(true_gyro_yaw_rate_rad_s) + float(self.imu_bias_rad_s)

    def apply_motor_gains(self, left_cmd: float, right_cmd: float) -> tuple[float, float]:
        """Actuator-path injection only."""
        g_l = 1.0 + float(self.motor_delta)
        g_r = 1.0 - float(self.motor_delta)
        return g_l * float(left_cmd), g_r * float(right_cmd)

    def is_nominal(self) -> bool:
        return abs(self.imu_bias_rad_s) < 1e-15 and abs(self.motor_delta) < 1e-15
