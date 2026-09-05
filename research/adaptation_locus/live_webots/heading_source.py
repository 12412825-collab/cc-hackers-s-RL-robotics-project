"""Sensor-derived heading from historical Gyro (Option B).

Integrates Webots Gyro yaw-rate with the controller timestep.
Does NOT read Supervisor pose/yaw.
"""

from __future__ import annotations

from .mismatch import wrap_angle_rad


class GyroHeadingIntegrator:
    """theta_t = wrap(theta_{t-1} + omega_t * dt)."""

    def __init__(self, dt: float, heading0_rad: float = 0.0):
        self.dt = float(dt)
        self.heading_rad = float(heading0_rad)

    def reset(self, heading0_rad: float = 0.0) -> None:
        self.heading_rad = wrap_angle_rad(heading0_rad)

    def update(self, yaw_rate_rad_s: float) -> float:
        self.heading_rad = wrap_angle_rad(
            self.heading_rad + float(yaw_rate_rad_s) * self.dt
        )
        return self.heading_rad
