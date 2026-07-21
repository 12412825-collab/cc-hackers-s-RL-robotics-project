"""Differential-drive command conversion parts.

The policy and user interfaces keep DonkeyCar's normalized values at their
boundaries, while the robot-facing interface uses physical linear velocity
(`v`, metres/second) and angular velocity (`omega`, radians/second).
"""

from dataclasses import dataclass


def _clip(value, low, high):
    return max(low, min(high, float(value)))


@dataclass(frozen=True)
class VelocityLimits:
    max_linear: float
    max_angular: float

    def __post_init__(self):
        if self.max_linear <= 0 or self.max_angular <= 0:
            raise ValueError("Velocity limits must be positive")


class VelocityDriveMode:
    """Select manual/base-pilot commands and apply an angular residual.

    DonkeyCar controllers and pilots produce normalized throttle/steering in
    [-1, 1]. This part converts them to physical `v` and `omega`. In automatic
    modes only angular velocity receives the RL residual (scheme A).
    """

    def __init__(self, max_linear_velocity, max_angular_velocity,
                 ai_throttle_mult=1.0):
        self.limits = VelocityLimits(
            float(max_linear_velocity), float(max_angular_velocity))
        self.ai_throttle_mult = float(ai_throttle_mult)

    def _command(self, steering, throttle):
        steering = 0.0 if steering is None else float(steering)
        throttle = 0.0 if throttle is None else float(throttle)
        v = _clip(throttle, -1.0, 1.0) * self.limits.max_linear
        omega = _clip(steering, -1.0, 1.0) * self.limits.max_angular
        return v, omega

    def run(self, mode, user_steering, user_throttle,
            pilot_steering, pilot_throttle, residual_omega=0.0):
        if mode == 'user':
            return self._command(user_steering, user_throttle)

        if mode == 'local_angle':
            v, omega = self._command(pilot_steering, user_throttle)
        else:
            pilot_throttle = (0.0 if pilot_throttle is None
                              else pilot_throttle * self.ai_throttle_mult)
            v, omega = self._command(pilot_steering, pilot_throttle)

        residual_omega = 0.0 if residual_omega is None else float(residual_omega)
        omega = _clip(
            omega + residual_omega,
            -self.limits.max_angular,
            self.limits.max_angular)
        return v, omega


class VelocityToNormalizedControl:
    """Convert physical `v/omega` back to legacy DonkeyCar channels.

    This compatibility boundary lets existing Tub recording and differential
    drivetrain parts continue to consume `throttle/steering` while Webots and
    future real-robot adapters use the physical velocity channels directly.
    """

    def __init__(self, max_linear_velocity, max_angular_velocity):
        self.limits = VelocityLimits(
            float(max_linear_velocity), float(max_angular_velocity))

    def run(self, linear_velocity, angular_velocity):
        v = 0.0 if linear_velocity is None else float(linear_velocity)
        omega = 0.0 if angular_velocity is None else float(angular_velocity)
        throttle = _clip(v / self.limits.max_linear, -1.0, 1.0)
        steering = _clip(omega / self.limits.max_angular, -1.0, 1.0)
        return steering, throttle


class DifferentialDriveKinematics:
    """Convert a body velocity command to left/right wheel angular velocity."""

    def __init__(self, wheel_radius, wheel_separation, max_wheel_speed=None):
        self.wheel_radius = float(wheel_radius)
        self.wheel_separation = float(wheel_separation)
        self.max_wheel_speed = (None if max_wheel_speed is None
                                else float(max_wheel_speed))
        if self.wheel_radius <= 0 or self.wheel_separation <= 0:
            raise ValueError("Wheel radius and separation must be positive")

    def run(self, linear_velocity, angular_velocity):
        v = 0.0 if linear_velocity is None else float(linear_velocity)
        omega = 0.0 if angular_velocity is None else float(angular_velocity)
        half_track = self.wheel_separation / 2.0
        left = (v - omega * half_track) / self.wheel_radius
        right = (v + omega * half_track) / self.wheel_radius
        if self.max_wheel_speed is not None:
            peak = max(abs(left), abs(right))
            if peak > self.max_wheel_speed:
                scale = self.max_wheel_speed / peak
                left *= scale
                right *= scale
        return left, right

