"""Base heading P-controller and residual hook for live Webots.

Historical Webots path has no heading P (vision KerasPilot optional).
Heading P is a research-necessary closed-loop base (class N / C-1 spirit).
Residual uses historical VelocityDriveMode Scheme A (Δω on angular velocity).
"""

from __future__ import annotations

from dataclasses import dataclass

from parts.differential_drive import VelocityDriveMode


@dataclass
class HeadingPController:
    """omega_base = clip(-kp * heading_est, ±omega_max)."""

    kp: float = 2.0
    omega_max: float = 1.50

    def __call__(self, heading_est_rad: float) -> float:
        omega = -float(self.kp) * float(heading_est_rad)
        return max(-self.omega_max, min(self.omega_max, omega))


class ResidualHook:
    """Bounded residual on ω; Step-2 default residual = 0 (adaptation OFF)."""

    def __init__(
        self,
        scale_rad_s: float = 0.75,
        max_angular_velocity: float = 1.50,
        max_linear_velocity: float = 0.20,
    ):
        self.scale_rad_s = float(scale_rad_s)
        self.drive = VelocityDriveMode(max_linear_velocity, max_angular_velocity)
        self._residual_action = 0.0  # a in [-1, 1]
        self._adaptation_enabled = False

    def enable_adaptation(self, enabled: bool = True) -> None:
        self._adaptation_enabled = bool(enabled)

    def set_action(self, action: float) -> None:
        if self._adaptation_enabled:
            raise RuntimeError("Step-2 forbids residual adaptation; leave disabled")
        # Even when disabled, allow explicit diagnostic set only if adaptation on.
        # For Step-2, residual stays zero unless force_set used in tests.
        self._residual_action = float(max(-1.0, min(1.0, action)))

    def force_set_action_for_tests(self, action: float) -> None:
        """Test-only bypass; not used by nominal research loop."""
        self._residual_action = float(max(-1.0, min(1.0, action)))

    def reset(self) -> None:
        self._residual_action = 0.0

    @property
    def residual_omega_rad_s(self) -> float:
        return self._residual_action * self.scale_rad_s

    def combine(
        self,
        linear_velocity_m_s: float,
        base_omega_rad_s: float,
    ) -> tuple[float, float, float]:
        """Return (v, omega_final, residual_omega) via historical VelocityDriveMode."""
        # Map physical (v, ω_base) into VelocityDriveMode AI path using normalized
        # pilot channels so residual-on-ω semantics match historical Scheme A.
        max_v = self.drive.limits.max_linear
        max_w = self.drive.limits.max_angular
        pilot_throttle = max(-1.0, min(1.0, linear_velocity_m_s / max_v))
        pilot_steering = max(-1.0, min(1.0, base_omega_rad_s / max_w))
        residual = self.residual_omega_rad_s
        v, omega = self.drive.run(
            mode="local",
            user_steering=0.0,
            user_throttle=0.0,
            pilot_steering=pilot_steering,
            pilot_throttle=pilot_throttle,
            residual_omega=residual,
        )
        return v, omega, residual
