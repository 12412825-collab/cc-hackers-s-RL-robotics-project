"""Low-dimensional heading estimator (Adaptation Locus φ).

Primary path (Step 3.5): heading-space fusion after fixed heading bias.
Secondary path (Step 3): rate-space fusion after gyro-rate bias.
"""

from __future__ import annotations

from dataclasses import dataclass

from .mismatch import wrap_angle_rad


@dataclass
class EstimatorParams:
    # Primary A1 parameter (heading bias in rad)
    heading_bias_hat_rad: float = 0.0
    # Secondary / legacy rate bias hat (rad/s) for gyro_rate_bias path
    imu_bias_hat_rad_s: float = 0.0
    fusion_weight: float = 0.85


class HeadingEstimator:
    """Research estimator φ — adaptation OFF in Steps 2–3.5."""

    def __init__(self, fusion_weight: float = 0.85, dt: float = 0.05):
        self.dt = float(dt)
        self.params = EstimatorParams(fusion_weight=float(fusion_weight))
        self.heading_est_rad = 0.0
        self._adaptation_enabled = False
        self._locked = False

    def enable_adaptation(self, enabled: bool = True) -> None:
        self._adaptation_enabled = bool(enabled)

    def lock(self) -> None:
        self._locked = True

    def unlock(self) -> None:
        self._locked = False

    @property
    def adaptation_enabled(self) -> bool:
        return self._adaptation_enabled

    def get_params(self) -> dict[str, float]:
        return {
            "heading_bias_hat_rad": float(self.params.heading_bias_hat_rad),
            "imu_bias_hat_rad_s": float(self.params.imu_bias_hat_rad_s),
            "fusion_weight": float(self.params.fusion_weight),
        }

    def set_params(
        self,
        heading_bias_hat_rad: float = 0.0,
        fusion_weight: float = 0.85,
        imu_bias_hat_rad_s: float = 0.0,
    ) -> None:
        if self._locked:
            raise RuntimeError("Estimator parameters are locked")
        self.params.heading_bias_hat_rad = float(heading_bias_hat_rad)
        self.params.imu_bias_hat_rad_s = float(imu_bias_hat_rad_s)
        self.params.fusion_weight = float(max(0.0, min(1.0, fusion_weight)))

    def reset(self, heading0: float = 0.0) -> None:
        self.heading_est_rad = wrap_angle_rad(heading0)

    def _guard_adaptation(self) -> None:
        if self._adaptation_enabled:
            raise RuntimeError("Step 3.5 forbids online estimator adaptation")

    def update_from_headings(
        self, observed_heading_rad: float, encoder_heading_rad: float
    ) -> float:
        """Primary path: fuse observed (possibly fixed-bias) heading with encoder heading."""
        self._guard_adaptation()
        w = self.params.fusion_weight
        corrected = wrap_angle_rad(
            float(observed_heading_rad) - self.params.heading_bias_hat_rad
        )
        # Linear blend in unwrapped local frame around corrected
        delta = wrap_angle_rad(float(encoder_heading_rad) - corrected)
        self.heading_est_rad = wrap_angle_rad(corrected + (1.0 - w) * delta)
        return self.heading_est_rad

    def update(
        self,
        imu_yaw_rate_obs_rad_s: float,
        encoder_yaw_rate_rad_s: float,
    ) -> float:
        """Secondary/legacy rate-space update (gyro_rate_bias path)."""
        self._guard_adaptation()
        w = self.params.fusion_weight
        imu_corr = float(imu_yaw_rate_obs_rad_s) - self.params.imu_bias_hat_rad_s
        fused = w * imu_corr + (1.0 - w) * float(encoder_yaw_rate_rad_s)
        self.heading_est_rad = wrap_angle_rad(self.heading_est_rad + fused * self.dt)
        return self.heading_est_rad
