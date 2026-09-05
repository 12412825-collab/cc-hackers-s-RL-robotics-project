"""Low-dimensional heading estimator (Adaptation Locus φ).

Historical SensorFusion is a 9-D normalizer for residual RL — it does NOT
estimate heading. This module is a research estimator (class N) that keeps
the Phase-0/1A spirit: bias_hat + fusion weight over IMU vs encoder yaw rate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EstimatorParams:
    imu_bias_hat_rad_s: float = 0.0
    fusion_weight: float = 0.85  # weight on IMU-corrected rate vs encoder


class HeadingEstimator:
    """Integrate fused yaw-rate; adaptation API exists but Step-2 keeps it OFF."""

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
            "imu_bias_hat_rad_s": float(self.params.imu_bias_hat_rad_s),
            "fusion_weight": float(self.params.fusion_weight),
        }

    def set_params(self, imu_bias_hat_rad_s: float, fusion_weight: float) -> None:
        if self._locked:
            raise RuntimeError("Estimator parameters are locked")
        self.params.imu_bias_hat_rad_s = float(imu_bias_hat_rad_s)
        self.params.fusion_weight = float(max(0.0, min(1.0, fusion_weight)))

    def reset(self, heading0: float = 0.0) -> None:
        self.heading_est_rad = float(heading0)
        # Keep params; reset clears integrated state only.

    def update(
        self,
        imu_yaw_rate_obs_rad_s: float,
        encoder_yaw_rate_rad_s: float,
    ) -> float:
        """Advance heading estimate from sensor rates (no Supervisor GT)."""
        w = self.params.fusion_weight
        imu_corr = float(imu_yaw_rate_obs_rad_s) - self.params.imu_bias_hat_rad_s
        fused = w * imu_corr + (1.0 - w) * float(encoder_yaw_rate_rad_s)
        self.heading_est_rad += fused * self.dt
        # Online adaptation deliberately disabled in Step 2.
        if self._adaptation_enabled:
            raise RuntimeError(
                "Step-2 forbids online estimator adaptation; leave disabled"
            )
        return self.heading_est_rad
