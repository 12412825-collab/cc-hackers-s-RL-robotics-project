"""A1 estimator adaptation: low-dimensional online bias / fusion updates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .env import Phase0CorridorEnv


@dataclass
class EstimatorAdapter:
    """Recursive bias estimate + light fusion-weight adaptation.

    Uses only measured IMU rate and encoder yaw-rate proxy (no true state).
    """

    lr: float = 0.05
    fusion_lr: float = 0.01
    history: list[dict[str, float]] = field(default_factory=list)

    def reset(self, env: Phase0CorridorEnv) -> None:
        env.unlock_estimator()
        env.set_estimator_params(imu_bias_hat=0.0, fusion_weight=env.cfg.fusion_weight_init)
        self.history.clear()

    def update(self, env: Phase0CorridorEnv, info: dict[str, Any]) -> dict[str, float]:
        """One online update from the latest transition info."""
        if env._estimator_locked:
            env.unlock_estimator()

        imu_rate = float(info["imu_rate_meas"])
        enc_rate = float(info["enc_yaw_rate"])
        # Bias residual: IMU should match encoder yaw proxy after bias removal
        innov = imu_rate - env.imu_bias_hat - enc_rate
        new_bias = env.imu_bias_hat + self.lr * innov

        # Increase IMU weight when corrected IMU agrees with encoder
        corrected = imu_rate - new_bias
        agree = -abs(corrected - enc_rate)
        new_w = float(np.clip(env.fusion_weight + self.fusion_lr * agree, 0.0, 1.0))

        env.set_estimator_params(imu_bias_hat=new_bias, fusion_weight=new_w)
        rec = {
            "imu_bias_hat": new_bias,
            "fusion_weight": new_w,
            "innovation": float(innov),
        }
        self.history.append(rec)
        return rec

    def parameter_magnitude(self) -> float:
        if not self.history:
            return 0.0
        last = self.history[-1]
        return float(abs(last["imu_bias_hat"]) + abs(last["fusion_weight"] - 0.7))


@dataclass
class OracleEstimatorAdapter(EstimatorAdapter):
    """Diagnostic oracle: uses true heading rate (NOT for primary GO/STOP)."""

    def update(self, env: Phase0CorridorEnv, info: dict[str, Any]) -> dict[str, float]:
        if env._estimator_locked:
            env.unlock_estimator()
        true_rate = float(info["heading_rate_true"])
        imu_rate = float(info["imu_rate_meas"])
        innov = imu_rate - true_rate - env.imu_bias_hat
        new_bias = env.imu_bias_hat + self.lr * innov
        env.set_estimator_params(imu_bias_hat=new_bias, fusion_weight=env.fusion_weight)
        rec = {
            "imu_bias_hat": new_bias,
            "fusion_weight": float(env.fusion_weight),
            "innovation": float(innov),
        }
        self.history.append(rec)
        return rec
