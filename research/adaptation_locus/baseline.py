"""Frozen Phase-0 baseline drawn from historical CorridorConfig defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


SEVERITY_IMU_BIAS_DEG_S = {
    "0": 0.0,
    "small": 0.5,
    "medium": 1.5,
    "large": 3.0,
}

SEVERITY_MOTOR_DELTA = {
    "0": 0.0,
    "small": 0.015,
    "medium": 0.03,
    "large": 0.045,
}


@dataclass(frozen=True)
class FrozenBaseline:
    """Nominal robot parameters frozen for Phase-0 (historical values)."""

    segment_length_cm: float = 100.0
    corridor_half_width_cm: float = 15.0
    max_steps: int = 200
    dt: float = 0.1
    base_speed_cm_s: float = 20.0
    wheel_track_cm: float = 18.0
    base_heading_kp: float = 1.5
    max_base_correction: float = 30.0
    max_residual_deg: float = 10.0
    ultrasonic_max_cm: float = 400.0
    wall_distance_cm: float = 200.0
    ultrasonic_stop_threshold_cm: float = 20.0
    max_heading_error_deg: float = 45.0
    stall_threshold_cm: float = 0.5
    encoder_noise_std_cm: float = 0.1
    imu_rate_noise_std_deg_s: float = 0.05
    # Reward weights (historical)
    w_progress: float = 1.0
    w_heading_improvement: float = 0.5
    w_heading_error: float = -0.3
    w_encoder_diff: float = -0.2
    w_residual_action: float = -0.1
    w_action_change: float = -0.05
    success_reward: float = 10.0
    collision_penalty: float = -20.0
    heading_limit_penalty: float = -15.0
    corridor_penalty: float = -15.0
    stall_penalty: float = -10.0
    # Phase-0 freezes (disable historical DR confounds)
    battery_effectiveness: float = 1.0
    initial_heading_deg: float = 0.0
    initial_lateral_cm: float = 0.0
    direction: float = 1.0
    # Budgets
    online_steps: int = 2000
    eval_interval: int = 200
    eval_episodes: int = 4
    recovery_threshold: float = 0.8
    fusion_weight_init: float = 0.85
    estimator_lr: float = 0.08
    residual_lr: float = 3e-4
    residual_batch_size: int = 64
    residual_warmup: int = 100
    residual_updates_per_step: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


BASELINE = FrozenBaseline()
