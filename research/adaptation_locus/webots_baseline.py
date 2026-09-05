"""Phase-1A Webots-faithful baseline (inherited NUS Webots parameters)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


SEVERITY_IMU_BIAS_RAD_S = {
    "0": 0.0,
    "small": 0.05,
    "medium": 0.15,
    "large": 0.30,
}

SEVERITY_MOTOR_DELTA = {
    "0": 0.0,
    "small": 0.02,
    "medium": 0.04,
    "large": 0.06,
}


@dataclass(frozen=True)
class WebotsFaithfulBaseline:
    """Frozen historical Webots / myconfig plant parameters."""

    # Geometry / limits (FourWheelRobot + myconfig)
    wheel_radius_m: float = 0.0325
    wheel_separation_m: float = 0.130
    max_wheel_speed: float = 12.0
    max_linear_velocity: float = 0.20
    max_angular_velocity: float = 1.50
    residual_angular_scale: float = 0.75
    dt: float = 0.05  # WEBOTS_TIMESTEP_MS = 50
    cruise_linear_velocity: float = 0.12
    robot_mass_kg: float = 0.80
    motor_torque_nm: float = 0.12

    # Episode geometry (corridor task for Adaptation Locus)
    segment_length_m: float = 1.0
    corridor_half_width_m: float = 0.15
    max_steps: int = 400  # 20 s at 20 Hz
    max_heading_error_rad: float = 0.785  # ~45 deg
    stall_threshold_m: float = 0.001

    # Base heading P (Amendment C-1)
    base_heading_kp: float = 2.0

    # Sensors
    gyro_noise_std_rad_s: float = 0.005
    encoder_noise_std_rad_s: float = 0.01
    fusion_weight_init: float = 0.85
    estimator_lr: float = 0.08

    # Residual learning
    residual_lr: float = 0.05

    # Budgets
    online_steps: int = 2000
    eval_interval: int = 200
    eval_episodes: int = 4
    recovery_threshold: float = 0.8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


WEBOTS_BASELINE = WebotsFaithfulBaseline()
