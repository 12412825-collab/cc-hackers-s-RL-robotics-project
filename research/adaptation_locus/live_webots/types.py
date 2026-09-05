"""Typed observations with a privileged-state firewall."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class PrivilegedEvalState:
    """Supervisor ground truth — evaluation / logging only."""

    true_position_m: list[float]
    true_yaw_rad: float
    true_linear_speed_m_s: float
    privileged_eval_only: bool = True


@dataclass
class ControllerObservation:
    """Signals allowed into the base controller / estimator.

    Must NOT contain Supervisor true yaw/position.
    Heading fields are sensor-derived (Gyro integration), not Supervisor.
    """

    sim_time_s: float
    imu_accel_g: list[float]
    imu_gyro_deg_s: list[float]
    raw_imu_yaw_rate_rad_s: float
    # Rate after optional gyro_rate_bias (secondary); equals raw when primary M1
    observed_imu_yaw_rate_rad_s: float
    gyro_rate_bias_rad_s: float
    # Heading pipeline (primary semantics)
    raw_heading_rad: float
    fixed_heading_bias_rad: float
    observed_heading_rad: float
    encoder_heading_rad: float
    encoder_left_rad_s: float
    encoder_right_rad_s: float
    encoder_speed_m_s: float
    distance_cm: Optional[float]
    heading_est_rad: float
    heading_source: str  # e.g. "gyro_integration"
    estimator_params: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LiveObservation:
    """Full timestep record for logging."""

    controller: ControllerObservation
    privileged: PrivilegedEvalState
    base_omega_rad_s: float
    residual_omega_rad_s: float
    final_omega_rad_s: float
    linear_velocity_m_s: float
    cmd_left_rad_s: float
    cmd_right_rad_s: float
    motor_gain_left: float
    motor_gain_right: float
    applied_left_rad_s: float
    applied_right_rad_s: float
    clipped_left: bool
    clipped_right: bool
    tracking_error_rad: float
    episode: int
    seed: int
    condition: str
    mismatch_type: str
    success: bool
    done: bool

    def to_log_row(self) -> dict[str, Any]:
        return {
            "simulation_time": self.controller.sim_time_s,
            "episode": self.episode,
            "seed": self.seed,
            "condition": self.condition,
            "mismatch_type": self.mismatch_type,
            "heading_source": self.controller.heading_source,
            "raw_imu_accel_g": self.controller.imu_accel_g,
            "raw_imu_gyro_deg_s": self.controller.imu_gyro_deg_s,
            "raw_imu_yaw_rate_rad_s": self.controller.raw_imu_yaw_rate_rad_s,
            "gyro_rate_bias_rad_s": self.controller.gyro_rate_bias_rad_s,
            "observed_imu_yaw_rate_rad_s": self.controller.observed_imu_yaw_rate_rad_s,
            "raw_heading_rad": self.controller.raw_heading_rad,
            "fixed_heading_bias_rad": self.controller.fixed_heading_bias_rad,
            "observed_heading_rad": self.controller.observed_heading_rad,
            "encoder_heading_rad": self.controller.encoder_heading_rad,
            "encoder_left_rad_s": self.controller.encoder_left_rad_s,
            "encoder_right_rad_s": self.controller.encoder_right_rad_s,
            "encoder_speed_m_s": self.controller.encoder_speed_m_s,
            "distance_cm": self.controller.distance_cm,
            "estimator_heading_rad": self.controller.heading_est_rad,
            "estimator_parameters": self.controller.estimator_params,
            "base_control_omega_rad_s": self.base_omega_rad_s,
            "residual_control_omega_rad_s": self.residual_omega_rad_s,
            "final_command_omega_rad_s": self.final_omega_rad_s,
            "linear_velocity_cmd_m_s": self.linear_velocity_m_s,
            "requested_left_rad_s": self.cmd_left_rad_s,
            "requested_right_rad_s": self.cmd_right_rad_s,
            "motor_gain_left": self.motor_gain_left,
            "motor_gain_right": self.motor_gain_right,
            "applied_left_rad_s": self.applied_left_rad_s,
            "applied_right_rad_s": self.applied_right_rad_s,
            "clipped_left": self.clipped_left,
            "clipped_right": self.clipped_right,
            "tracking_error_rad": self.tracking_error_rad,
            "success": self.success,
            "done": self.done,
            "privileged_eval_only": True,
            "true_position_m": self.privileged.true_position_m,
            "true_yaw_rad": self.privileged.true_yaw_rad,
            "true_linear_speed_m_s": self.privileged.true_linear_speed_m_s,
        }
