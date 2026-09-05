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
    """

    sim_time_s: float
    # Raw Webots sensors (historical channel semantics)
    imu_accel_g: list[float]
    imu_gyro_deg_s: list[float]  # DonkeyCar/WebotsAdapter convention
    imu_gyro_yaw_rate_rad_s: float  # research convenience (observed, may be biased)
    encoder_left_rad_s: float
    encoder_right_rad_s: float
    encoder_speed_m_s: float
    distance_cm: Optional[float]
    # Estimator-derived (not privileged GT)
    heading_est_rad: float
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
    applied_left_rad_s: float
    applied_right_rad_s: float
    tracking_error_rad: float
    episode: int
    seed: int
    condition: str
    success: bool
    done: bool

    def to_log_row(self) -> dict[str, Any]:
        row = {
            "simulation_time": self.controller.sim_time_s,
            "episode": self.episode,
            "seed": self.seed,
            "condition": self.condition,
            "raw_imu_accel_g": self.controller.imu_accel_g,
            "raw_imu_gyro_deg_s": self.controller.imu_gyro_deg_s,
            "observed_imu_yaw_rate_rad_s": self.controller.imu_gyro_yaw_rate_rad_s,
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
            "cmd_left_rad_s": self.cmd_left_rad_s,
            "cmd_right_rad_s": self.cmd_right_rad_s,
            "applied_left_rad_s": self.applied_left_rad_s,
            "applied_right_rad_s": self.applied_right_rad_s,
            "tracking_error_rad": self.tracking_error_rad,
            "success": self.success,
            "done": self.done,
            # Privileged block — firewall tagged
            "privileged_eval_only": True,
            "true_position_m": self.privileged.true_position_m,
            "true_yaw_rad": self.privileged.true_yaw_rad,
            "true_linear_speed_m_s": self.privileged.true_linear_speed_m_s,
        }
        return row
