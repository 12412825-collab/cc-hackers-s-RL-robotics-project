"""Webots-faithful plant for Phase-1A Adaptation Locus replication.

Inherits historical DifferentialDriveKinematics and VelocityDriveMode residual
semantics. Does not require a live Webots binary (Amendment W-1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from parts.differential_drive import DifferentialDriveKinematics

from .webots_baseline import WEBOTS_BASELINE, WebotsFaithfulBaseline
from .mismatches import MismatchSpec, make_mismatch


# Local severity overrides for Phase-1A (rad/s / delta) — see webots_baseline
from .webots_baseline import SEVERITY_IMU_BIAS_RAD_S, SEVERITY_MOTOR_DELTA


def make_webots_mismatch(family: str, severity: str) -> MismatchSpec:
    """Build mismatch using Phase-1A Webots severities (IMU in rad/s)."""
    if family == "none" or severity == "0":
        return MismatchSpec(
            family=family if family != "none" else "none",  # type: ignore[arg-type]
            severity=severity,  # type: ignore[arg-type]
            imu_bias_deg_s=0.0,  # unused; see imu_bias_rad_s property via info
            motor_delta=0.0,
        )
    if family == "imu_bias":
        # Store rad/s numerically in imu_bias_deg_s field for reuse of MismatchSpec
        # (documented: Phase-1A interprets this field as rad/s for Webots plant).
        return MismatchSpec(
            family="imu_bias",
            severity=severity,  # type: ignore[arg-type]
            imu_bias_deg_s=SEVERITY_IMU_BIAS_RAD_S[severity],
            motor_delta=0.0,
        )
    if family == "motor_asymmetry":
        return MismatchSpec(
            family="motor_asymmetry",
            severity=severity,  # type: ignore[arg-type]
            imu_bias_deg_s=0.0,
            motor_delta=SEVERITY_MOTOR_DELTA[severity],
        )
    raise ValueError(family)


@dataclass
class WebotsPlantState:
    x_m: float = 0.0
    y_m: float = 0.0  # along-track progress
    yaw_true_rad: float = 0.0
    yaw_est_rad: float = 0.0
    imu_yaw_rad: float = 0.0
    left_dist_m: float = 0.0
    right_dist_m: float = 0.0
    step_count: int = 0
    prev_action: float = 0.0
    prev_abs_yaw_est: float = 0.0
    done: bool = False


class WebotsFaithfulEnv:
    """Episode env using inherited Webots kinematics + Phase-1A controllers."""

    def __init__(
        self,
        mismatch: Optional[MismatchSpec] = None,
        baseline: WebotsFaithfulBaseline = WEBOTS_BASELINE,
        seed: int = 0,
        family: str = "none",
        severity: str = "0",
    ):
        self.cfg = baseline
        self.mismatch = mismatch or make_webots_mismatch(family, severity)
        self.rng = np.random.RandomState(seed)
        self.kinematics = DifferentialDriveKinematics(
            baseline.wheel_radius_m,
            baseline.wheel_separation_m,
            baseline.max_wheel_speed,
        )
        self.state = WebotsPlantState()
        self.imu_bias_hat = 0.0
        self.fusion_weight = baseline.fusion_weight_init
        self._estimator_locked = False
        self._controller_kp = baseline.base_heading_kp
        # Phase-1A: mismatch.imu_bias_deg_s field holds rad/s (see make_webots_mismatch)
        self._imu_bias_rad_s = float(self.mismatch.imu_bias_deg_s)

    @property
    def imu_bias_true_rad_s(self) -> float:
        return self._imu_bias_rad_s

    def lock_estimator(self) -> None:
        self._estimator_locked = True

    def unlock_estimator(self) -> None:
        self._estimator_locked = False

    def set_estimator_params(self, imu_bias_hat: float, fusion_weight: float) -> None:
        if self._estimator_locked:
            raise RuntimeError("Estimator parameters are locked")
        self.imu_bias_hat = float(imu_bias_hat)
        self.fusion_weight = float(np.clip(fusion_weight, 0.0, 1.0))

    def get_estimator_params(self) -> dict[str, float]:
        return {
            "imu_bias_hat": float(self.imu_bias_hat),
            "fusion_weight": float(self.fusion_weight),
        }

    def get_controller_params(self) -> dict[str, float]:
        return {
            "base_heading_kp": float(self._controller_kp),
            "max_angular_velocity": float(self.cfg.max_angular_velocity),
            "residual_angular_scale": float(self.cfg.residual_angular_scale),
        }

    def reset(self, initial_yaw_rad: Optional[float] = None) -> np.ndarray:
        yaw0 = 0.0 if initial_yaw_rad is None else float(initial_yaw_rad)
        self.state = WebotsPlantState(
            x_m=0.0,
            y_m=0.0,
            yaw_true_rad=yaw0,
            yaw_est_rad=yaw0,
            imu_yaw_rad=yaw0,
            left_dist_m=0.0,
            right_dist_m=0.0,
            step_count=0,
            prev_action=0.0,
            prev_abs_yaw_est=abs(yaw0),
            done=False,
        )
        return self._get_observation()

    def step(self, residual_action: float):
        if self.state.done:
            return self._get_observation(), 0.0, True, {"reason": "already_done"}

        cfg = self.cfg
        action = float(np.clip(residual_action, -1.0, 1.0))
        self.state.step_count += 1

        # Controllers see estimated yaw only
        yaw_for_control = self.state.yaw_est_rad
        omega_base = float(
            np.clip(
                -self._controller_kp * yaw_for_control,
                -cfg.max_angular_velocity,
                cfg.max_angular_velocity,
            )
        )
        omega_res = -action * cfg.residual_angular_scale  # map Phase-0 stabilizing polarity → Webots ω
        omega_cmd = float(
            np.clip(
                omega_base + omega_res,
                -cfg.max_angular_velocity,
                cfg.max_angular_velocity,
            )
        )
        v_cmd = cfg.cruise_linear_velocity

        # Historical kinematics (commanded wheel rad/s)
        w_l_cmd, w_r_cmd = self.kinematics.run(v_cmd, omega_cmd)

        # DYNAMICS mismatch: scale applied wheel commands only
        w_l = w_l_cmd * self.mismatch.left_gain
        w_r = w_r_cmd * self.mismatch.right_gain

        # TRUE motion from realized differential drive (no IMU bias)
        track = cfg.wheel_separation_m
        R = cfg.wheel_radius_m
        v_l = w_l * R
        v_r = w_r * R
        yaw_rate_true = (v_r - v_l) / track
        v_body = 0.5 * (v_l + v_r)

        self.state.yaw_true_rad += yaw_rate_true * cfg.dt
        self.state.y_m += v_body * math.cos(self.state.yaw_true_rad) * cfg.dt
        self.state.x_m += v_body * math.sin(self.state.yaw_true_rad) * cfg.dt

        # Encoder observations (noisy wheel rates)
        w_l_meas = w_l + self.rng.normal(0.0, cfg.encoder_noise_std_rad_s)
        w_r_meas = w_r + self.rng.normal(0.0, cfg.encoder_noise_std_rad_s)
        self.state.left_dist_m += abs(w_l_meas * R * cfg.dt)
        self.state.right_dist_m += abs(w_r_meas * R * cfg.dt)
        enc_yaw_rate = ((w_r_meas - w_l_meas) * R) / track

        # OBSERVATION: IMU gyro yaw-rate + bias (rad/s)
        imu_noise = self.rng.normal(0.0, cfg.gyro_noise_std_rad_s)
        imu_rate_meas = yaw_rate_true + self._imu_bias_rad_s + imu_noise
        self.state.imu_yaw_rad += imu_rate_meas * cfg.dt

        corrected = imu_rate_meas - self.imu_bias_hat
        w = float(np.clip(self.fusion_weight, 0.0, 1.0))
        fused_rate = w * corrected + (1.0 - w) * enc_yaw_rate
        self.state.yaw_est_rad += fused_rate * cfg.dt

        obs = self._get_observation()
        reward, done, info = self._reward_and_done(
            action=action,
            v_body=v_body,
            omega_base=omega_base,
            omega_res=omega_res,
            omega_cmd=omega_cmd,
        )
        self.state.done = done
        self.state.prev_action = action
        self.state.prev_abs_yaw_est = abs(self.state.yaw_est_rad)

        info.update(
            {
                "step": self.state.step_count,
                "heading_true_deg": math.degrees(self.state.yaw_true_rad),
                "heading_est_deg": math.degrees(self.state.yaw_est_rad),
                "yaw_true_rad": self.state.yaw_true_rad,
                "yaw_est_rad": self.state.yaw_est_rad,
                "imu_yaw_rad": self.state.imu_yaw_rad,
                "lateral_cm": self.state.x_m * 100.0,
                "distance_cm": self.state.y_m * 100.0,
                "encoder_diff_cm": (self.state.left_dist_m - self.state.right_dist_m)
                * 100.0,
                "base_correction": omega_base,
                "residual_correction": omega_res,
                "total_correction": omega_cmd,
                "imu_bias_true": self._imu_bias_rad_s,
                "imu_bias_hat": self.imu_bias_hat,
                "fusion_weight": self.fusion_weight,
                "left_gain": self.mismatch.left_gain,
                "right_gain": self.mismatch.right_gain,
                "imu_rate_meas": imu_rate_meas,
                "enc_yaw_rate": enc_yaw_rate,
                "heading_rate_true": yaw_rate_true,
                "wheel_cmd_left": w_l_cmd,
                "wheel_cmd_right": w_r_cmd,
                "wheel_applied_left": w_l,
                "wheel_applied_right": w_r,
            }
        )
        return obs, reward, done, info

    def _reward_and_done(self, *, action, v_body, omega_base, omega_res, omega_cmd):
        cfg = self.cfg
        abs_est = abs(self.state.yaw_est_rad)
        abs_true = abs(self.state.yaw_true_rad)
        progress = max(0.0, v_body * cfg.dt / cfg.segment_length_m)
        reward = progress
        reward += 0.5 * (self.state.prev_abs_yaw_est - abs_est)
        reward += -0.3 * abs_est / cfg.max_heading_error_rad
        reward += -0.1 * abs(action)
        reward += -0.05 * abs(action - self.state.prev_action)

        done = False
        reason = "ongoing"
        if self.state.y_m >= cfg.segment_length_m:
            done = True
            reason = "success" if abs_true < math.radians(5.0) else "success_with_drift"
            reward += 10.0 if reason == "success" else 5.0
        elif abs_true > cfg.max_heading_error_rad:
            done = True
            reason = "heading_limit"
            reward -= 15.0
        elif abs(self.state.x_m) > cfg.corridor_half_width_m:
            done = True
            reason = "corridor_exit"
            reward -= 15.0
        elif self.state.step_count > 10 and abs(v_body) * cfg.dt < cfg.stall_threshold_m:
            done = True
            reason = "stall"
            reward -= 10.0
        elif self.state.step_count >= cfg.max_steps:
            done = True
            reason = "max_steps"

        return reward, done, {
            "reason": reason,
            "control_effort": abs(omega_cmd),
            "base_effort": abs(omega_base),
            "residual_effort": abs(omega_res),
        }

    def _get_observation(self) -> np.ndarray:
        """5-D vector aligned with library-observation-v1 normalization spirit."""
        cfg = self.cfg
        progress = min(1.0, max(0.0, self.state.y_m / cfg.segment_length_m))
        heading_deg = math.degrees(self.state.yaw_est_rad)
        enc_err_cm = (self.state.left_dist_m - self.state.right_dist_m) * 100.0
        front_us = max(0.0, (cfg.segment_length_m - self.state.y_m) * 100.0 + 100.0)

        # Normalize roughly to [-1,1]
        obs = np.array(
            [
                1.0,
                2.0 * progress - 1.0,
                float(np.clip(heading_deg / 45.0, -1.0, 1.0)),
                float(np.clip(enc_err_cm / 10.0, -1.0, 1.0)),
                float(np.clip(2.0 * (front_us / 400.0) - 1.0, -1.0, 1.0)),
            ],
            dtype=np.float32,
        )
        return obs
