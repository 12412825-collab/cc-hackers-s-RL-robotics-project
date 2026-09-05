"""Phase-0 corridor environment with observation/dynamics separation.

Scientific corrections relative to historical LibraryCorridorEnv:
1. IMU bias corrupts measured / estimated heading rates, NOT true dynamics.
2. Motor asymmetry corrupts true wheel speeds only.
3. Controllers act on estimated heading unless oracle mode is requested.
4. Plant closure: heading integrates from realized differential-drive yaw
   (required so motor asymmetry is a true dynamics mismatch).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from library_residual.observation import LibraryObservationV1

from .baseline import BASELINE, FrozenBaseline
from .mismatches import MismatchSpec, make_mismatch


@dataclass
class Phase0State:
    x: float = 0.0
    d: float = 0.0
    heading_true_deg: float = 0.0
    left_dist: float = 0.0
    right_dist: float = 0.0
    imu_heading_deg: float = 0.0
    estimated_heading_deg: float = 0.0
    step_count: int = 0
    prev_action: float = 0.0
    prev_abs_heading_est: float = 0.0
    done: bool = False


class Phase0CorridorEnv:
    """Controlled Phase-0 simulator for mismatch × adaptation experiments."""

    def __init__(
        self,
        mismatch: Optional[MismatchSpec] = None,
        baseline: FrozenBaseline = BASELINE,
        seed: int = 0,
        use_privileged_heading: bool = False,
    ):
        self.cfg = baseline
        self.mismatch = mismatch or make_mismatch("none", "0")
        self.rng = np.random.RandomState(seed)
        self.use_privileged_heading = use_privileged_heading
        self.state = Phase0State()
        self.imu_bias_hat = 0.0
        self.fusion_weight = baseline.fusion_weight_init
        self._estimator_locked = False
        self._controller_kp = baseline.base_heading_kp

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
            "max_base_correction": float(self.cfg.max_base_correction),
            "max_residual_deg": float(self.cfg.max_residual_deg),
        }

    def reset(self, initial_heading_deg: Optional[float] = None) -> np.ndarray:
        cfg = self.cfg
        h0 = (
            cfg.initial_heading_deg
            if initial_heading_deg is None
            else float(initial_heading_deg)
        )
        self.state = Phase0State(
            x=cfg.initial_lateral_cm,
            d=0.0,
            heading_true_deg=h0,
            left_dist=0.0,
            right_dist=0.0,
            imu_heading_deg=h0,
            estimated_heading_deg=h0,
            step_count=0,
            prev_action=0.0,
            prev_abs_heading_est=abs(h0),
            done=False,
        )
        return self._get_observation()

    def step(self, residual_action: float):
        if self.state.done:
            return self._get_observation(), 0.0, True, {"reason": "already_done"}

        cfg = self.cfg
        action = float(np.clip(residual_action, -1.0, 1.0))
        self.state.step_count += 1

        heading_for_control = (
            self.state.heading_true_deg
            if self.use_privileged_heading
            else self.state.estimated_heading_deg
        )

        base_correction = self._controller_kp * heading_for_control
        base_correction = float(
            np.clip(base_correction, -cfg.max_base_correction, cfg.max_base_correction)
        )
        residual_correction = action * cfg.max_residual_deg
        total_correction = base_correction + residual_correction

        # Commanded body rates (historical damping scale retained)
        v_cmd = cfg.base_speed_cm_s * cfg.battery_effectiveness
        omega_cmd_deg_s = -total_correction * 0.1
        omega_cmd_rad_s = math.radians(omega_cmd_deg_s)
        track = max(cfg.wheel_track_cm, 1e-6)

        v_l_cmd = v_cmd - 0.5 * omega_cmd_rad_s * track
        v_r_cmd = v_cmd + 0.5 * omega_cmd_rad_s * track

        # DYNAMICS mismatch: actuator gains on commanded wheel speeds
        left_speed = v_l_cmd * self.mismatch.left_gain
        right_speed = v_r_cmd * self.mismatch.right_gain

        # TRUE motion from realized differential drive (no IMU bias term)
        heading_rate_true = math.degrees((right_speed - left_speed) / track)
        avg_speed = 0.5 * (left_speed + right_speed)
        self.state.heading_true_deg += heading_rate_true * cfg.dt

        left_noise = self.rng.normal(0.0, cfg.encoder_noise_std_cm)
        right_noise = self.rng.normal(0.0, cfg.encoder_noise_std_cm)
        left_step = left_speed * cfg.dt + left_noise
        right_step = right_speed * cfg.dt + right_noise
        self.state.left_dist += abs(left_step)
        self.state.right_dist += abs(right_step)

        distance_step = avg_speed * cfg.dt
        heading_rad = math.radians(self.state.heading_true_deg)
        self.state.d += distance_step
        self.state.x += distance_step * math.sin(heading_rad)

        # OBSERVATION: IMU measures true rate + bias + noise
        imu_noise = self.rng.normal(0.0, cfg.imu_rate_noise_std_deg_s)
        imu_rate_meas = heading_rate_true + self.mismatch.imu_bias_deg_s + imu_noise
        self.state.imu_heading_deg += imu_rate_meas * cfg.dt

        # Encoder yaw-rate proxy from measured wheel steps
        enc_yaw_rate = math.degrees((right_step - left_step) / track / cfg.dt)

        corrected_imu_rate = imu_rate_meas - self.imu_bias_hat
        w = float(np.clip(self.fusion_weight, 0.0, 1.0))
        fused_rate = w * corrected_imu_rate + (1.0 - w) * enc_yaw_rate
        self.state.estimated_heading_deg += fused_rate * cfg.dt

        obs = self._get_observation()
        reward, done, info = self._reward_and_done(
            action=action,
            distance_step=distance_step,
            base_correction=base_correction,
            residual_correction=residual_correction,
            total_correction=total_correction,
        )
        self.state.done = done
        self.state.prev_action = action
        self.state.prev_abs_heading_est = abs(self.state.estimated_heading_deg)
        info.update(
            {
                "step": self.state.step_count,
                "heading_true_deg": self.state.heading_true_deg,
                "heading_est_deg": self.state.estimated_heading_deg,
                "imu_heading_deg": self.state.imu_heading_deg,
                "lateral_cm": self.state.x,
                "distance_cm": self.state.d,
                "encoder_diff_cm": self.state.left_dist - self.state.right_dist,
                "base_correction": base_correction,
                "residual_correction": residual_correction,
                "total_correction": total_correction,
                "imu_bias_true": self.mismatch.imu_bias_deg_s,
                "imu_bias_hat": self.imu_bias_hat,
                "fusion_weight": self.fusion_weight,
                "left_gain": self.mismatch.left_gain,
                "right_gain": self.mismatch.right_gain,
                "imu_rate_meas": imu_rate_meas,
                "enc_yaw_rate": enc_yaw_rate,
                "heading_rate_true": heading_rate_true,
            }
        )
        return obs, reward, done, info

    def _reward_and_done(
        self,
        *,
        action: float,
        distance_step: float,
        base_correction: float,
        residual_correction: float,
        total_correction: float,
    ):
        cfg = self.cfg
        current_abs_est = abs(self.state.estimated_heading_deg)
        current_abs_true = abs(self.state.heading_true_deg)
        enc_diff = abs(self.state.left_dist - self.state.right_dist)

        reward = 0.0
        reward += cfg.w_progress * (distance_step / cfg.segment_length_cm)
        heading_improvement = self.state.prev_abs_heading_est - current_abs_est
        reward += cfg.w_heading_improvement * heading_improvement
        reward += cfg.w_heading_error * current_abs_est / cfg.max_heading_error_deg
        reward += cfg.w_encoder_diff * enc_diff / 10.0
        reward += cfg.w_residual_action * abs(action)
        reward += cfg.w_action_change * abs(action - self.state.prev_action)

        front_us = self._front_us()
        done = False
        reason = "ongoing"

        if self.state.d >= cfg.segment_length_cm:
            done = True
            if current_abs_true < 5.0 and enc_diff < 2.0:
                reward += cfg.success_reward
                reason = "success"
            else:
                reward += cfg.success_reward * 0.5
                reason = "success_with_drift"
        elif front_us < cfg.ultrasonic_stop_threshold_cm:
            done = True
            reward += cfg.collision_penalty
            reason = "collision"
        elif current_abs_true > cfg.max_heading_error_deg:
            done = True
            reward += cfg.heading_limit_penalty
            reason = "heading_limit"
        elif abs(self.state.x) > cfg.corridor_half_width_cm:
            done = True
            reward += cfg.corridor_penalty
            reason = "corridor_exit"
        elif self.state.step_count > 10 and distance_step < cfg.stall_threshold_cm:
            done = True
            reward += cfg.stall_penalty
            reason = "stall"
        elif self.state.step_count >= cfg.max_steps:
            done = True
            reason = "max_steps"

        return reward, done, {
            "reason": reason,
            "control_effort": abs(total_correction),
            "base_effort": abs(base_correction),
            "residual_effort": abs(residual_correction),
        }

    def _front_us(self) -> float:
        return max(0.0, self.cfg.wall_distance_cm - self.state.d)

    def _get_observation(self) -> np.ndarray:
        heading_feat = (
            self.state.heading_true_deg
            if self.use_privileged_heading
            else self.state.estimated_heading_deg
        )
        obs = LibraryObservationV1(
            motion_direction=self.cfg.direction,
            segment_progress=min(1.0, self.state.d / self.cfg.segment_length_cm),
            fused_heading_error=heading_feat,
            left_right_encoder_error=self.state.left_dist - self.state.right_dist,
            front_ultrasonic_distance=min(self._front_us(), self.cfg.ultrasonic_max_cm),
        )
        return obs.normalize()

    def snapshot_true_metrics(self) -> dict[str, float]:
        return {
            "heading_true_abs": abs(self.state.heading_true_deg),
            "lateral_abs": abs(self.state.x),
            "distance": self.state.d,
            "encoder_diff_abs": abs(self.state.left_dist - self.state.right_dist),
        }
