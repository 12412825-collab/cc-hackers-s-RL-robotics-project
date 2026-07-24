"""Lightweight differential-drive straight-corridor training environment.

Trains only straight-line residual heading correction.
Does NOT train route planning or obstacle decisions.

At every step:
  1. The fixed base controller calculates its normal heading correction.
  2. SAC outputs one small residual correction.
  3. The environment applies base + residual.
  4. The simulation advances one step.
  5. Returns the next 5-dim observation and reward.

Episode terminates on:
  - segment completion (success)
  - collision (forward ultrasonic < threshold)
  - excessive heading error
  - leaving the permitted corridor
  - encoder stall
  - max steps reached
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .observation import LibraryObservationV1
from .types import OBSERVATION_DIM


@dataclass
class CorridorConfig:
    """Configuration for the simulated corridor environment."""

    # Segment
    segment_length_cm: float = 100.0
    corridor_half_width_cm: float = 15.0
    max_steps: int = 200
    dt: float = 0.1  # seconds per step

    # Robot
    base_speed_cm_s: float = 20.0
    wheel_track_cm: float = 18.0
    base_heading_kp: float = 1.5
    max_base_correction: float = 30.0

    # Ultrasonic
    ultrasonic_max_cm: float = 400.0
    wall_distance_cm: float = 200.0  # distance to front wall at start
    ultrasonic_stop_threshold_cm: float = 20.0

    # Safety
    max_heading_error_deg: float = 45.0
    stall_threshold_cm: float = 0.5  # minimum distance per step

    # Randomisation ranges
    heading_error_range_deg: tuple[float, float] = (-5.0, 5.0)
    motor_gain_mismatch_range: tuple[float, float] = (0.9, 1.1)
    imu_bias_range_deg_s: tuple[float, float] = (-1.0, 1.0)
    encoder_noise_std_cm: float = 0.1
    lateral_offset_range_cm: tuple[float, float] = (-3.0, 3.0)
    battery_effectiveness_range: tuple[float, float] = (0.85, 1.0)

    # Reward weights
    w_progress: float = 1.0
    w_heading_improvement: float = 0.5
    w_heading_error: float = -0.3
    w_encoder_diff: float = -0.2
    w_residual_action: float = -0.1
    w_action_change: float = -0.05

    # Terminal rewards/penalties
    success_reward: float = 10.0
    collision_penalty: float = -20.0
    heading_limit_penalty: float = -15.0
    corridor_penalty: float = -15.0
    stall_penalty: float = -10.0
    safety_stop_penalty: float = -10.0


class LibraryCorridorEnv:
    """Lightweight differential-drive corridor simulator for SAC training.

    The observation is the 5-dim library-observation-v1 vector (normalised).
    The action is a single float in [-1, 1] (residual heading correction).
    """

    def __init__(self, config: Optional[CorridorConfig] = None, seed: int = 0):
        self.cfg = config or CorridorConfig()
        self.rng = np.random.RandomState(seed)

        # State variables (reset in reset())
        self._x = 0.0  # lateral offset (cm)
        self._d = 0.0  # distance along corridor (cm)
        self._heading = 0.0  # heading error (degrees)
        self._left_dist = 0.0  # left encoder distance (cm)
        self._right_dist = 0.0  # right encoder distance (cm)
        self._direction = 1.0  # +1 forward, -1 backward
        self._left_gain = 1.0
        self._right_gain = 1.0
        self._imu_bias = 0.0
        self._battery = 1.0
        self._step_count = 0
        self._prev_action = 0.0
        self._prev_heading = 0.0
        self._target_distance = self.cfg.segment_length_cm
        self._done = False

    @property
    def observation_dim(self) -> int:
        return OBSERVATION_DIM

    @property
    def action_dim(self) -> int:
        return 1

    def reset(self, direction: Optional[float] = None) -> np.ndarray:
        """Reset the environment and return the initial observation."""
        cfg = self.cfg

        # Randomise direction if not specified
        if direction is None:
            self._direction = self.rng.choice([1.0, -1.0])
        else:
            self._direction = float(direction)

        # Randomise initial conditions
        self._heading = self.rng.uniform(*cfg.heading_error_range_deg)
        self._x = self.rng.uniform(*cfg.lateral_offset_range_cm)
        self._d = 0.0
        self._left_dist = 0.0
        self._right_dist = 0.0
        self._left_gain = self.rng.uniform(*cfg.motor_gain_mismatch_range)
        self._right_gain = self.rng.uniform(*cfg.motor_gain_mismatch_range)
        self._imu_bias = self.rng.uniform(*cfg.imu_bias_range_deg_s)
        self._battery = self.rng.uniform(*cfg.battery_effectiveness_range)
        self._step_count = 0
        self._prev_action = 0.0
        self._prev_heading = abs(self._heading)
        self._target_distance = cfg.segment_length_cm
        self._done = False

        return self._get_observation()

    def step(self, action: float):
        """Execute one simulation step.

        Returns: (observation, reward, done, info)
        """
        if self._done:
            return self._get_observation(), 0.0, True, {"reason": "already_done"}

        cfg = self.cfg
        action = float(np.clip(action, -1.0, 1.0))
        self._step_count += 1

        # Base controller heading correction
        base_correction = cfg.base_heading_kp * self._heading
        base_correction = np.clip(
            base_correction, -cfg.max_base_correction, cfg.max_base_correction
        )

        # Residual correction (simulated as a small heading adjustment)
        max_residual_deg = 3.0  # max degrees the residual can affect
        residual_correction = action * max_residual_deg

        # Total correction changes heading
        total_correction = base_correction + residual_correction
        heading_change = -total_correction * cfg.dt * 0.1  # damped

        # Add IMU bias drift
        heading_change += self._imu_bias * cfg.dt

        # Update heading
        self._heading += heading_change

        # Compute wheel speeds with gain mismatch
        base_speed = cfg.base_speed_cm_s * self._battery
        heading_rad = math.radians(self._heading)

        left_speed = base_speed * self._left_gain
        right_speed = base_speed * self._right_gain

        # Add encoder noise
        left_noise = self.rng.normal(0, cfg.encoder_noise_std_cm)
        right_noise = self.rng.normal(0, cfg.encoder_noise_std_cm)

        left_step = left_speed * cfg.dt + left_noise
        right_step = right_speed * cfg.dt + right_noise

        self._left_dist += abs(left_step)
        self._right_dist += abs(right_step)

        # Update position
        avg_speed = (left_speed + right_speed) / 2.0
        distance_step = avg_speed * cfg.dt
        self._d += distance_step
        self._x += distance_step * math.sin(heading_rad)

        # Compute ultrasonic (simulated forward wall)
        front_us = cfg.wall_distance_cm - self._d * self._direction
        if self._direction < 0:
            front_us = cfg.wall_distance_cm  # no rear obstacle sensor

        # Build observation
        obs = self._get_observation()

        # Compute reward
        reward = 0.0
        info = {"reason": "ongoing"}

        # Progress reward
        progress = distance_step / cfg.segment_length_cm
        reward += cfg.w_progress * progress

        # Heading improvement reward
        current_abs_heading = abs(self._heading)
        heading_improvement = self._prev_heading - current_abs_heading
        reward += cfg.w_heading_improvement * heading_improvement

        # Heading error penalty
        reward += cfg.w_heading_error * current_abs_heading / cfg.max_heading_error_deg

        # Encoder difference penalty
        enc_diff = abs(self._left_dist - self._right_dist)
        reward += cfg.w_encoder_diff * enc_diff / 10.0

        # Residual action penalty (prefer small corrections)
        reward += cfg.w_residual_action * abs(action)

        # Action change penalty (discourage oscillation)
        reward += cfg.w_action_change * abs(action - self._prev_action)

        self._prev_heading = current_abs_heading
        self._prev_action = action

        # Terminal conditions
        done = False

        # Success: segment complete
        if self._d >= cfg.segment_length_cm:
            done = True
            if (
                current_abs_heading < 5.0
                and enc_diff < 2.0
            ):
                reward += cfg.success_reward
                info["reason"] = "success"
            else:
                reward += cfg.success_reward * 0.5
                info["reason"] = "success_with_drift"

        # Collision (forward ultrasonic)
        elif (
            self._direction > 0
            and front_us < cfg.ultrasonic_stop_threshold_cm
        ):
            done = True
            reward += cfg.collision_penalty
            info["reason"] = "collision"

        # Excessive heading error
        elif current_abs_heading > cfg.max_heading_error_deg:
            done = True
            reward += cfg.heading_limit_penalty
            info["reason"] = "heading_limit"

        # Left corridor
        elif abs(self._x) > cfg.corridor_half_width_cm:
            done = True
            reward += cfg.corridor_penalty
            info["reason"] = "corridor_exit"

        # Stall
        elif (
            self._step_count > 10
            and distance_step < cfg.stall_threshold_cm
        ):
            done = True
            reward += cfg.stall_penalty
            info["reason"] = "stall"

        # Max steps
        elif self._step_count >= cfg.max_steps:
            done = True
            info["reason"] = "max_steps"

        self._done = done
        info["step"] = self._step_count
        info["heading_deg"] = self._heading
        info["lateral_cm"] = self._x
        info["distance_cm"] = self._d
        info["encoder_diff_cm"] = self._left_dist - self._right_dist

        return obs, reward, done, info

    def _get_observation(self) -> np.ndarray:
        """Build and normalise the 5-dim observation."""
        cfg = self.cfg

        # Front ultrasonic
        if self._direction > 0:
            front_us = max(0.0, cfg.wall_distance_cm - self._d)
        else:
            front_us = cfg.wall_distance_cm  # no rear sensor

        obs = LibraryObservationV1(
            motion_direction=self._direction,
            segment_progress=min(1.0, self._d / self._target_distance),
            fused_heading_error=self._heading,
            left_right_encoder_error=self._left_dist - self._right_dist,
            front_ultrasonic_distance=min(front_us, cfg.ultrasonic_max_cm),
        )
        return obs.normalize()

    def get_raw_observation(self) -> LibraryObservationV1:
        """Return the current unprocessed observation (for logging)."""
        cfg = self.cfg
        if self._direction > 0:
            front_us = max(0.0, cfg.wall_distance_cm - self._d)
        else:
            front_us = cfg.wall_distance_cm
        return LibraryObservationV1(
            motion_direction=self._direction,
            segment_progress=min(1.0, self._d / self._target_distance),
            fused_heading_error=self._heading,
            left_right_encoder_error=self._left_dist - self._right_dist,
            front_ultrasonic_distance=min(front_us, cfg.ultrasonic_max_cm),
        )


# ======================================================================
# Evaluation helpers
# ======================================================================


def make_eval_scenarios(env: LibraryCorridorEnv) -> list[dict]:
    """Return deterministic evaluation scenario configurations."""
    return [
        {"name": "forward_nominal", "direction": 1.0},
        {"name": "backward_nominal", "direction": -1.0},
        {"name": "left_motor_weak", "direction": 1.0},
        {"name": "right_motor_weak", "direction": 1.0},
        {"name": "positive_heading_error", "direction": 1.0},
        {"name": "negative_heading_error", "direction": 1.0},
    ]


def run_eval_episode(
    env: LibraryCorridorEnv,
    agent,
    scenario: dict,
    max_steps: int = 200,
) -> dict:
    """Run a single deterministic evaluation episode."""
    obs = env.reset(direction=scenario.get("direction", 1.0))

    # Apply scenario-specific overrides
    name = scenario["name"]
    if name == "left_motor_weak":
        env._left_gain = 0.85
    elif name == "right_motor_weak":
        env._right_gain = 0.85
    elif name == "positive_heading_error":
        env._heading = 8.0
    elif name == "negative_heading_error":
        env._heading = -8.0

    obs = env._get_observation()
    total_reward = 0.0
    actions = []
    headings = []
    steps = 0

    for _ in range(max_steps):
        action = agent.select_action(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        actions.append(abs(action))
        headings.append(abs(info.get("heading_deg", 0.0)))
        steps += 1
        if done:
            break

    sign_changes = sum(
        1
        for i in range(1, len(actions))
        if (actions[i] > 0) != (actions[i - 1] > 0)
    ) if len(actions) > 1 else 0

    return {
        "scenario": name,
        "return": total_reward,
        "success": info.get("reason", "").startswith("success"),
        "steps": steps,
        "reason": info.get("reason", ""),
        "mean_abs_heading": float(np.mean(headings)) if headings else 0.0,
        "max_heading": float(np.max(headings)) if headings else 0.0,
        "mean_residual": float(np.mean(actions)) if actions else 0.0,
        "oscillation_count": sign_changes,
    }
