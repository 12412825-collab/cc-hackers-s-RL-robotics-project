"""
Multi-Sensor Fusion Module for Intelligent Mobile Robot
========================================================

Provides sensor abstraction, data collection, timestamp synchronization,
and observation vector construction for the RL decision-making module.

Architecture:
  Raw Sensors (Encoder/IMU/HC-SR04)
    → Individual Sensor Wrappers (calibration + filtering)
    → SensorFusion (timestamp alignment + normalization)
    → Unified Observation Vector → SAC Multi-Modal Input

Sensor Vector Layout (9-dim):
  Index | Sensor          | Description                    | Unit
  ------|-----------------|--------------------------------|------
  0     | encoder_speed   | Vehicle speed from encoder     | m/s (normalized)
  1     | encoder_accel   | Acceleration (derived)         | m/s² (normalized)
  2     | imu_accel_x     | IMU accelerometer X-axis       | g (normalized)
  3     | imu_accel_y     | IMU accelerometer Y-axis       | g
  4     | imu_accel_z     | IMU accelerometer Z-axis       | g
  5     | imu_gyro_x      | IMU gyroscope X-axis           | deg/s (normalized)
  6     | imu_gyro_y      | IMU gyroscope Y-axis           | deg/s
  7     | imu_gyro_z      | IMU gyroscope Z-axis           | deg/s
  8     | obstacle_dist   | HC-SR04 ultrasonic distance    | cm (normalized)

Design Principles:
  - Each sensor is an independent wrapper (configurable, testable in isolation)
  - SensorFusion handles timestamp alignment via nearest-neighbor interpolation
  - Missing sensors are filled with zeros (robust to partial sensor availability)
  - All outputs normalized to [-1, 1] or [0, 1] for stable RL training
  - Duck-typed Part interface: each sensor exposes run() method
  - This module is the PERCEPTION LAYER — DL perception + RL decision above it
"""

import time
import logging
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ===========================================================================
# Sensor Configuration
# ===========================================================================

@dataclass
class SensorConfig:
    """Configuration for all sensors and fusion parameters."""

    # --- Sensor enable flags ---
    ENABLE_ENCODER: bool = False        # Wheel encoder / odometry
    ENABLE_IMU: bool = False            # MPU6050 / MPU9250 IMU
    ENABLE_OBSTACLE: bool = False       # HC-SR04 ultrasonic distance sensor

    # --- Encoder ---
    ENCODER_TYPE: str = 'GPIO'          # 'GPIO' | 'Arduino' | 'Simulated'
    ENCODER_MM_PER_TICK: float = 12.7625  # mm travel per encoder tick
    ENCODER_MAX_SPEED: float = 5.0      # m/s, for normalization

    # --- IMU ---
    IMU_SENSOR: str = 'mpu6050'         # 'mpu6050' | 'mpu9250' | 'icm20948'
    IMU_ACCEL_RANGE: float = 2.0        # g (±2g default for MPU6050)
    IMU_GYRO_RANGE: float = 250.0       # deg/s (±250 deg/s default)

    # --- HC-SR04 Ultrasonic ---
    OBSTACLE_TYPE: str = 'HC_SR04'      # 'HC_SR04' | 'tfmini' | 'infrared'
    OBSTACLE_MAX_DIST: float = 400.0    # cm, for normalization
    OBSTACLE_MIN_DIST: float = 2.0      # cm, minimum detectable distance

    # --- Fusion ---
    SENSOR_FUSION_RATE: float = 20.0    # Hz, sensor fusion output rate
    SENSOR_BUFFER_SIZE: int = 10        # rolling buffer for temporal smoothing
    SENSOR_DIM: int = 9                 # output observation vector dimension

    # --- Default normalization ranges (per sensor channel) ---
    # Used for clipping and normalization to [-1, 1]
    NORM_RANGES: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'encoder_speed':     (-1.0, 1.0),   # after / MAX_SPEED
        'encoder_accel':     (-1.0, 1.0),   # after / (MAX_SPEED * 2)
        'imu_accel':         (-2.0, 2.0),   # g
        'imu_gyro':          (-250.0, 250.0), # deg/s
        'obstacle_dist':     (0.0, 400.0),  # cm → normalize to [0, 1] then 2x-1 to [-1, 1]
    })


# ===========================================================================
# Individual Sensor Wrappers
# ===========================================================================

class EncoderSensor:
    """
    Wheel encoder sensor wrapper.

    Input:  encoder ticks from GPIO/Arduino (via 'enc/speed' memory channel)
    Output: normalized speed + derived acceleration

    For DonkeyCar integration, reads from 'enc/speed' channel
    if HAVE_ODOM=True in config.
    """

    def __init__(self, config: SensorConfig):
        self.cfg = config
        self.max_speed = config.ENCODER_MAX_SPEED
        self.last_speed = 0.0
        self.last_update = time.time()
        self.speed = 0.0
        self.accel = 0.0
        self.enabled = config.ENABLE_ENCODER

    def update(self, raw_speed: Optional[float] = None,
               raw_distance: Optional[float] = None) -> np.ndarray:
        """
        Process raw encoder data and return normalized [speed, accel].

        Args:
            raw_speed: speed in m/s from encoder (or None to use internal state)
            raw_distance: cumulative distance in m (optional)
        Returns:
            np.ndarray: [speed_norm, accel_norm] both in [-1, 1]
        """
        now = time.time()
        dt = max(now - self.last_update, 0.001)

        if raw_speed is not None:
            self.speed = raw_speed

        # Derive acceleration from speed change
        self.accel = (self.speed - self.last_speed) / dt
        self.last_speed = self.speed
        self.last_update = now

        # Normalize to [-1, 1]
        speed_norm = np.clip(self.speed / self.max_speed, -1.0, 1.0)
        accel_norm = np.clip(self.accel / (self.max_speed * 2.0), -1.0, 1.0)

        return np.array([speed_norm, accel_norm], dtype=np.float32)

    def reset(self):
        """Reset internal state."""
        self.speed = 0.0
        self.accel = 0.0
        self.last_speed = 0.0
        self.last_update = time.time()


class IMUSensor:
    """
    IMU (Inertial Measurement Unit) sensor wrapper.

    Input:  raw accelerometer + gyroscope readings
    Output: normalized [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]

    Supports MPU6050 (default), MPU9250, ICM20948.
    Reads from DonkeyCar memory channels: imu/acl_x, imu/acl_y, imu/acl_z,
                                          imu/gyr_x, imu/gyr_y, imu/gyr_z
    """

    def __init__(self, config: SensorConfig):
        self.cfg = config
        self.accel_range = config.IMU_ACCEL_RANGE    # ±g
        self.gyro_range = config.IMU_GYRO_RANGE      # ±deg/s
        self.enabled = config.ENABLE_IMU

        # Bias calibration (set via calibrate())
        self.accel_bias = np.zeros(3, dtype=np.float32)
        self.gyro_bias = np.zeros(3, dtype=np.float32)

        # Low-pass filter coefficient (simple exponential smoothing)
        self.alpha = 0.8  # smoothing factor: higher = more smoothing
        self.accel_filtered = np.zeros(3, dtype=np.float32)
        self.gyro_filtered = np.zeros(3, dtype=np.float32)

        # Latest readings
        self.accel = np.zeros(3, dtype=np.float32)
        self.gyro = np.zeros(3, dtype=np.float32)

    def calibrate(self, num_samples: int = 500, sample_delay: float = 0.002):
        """
        Calibrate IMU biases at rest.
        Call this once with the IMU stationary.

        Args:
            num_samples: number of samples for averaging
            sample_delay: delay between samples in seconds
        """
        logger.info(f"Calibrating IMU with {num_samples} samples...")

        accel_sum = np.zeros(3)
        gyro_sum = np.zeros(3)

        for _ in range(num_samples):
            accel_sum += self.accel
            gyro_sum += self.gyro
            time.sleep(sample_delay)

        self.accel_bias = accel_sum / num_samples
        self.gyro_bias = gyro_sum / num_samples

        # Account for gravity on Z-axis (should read ~1g when flat)
        self.accel_bias[2] -= 1.0

        logger.info(
            f"IMU Calibration complete:\n"
            f"  Accel bias: {self.accel_bias}\n"
            f"  Gyro bias:  {self.gyro_bias}"
        )

    def update(self, accel: Optional[Tuple[float, float, float]] = None,
               gyro: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
        """
        Process raw IMU data and return normalized 6-DOF vector.

        Args:
            accel: (ax, ay, az) in g, or None to use last values
            gyro:  (gx, gy, gz) in deg/s, or None to use last values
        Returns:
            np.ndarray: [ax, ay, az, gx, gy, gz] all normalized to [-1, 1]
        """
        if accel is not None:
            self.accel = np.array(accel, dtype=np.float32) - self.accel_bias
        if gyro is not None:
            self.gyro = np.array(gyro, dtype=np.float32) - self.gyro_bias

        # Exponential smoothing
        self.accel_filtered = (
            self.alpha * self.accel_filtered +
            (1 - self.alpha) * self.accel
        )
        self.gyro_filtered = (
            self.alpha * self.gyro_filtered +
            (1 - self.alpha) * self.gyro
        )

        # Normalize to [-1, 1]
        accel_norm = np.clip(self.accel_filtered / self.accel_range, -1.0, 1.0)
        gyro_norm = np.clip(self.gyro_filtered / self.gyro_range, -1.0, 1.0)

        return np.concatenate([accel_norm, gyro_norm]).astype(np.float32)

    def reset(self):
        """Reset filtered state."""
        self.accel_filtered = np.zeros(3, dtype=np.float32)
        self.gyro_filtered = np.zeros(3, dtype=np.float32)


class ObstacleSensor:
    """
    HC-SR04 Ultrasonic distance sensor wrapper.

    Input:  raw pulse duration → distance in cm
    Output: normalized distance in [-1, 1] (1 = far/safe, -1 = very close/danger)

    HC-SR04 specs: 2cm–400cm range, 5V, Trig/Echo pins.
    On Arduino: pulseIn(echo_pin, HIGH) / 58.0 → distance in cm.
    """

    def __init__(self, config: SensorConfig):
        self.cfg = config
        self.max_dist = config.OBSTACLE_MAX_DIST    # cm
        self.min_dist = config.OBSTACLE_MIN_DIST    # cm
        self.enabled = config.ENABLE_OBSTACLE

        self.distance = self.max_dist  # assume clear at start
        self.last_readings = deque(maxlen=5)  # median filter

    def update(self, raw_distance_cm: Optional[float] = None) -> np.ndarray:
        """
        Process raw distance reading.

        Args:
            raw_distance_cm: distance in cm, or None to use last value
        Returns:
            np.ndarray: [distance_norm] in [-1, 1]
                       1.0 = far (safe), -1.0 = very close (danger)
        """
        if raw_distance_cm is not None:
            self.last_readings.append(raw_distance_cm)
            # Median filter for noise rejection
            if len(self.last_readings) >= 3:
                self.distance = float(np.median(list(self.last_readings)))
            else:
                self.distance = raw_distance_cm

        # Clamp to valid range
        dist_clamped = np.clip(self.distance, self.min_dist, self.max_dist)

        # Normalize: [min_dist, max_dist] → [0, 1] → [1, -1] (closer = more negative)
        # This way the RL agent sees "danger" as negative values
        dist_01 = (dist_clamped - self.min_dist) / (self.max_dist - self.min_dist)
        dist_norm = 2.0 * dist_01 - 1.0  # close=-1, far=+1

        return np.array([dist_norm], dtype=np.float32)

    def is_obstacle_close(self, threshold_cm: float = 30.0) -> bool:
        """Check if obstacle is within threshold distance."""
        return self.distance < threshold_cm

    def reset(self):
        """Reset state."""
        self.distance = self.max_dist
        self.last_readings.clear()


class LineTrackingSensor:
    """
    IR line tracking sensor array.

    Monitors the line position under the vehicle for lane keeping.
    Typical configuration: 3–5 IR sensors in a row under the chassis.

    Input:  raw analog/digital readings per sensor (0=no line, 1=on line)
    Output: normalized sensor values + derived line position
    """

    def __init__(self, config: SensorConfig):
        self.cfg = config
        self.num_sensors = config.LINE_SENSOR_COUNT
        self.active_low = config.LINE_SENSOR_ACTIVE_LOW
        self.enabled = config.ENABLE_LINE_TRACKING

        # Latest readings
        self.values = np.zeros(self.num_sensors, dtype=np.float32)
        self.line_position = 0.0  # -1 (far left) to +1 (far right)

    def update(self, raw_values: Optional[List[float]] = None,
               digital: bool = True) -> np.ndarray:
        """
        Process raw line sensor readings.

        Args:
            raw_values: list of sensor readings (0/1 digital or 0–1023 analog)
            digital: True if digital (binary), False if analog
        Returns:
            np.ndarray: [left, center, right] summary values in [0, 1]
                        + derived line position
        """
        if raw_values is not None:
            raw = np.array(raw_values, dtype=np.float32)

            if self.active_low and digital:
                # Active low: LOW (0) means detected, HIGH (1) means not
                raw = 1.0 - raw

            if not digital:
                # Analog: normalize to [0, 1]
                raw = raw / 1023.0

            self.values = raw

        # Compute line position (weighted average of sensor indices)
        total = self.values.sum()
        if total > 0.01:
            indices = np.arange(self.num_sensors, dtype=np.float32)
            center_of_mass = np.sum(indices * self.values) / total
            # Normalize to [-1, 1]
            self.line_position = (
                2.0 * center_of_mass / (self.num_sensors - 1) - 1.0
            )
        else:
            self.line_position = 0.0  # no line detected

        # Return 3 representative values: left, center, right segments
        n = self.num_sensors
        third = max(n // 3, 1)
        left_val = float(np.mean(self.values[:third]))
        center_val = float(
            np.mean(self.values[third:min(2*third, n)])
        )
        right_val = float(np.mean(self.values[min(2*third, n):]))

        return np.array([left_val, center_val, right_val], dtype=np.float32)

    def get_line_position(self) -> float:
        """Return derived line position in [-1, 1]."""
        return self.line_position

    def reset(self):
        """Reset state."""
        self.values = np.zeros(self.num_sensors, dtype=np.float32)
        self.line_position = 0.0


# ===========================================================================
# Sensor Fusion — unified observation vector
# ===========================================================================

class SensorFusion:
    """
    Multi-sensor fusion module that collects readings from all available
    sensors, performs timestamp alignment, normalization, and outputs a
    unified observation vector for the RL agent.

    This is the central PERCEPTION LAYER. It converts raw sensor data into
    a structured observation that the RL decision-making module consumes.

    DonkeyCar Part Interface:
        V.add(fusion, inputs=[
            'enc/speed',
            'imu/acl_x', 'imu/acl_y', 'imu/acl_z',
            'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z',
            'obs/distance',
            'line/raw_values',
        ], outputs=['sensor/observation'])

    The output 'sensor/observation' is a 1D numpy array (sensor_dim,)
    that feeds into the SAC multi-modal input alongside the camera image.
    """

    def __init__(self, config: SensorConfig):
        self.cfg = config

        # Sensor dimension
        self.sensor_dim = config.SENSOR_DIM

        # Individual sensor wrappers
        self.encoder = EncoderSensor(config) if config.ENABLE_ENCODER else None
        self.imu = IMUSensor(config) if config.ENABLE_IMU else None
        self.obstacle = ObstacleSensor(config) if config.ENABLE_OBSTACLE else None

        # Active sensor count
        self.active_sensors = sum([
            config.ENABLE_ENCODER,
            config.ENABLE_IMU,
            config.ENABLE_OBSTACLE,
        ])
        logger.info(
            f"SensorFusion: {self.active_sensors} sensor(s) enabled, "
            f"output dim = {self.sensor_dim}"
        )

        # Statistics
        self.update_count = 0
        self.last_observation = np.zeros(self.sensor_dim, dtype=np.float32)

    def update_encoder(self, speed: Optional[float] = None):
        """Update encoder from DonkeyCar memory value."""
        if self.encoder:
            self.encoder.update(raw_speed=speed)

    def update_imu(self, accel_x=None, accel_y=None, accel_z=None,
                   gyro_x=None, gyro_y=None, gyro_z=None):
        """Update IMU from DonkeyCar memory channels."""
        if self.imu:
            accel = None
            gyro = None
            if all(v is not None for v in [accel_x, accel_y, accel_z]):
                accel = (accel_x, accel_y, accel_z)
            if all(v is not None for v in [gyro_x, gyro_y, gyro_z]):
                gyro = (gyro_x, gyro_y, gyro_z)
            self.imu.update(accel=accel, gyro=gyro)

    def update_obstacle(self, distance: Optional[float] = None):
        """Update obstacle distance from DonkeyCar memory value."""
        if self.obstacle:
            self.obstacle.update(raw_distance_cm=distance)

    def get_observation(self) -> np.ndarray:
        """
        Build the unified observation vector from all active sensors.

        Returns:
            np.ndarray: (sensor_dim,) float32, normalized to [-1, 1]
                         Inactive sensor channels are filled with zeros.

        Layout (9-dim):
            [0:2]   = encoder [speed, accel]
            [2:8]   = IMU [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
            [8]     = HC-SR04 ultrasonic [distance]
        """
        obs = np.zeros(self.sensor_dim, dtype=np.float32)
        idx = 0

        # Encoder: 2 values → obs[0:2]
        if self.encoder and self.encoder.enabled:
            enc_data = self.encoder.update()
            obs[idx:idx+2] = enc_data
        idx += 2

        # IMU: 6 values → obs[2:8]
        if self.imu and self.imu.enabled:
            imu_data = self.imu.update()
            obs[idx:idx+6] = imu_data
        idx += 6

        # HC-SR04 Ultrasonic: 1 value → obs[8]
        if self.obstacle and self.obstacle.enabled:
            obs_data = self.obstacle.update()
            obs[idx:idx+1] = obs_data

        self.last_observation = obs
        self.update_count += 1
        return obs

    # ---- DonkeyCar Part Interface ----

    def run(self,
            enc_speed=None,
            imu_acl_x=None, imu_acl_y=None, imu_acl_z=None,
            imu_gyr_x=None, imu_gyr_y=None, imu_gyr_z=None,
            obs_distance=None) -> np.ndarray:
        """
        DonkeyCar Part run() method.
        Reads from named memory channels and outputs unified observation.

        Wiring:
            V.add(fusion,
                  inputs=['enc/speed',
                          'imu/acl_x', 'imu/acl_y', 'imu/acl_z',
                          'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z',
                          'obs/distance'],
                  outputs=['sensor/observation'])
        """
        obs = np.zeros(self.sensor_dim, dtype=np.float32)
        if self.encoder and self.encoder.enabled:
            obs[0:2] = self.encoder.update(raw_speed=enc_speed)

        if self.imu and self.imu.enabled:
            accel = None
            gyro = None
            if all(value is not None for value in
                   [imu_acl_x, imu_acl_y, imu_acl_z]):
                accel = (imu_acl_x, imu_acl_y, imu_acl_z)
            if all(value is not None for value in
                   [imu_gyr_x, imu_gyr_y, imu_gyr_z]):
                gyro = (imu_gyr_x, imu_gyr_y, imu_gyr_z)
            obs[2:8] = self.imu.update(accel=accel, gyro=gyro)

        if self.obstacle and self.obstacle.enabled:
            obs[8:9] = self.obstacle.update(raw_distance_cm=obs_distance)

        self.last_observation = obs
        self.update_count += 1
        return obs.copy()

    def reset(self):
        """Reset all sensor internal states."""
        for sensor in [self.encoder, self.imu, self.obstacle]:
            if sensor:
                sensor.reset()
        self.last_observation = np.zeros(self.sensor_dim, dtype=np.float32)
        self.update_count = 0

    def shutdown(self):
        """Cleanup sensor resources."""
        self.reset()


# ===========================================================================
# Sensor Buffer — temporal smoothing for offline/online training
# ===========================================================================

class SensorBuffer:
    """
    Rolling buffer for sensor observation history.

    Used for:
      1. Temporal smoothing (moving average over recent observations)
      2. Storing sensor context alongside images in replay buffer
      3. Handling sensor dropout (fill gaps with last known values)
    """

    def __init__(self, sensor_dim: int = 12, buffer_size: int = 10):
        self.sensor_dim = sensor_dim
        self.buffer = deque(maxlen=buffer_size)
        # Initialize with zeros
        for _ in range(buffer_size):
            self.buffer.append(np.zeros(sensor_dim, dtype=np.float32))

    def push(self, observation: np.ndarray):
        """Add a new observation to the buffer."""
        self.buffer.append(observation.astype(np.float32))

    def get_latest(self) -> np.ndarray:
        """Get the most recent observation."""
        return self.buffer[-1].copy()

    def get_smoothed(self, window: int = 5) -> np.ndarray:
        """
        Get exponentially smoothed observation.

        Args:
            window: number of recent samples to average
        Returns:
            np.ndarray: smoothed observation vector
        """
        w = min(window, len(self.buffer))
        recent = list(self.buffer)[-w:]
        weights = np.exp(np.linspace(-2, 0, w))  # exponential weights
        weights /= weights.sum()
        return np.average(recent, axis=0, weights=weights).astype(np.float32)

    def __len__(self):
        return len(self.buffer)


# ===========================================================================
# Sensor Normalizer — running mean/std for observation normalization
# ===========================================================================

class RunningNormalizer:
    """
    Running mean and standard deviation tracker for sensor observations.
    Used to normalize sensor inputs to zero mean, unit variance during RL training.

    This improves SAC training stability by ensuring sensor features
    are on a consistent scale regardless of raw sensor ranges.
    """

    def __init__(self, shape: Tuple[int, ...], epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon  # start with small count for stability
        self.epsilon = epsilon

    def update(self, x: np.ndarray):
        """Update running statistics with a batch of observations."""
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean.astype(np.float32)
        self.var = new_var.astype(np.float32)
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize observation to zero mean, unit variance."""
        return (x - self.mean) / (np.sqrt(self.var) + self.epsilon)


# ===========================================================================
# Factory function
# ===========================================================================

def create_sensor_fusion(cfg) -> SensorFusion:
    """
    Create a SensorFusion instance from a DonkeyCar config object.

    Maps cfg attributes to SensorConfig fields.
    Works with both the new SensorConfig dataclass and legacy myconfig.py attributes.

    Usage:
        from parts.sensors import create_sensor_fusion
        fusion = create_sensor_fusion(cfg)
    """
    sensor_cfg = SensorConfig(
        ENABLE_ENCODER=getattr(cfg, 'ENABLE_ENCODER', getattr(cfg, 'HAVE_ODOM', False)),
        ENABLE_IMU=getattr(cfg, 'ENABLE_IMU', getattr(cfg, 'HAVE_IMU', False)),
        ENABLE_OBSTACLE=getattr(cfg, 'ENABLE_OBSTACLE', getattr(cfg, 'HAVE_TFMINI', False)),

        ENCODER_TYPE=getattr(cfg, 'ENCODER_TYPE', 'GPIO'),
        ENCODER_MM_PER_TICK=getattr(cfg, 'ENCODER_MM_PER_TICK', 12.7625),
        ENCODER_MAX_SPEED=getattr(cfg, 'ENCODER_MAX_SPEED', 5.0),

        IMU_SENSOR=getattr(cfg, 'IMU_SENSOR', 'mpu6050'),
        IMU_ACCEL_RANGE=getattr(cfg, 'IMU_ACCEL_RANGE', 2.0),
        IMU_GYRO_RANGE=getattr(cfg, 'IMU_GYRO_RANGE', 250.0),

        OBSTACLE_TYPE=getattr(cfg, 'OBSTACLE_TYPE', 'HC_SR04'),
        OBSTACLE_MAX_DIST=getattr(cfg, 'OBSTACLE_MAX_DIST', 400.0),
        OBSTACLE_MIN_DIST=getattr(cfg, 'OBSTACLE_MIN_DIST', 2.0),

        SENSOR_FUSION_RATE=getattr(cfg, 'SENSOR_FUSION_RATE', 20.0),
        SENSOR_BUFFER_SIZE=getattr(cfg, 'SENSOR_BUFFER_SIZE', 10),
        SENSOR_DIM=getattr(cfg, 'SENSOR_DIM', 9),
    )
    return SensorFusion(sensor_cfg)
