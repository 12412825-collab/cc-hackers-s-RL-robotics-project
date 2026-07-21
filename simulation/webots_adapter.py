"""Webots device adapter exposed as a DonkeyCar Part.

This module must run inside a Webots Python controller. Webots owns the
simulation clock, so the adapter intentionally runs synchronously in the
DonkeyCar Vehicle loop.
"""

import logging
import math

import numpy as np

from parts.differential_drive import DifferentialDriveKinematics
from parts.simulation_control import acknowledge_reset, consume_reset


logger = logging.getLogger(__name__)


class WebotsAdapter:
    """Drive a differential robot and publish DonkeyCar-compatible channels."""

    def __init__(self, cfg):
        try:
            from controller import Supervisor
        except ImportError as exc:
            raise ImportError(
                "Webots Python controller API is unavailable. Start manage.py "
                "as the robot controller from Webots, or add WEBOTS_HOME/lib/" 
                "controller/python to PYTHONPATH."
            ) from exc

        self.cfg = cfg
        self.robot = Supervisor()
        self.timestep = int(getattr(
            cfg, 'WEBOTS_TIMESTEP_MS', self.robot.getBasicTimeStep()))
        self.dt = self.timestep / 1000.0
        self.kinematics = DifferentialDriveKinematics(
            cfg.WHEEL_RADIUS,
            cfg.WHEEL_SEPARATION,
            getattr(cfg, 'MAX_WHEEL_SPEED', None))

        left_motor_names = getattr(
            cfg, 'WEBOTS_LEFT_MOTORS', [cfg.WEBOTS_LEFT_MOTOR])
        right_motor_names = getattr(
            cfg, 'WEBOTS_RIGHT_MOTORS', [cfg.WEBOTS_RIGHT_MOTOR])
        self.left_motors = [self._device(name) for name in left_motor_names]
        self.right_motors = [self._device(name) for name in right_motor_names]
        for motor in self.left_motors + self.right_motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(0.0)

        left_encoder_names = getattr(
            cfg, 'WEBOTS_LEFT_ENCODERS',
            [getattr(cfg, 'WEBOTS_LEFT_ENCODER', 'left wheel sensor')])
        right_encoder_names = getattr(
            cfg, 'WEBOTS_RIGHT_ENCODERS',
            [getattr(cfg, 'WEBOTS_RIGHT_ENCODER', 'right wheel sensor')])
        self.left_encoders = [self._optional_device(name)
                              for name in left_encoder_names]
        self.right_encoders = [self._optional_device(name)
                               for name in right_encoder_names]
        self.left_encoders = [sensor for sensor in self.left_encoders
                              if sensor is not None]
        self.right_encoders = [sensor for sensor in self.right_encoders
                               if sensor is not None]
        self.camera = self._optional_device(
            getattr(cfg, 'WEBOTS_CAMERA', 'camera'))
        self.accelerometer = self._optional_device(
            getattr(cfg, 'WEBOTS_ACCELEROMETER', 'accelerometer'))
        self.gyro = self._optional_device(getattr(cfg, 'WEBOTS_GYRO', 'gyro'))
        self.distance_sensor = self._optional_device(
            getattr(cfg, 'WEBOTS_DISTANCE_SENSOR', 'front distance sensor'))

        for sensor in (self.left_encoders + self.right_encoders +
                       [self.camera, self.accelerometer, self.gyro,
                        self.distance_sensor]):
            if sensor is not None:
                sensor.enable(self.timestep)

        self.self_node = self.robot.getSelf()
        self.initial_translation = list(self.self_node.getPosition())
        self.initial_rotation = [0.0, 1.0, 0.0, 0.0]
        if hasattr(self.self_node, 'getField'):
            rotation_field = self.self_node.getField('rotation')
            if rotation_field is not None and hasattr(rotation_field, 'getSFRotation'):
                self.initial_rotation = list(rotation_field.getSFRotation())
        self.max_position_m = float(getattr(cfg, 'WEBOTS_GEOFENCE_M', 45.0))
        self.min_height_m = float(getattr(cfg, 'WEBOTS_MIN_HEIGHT_M', -0.25))
        self.previous_left_positions = None
        self.previous_right_positions = None
        self.last_image = np.zeros(
            (cfg.IMAGE_H, cfg.IMAGE_W, cfg.IMAGE_DEPTH), dtype=np.uint8)
        self.terminated = False
        self.state_was_reset = False

        # Prime sensors once before the DonkeyCar pipeline consumes them.
        if self.robot.step(self.timestep) == -1:
            self.terminated = True

        logger.info(
            "WebotsAdapter ready: timestep=%d ms, wheel_radius=%.4f m, "
            "wheel_separation=%.4f m",
            self.timestep, cfg.WHEEL_RADIUS, cfg.WHEEL_SEPARATION)

    def _device(self, name):
        device = self.robot.getDevice(name)
        if device is None:
            raise ValueError(f"Required Webots device not found: {name}")
        return device

    def _optional_device(self, name):
        if not name:
            return None
        try:
            return self.robot.getDevice(name)
        except Exception:
            logger.warning("Optional Webots device not found: %s", name)
            return None

    def _camera_image(self):
        if self.camera is None:
            return self.last_image
        raw = self.camera.getImage()
        if raw is None:
            return self.last_image
        height = self.camera.getHeight()
        width = self.camera.getWidth()
        bgra = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
        image = bgra[:, :, :3].copy()
        if image.shape[:2] != (self.cfg.IMAGE_H, self.cfg.IMAGE_W):
            try:
                import cv2
                image = cv2.resize(image, (self.cfg.IMAGE_W, self.cfg.IMAGE_H))
            except ImportError as exc:
                raise RuntimeError(
                    "Webots camera size differs from IMAGE_W/IMAGE_H and "
                    "OpenCV is unavailable for resizing") from exc
        self.last_image = image
        return image

    def _wheel_speeds(self):
        if not self.left_encoders or not self.right_encoders:
            return 0.0, 0.0, 0.0
        left_positions = [float(sensor.getValue())
                          for sensor in self.left_encoders]
        right_positions = [float(sensor.getValue())
                           for sensor in self.right_encoders]
        if self.previous_left_positions is None:
            left_rate = right_rate = 0.0
        else:
            left_rate = float(np.mean([
                (position - previous) / self.dt
                for position, previous in zip(
                    left_positions, self.previous_left_positions)]))
            right_rate = float(np.mean([
                (position - previous) / self.dt
                for position, previous in zip(
                    right_positions, self.previous_right_positions)]))
        self.previous_left_positions = left_positions
        self.previous_right_positions = right_positions
        speed = self.cfg.WHEEL_RADIUS * (left_rate + right_rate) / 2.0
        return left_rate, right_rate, speed

    def _imu(self):
        accel = ([0.0, 0.0, 0.0] if self.accelerometer is None
                 else [value / 9.80665
                       for value in self.accelerometer.getValues()])
        gyro = ([0.0, 0.0, 0.0] if self.gyro is None
                else list(self.gyro.getValues()))
        # Webots gyro is rad/s; existing DonkeyCar channel convention is deg/s.
        gyro = [math.degrees(value) for value in gyro]
        return accel + gyro

    def _distance_cm(self):
        if self.distance_sensor is None:
            return None
        scale = float(getattr(self.cfg, 'WEBOTS_DISTANCE_TO_CM', 100.0))
        return float(self.distance_sensor.getValue()) * scale

    def _ground_truth(self):
        if self.self_node is None:
            return 0.0, 0.0, 0.0, 0.0
        position = self.self_node.getPosition()
        velocity = self.self_node.getVelocity()
        speed = math.hypot(velocity[0], velocity[2])
        axis = getattr(self.cfg, 'WEBOTS_CTE_AXIS', 'x').lower()
        axis_index = 0 if axis == 'x' else 2
        center = float(getattr(self.cfg, 'WEBOTS_TRACK_CENTER', 0.0))
        cte = float(position[axis_index]) - center
        return float(position[0]), float(position[1]), float(position[2]), speed, cte

    def run(self, linear_velocity=0.0, angular_velocity=0.0):
        if self.terminated:
            return self._outputs()
        if consume_reset():
            self._reset_robot("dashboard request")
            self.state_was_reset = True
            acknowledge_reset()
            return self._outputs()
        left, right = self.kinematics.run(linear_velocity, angular_velocity)
        for motor in self.left_motors:
            motor.setVelocity(left)
        for motor in self.right_motors:
            motor.setVelocity(right)
        if self.robot.step(self.timestep) == -1:
            self.terminated = True
        self.state_was_reset = self._enforce_geofence()
        return self._outputs()

    def _enforce_geofence(self):
        """Recover from a fall or runaway command without corrupting a run."""
        if self.self_node is None:
            return False
        position = self.self_node.getPosition()
        invalid = (not all(math.isfinite(value) for value in position) or
                   abs(position[0]) > self.max_position_m or
                   abs(position[2]) > self.max_position_m or
                   position[1] < self.min_height_m)
        if not invalid:
            return False
        logger.warning("Webots geofence reset at position %s", position)
        self._reset_robot("geofence")
        return True

    def _reset_robot(self, reason):
        logger.warning("Resetting Webots robot: %s", reason)
        for motor in self.left_motors + self.right_motors:
            motor.setVelocity(0.0)
        if self.self_node is not None and hasattr(self.self_node, 'getField'):
            self.self_node.getField('translation').setSFVec3f(
                self.initial_translation)
            self.self_node.getField('rotation').setSFRotation(
                self.initial_rotation)
        if self.self_node is not None and hasattr(self.self_node, 'resetPhysics'):
            self.self_node.resetPhysics()
        self.previous_left_positions = None
        self.previous_right_positions = None

    def _outputs(self):
        image = self._camera_image()
        left_rate, right_rate, speed = self._wheel_speeds()
        imu = self._imu()
        distance = self._distance_cm()
        if self.state_was_reset:
            pos_x, pos_y, pos_z = self.initial_translation
            true_speed = 0.0
            axis = getattr(self.cfg, 'WEBOTS_CTE_AXIS', 'x').lower()
            center = float(getattr(self.cfg, 'WEBOTS_TRACK_CENTER', 0.0))
            cte = (pos_x if axis == 'x' else pos_z) - center
            self.state_was_reset = False
        else:
            pos_x, pos_y, pos_z, true_speed, cte = self._ground_truth()
        return (image, left_rate, right_rate, speed,
                imu[0], imu[1], imu[2], imu[3], imu[4], imu[5],
                distance, pos_x, pos_y, pos_z, true_speed, cte)

    def shutdown(self):
        for motor in self.left_motors + self.right_motors:
            motor.setVelocity(0.0)
