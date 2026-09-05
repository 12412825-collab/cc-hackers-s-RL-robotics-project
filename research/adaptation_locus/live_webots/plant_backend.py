"""Live Webots device backend — inherits historical WebotsAdapter semantics.

Classification:
  H — device names, units, kinematics, resetPhysics pattern from WebotsAdapter
  C — must run inside Webots controller Python (R2025a)
  R — research episode/reset/seed wrappers
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

from parts.differential_drive import DifferentialDriveKinematics


MOTOR_LEFT = ("front left wheel motor", "rear left wheel motor")
MOTOR_RIGHT = ("front right wheel motor", "rear right wheel motor")
ENC_LEFT = ("front left wheel sensor", "rear left wheel sensor")
ENC_RIGHT = ("front right wheel sensor", "rear right wheel sensor")


def yaw_from_rotation(axis_angle) -> float:
    """Yaw about +Y (NUE) from Webots SFRotation."""
    ax, ay, az, angle = [float(v) for v in axis_angle]
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    r00 = t * ax * ax + c
    r02 = t * ax * az - s * ay
    return math.atan2(r02, r00)


@dataclass
class SensorPacket:
    sim_time_s: float
    accel_g: list[float]
    gyro_deg_s: list[float]
    gyro_yaw_rate_rad_s: float
    left_rad_s: float
    right_rad_s: float
    speed_m_s: float
    distance_cm: Optional[float]
    # Privileged (Supervisor) — must not enter controller obs builder blindly
    true_position_m: list[float]
    true_yaw_rad: float
    true_speed_m_s: float


class LiveWebotsBackend:
    """Thin Supervisor I/O. Pose advances only via robot.step()."""

    def __init__(
        self,
        wheel_radius_m: float = 0.0325,
        wheel_separation_m: float = 0.130,
        max_wheel_speed: float = 12.0,
        spawn_translation: Optional[list[float]] = None,
        spawn_rotation: Optional[list[float]] = None,
    ):
        from controller import Supervisor  # Webots-only import

        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.dt = self.timestep / 1000.0
        self.kinematics = DifferentialDriveKinematics(
            wheel_radius_m, wheel_separation_m, max_wheel_speed
        )
        self.self_node = self.robot.getSelf()
        if self.self_node is None:
            raise RuntimeError("Supervisor.getSelf() returned None")

        self.left_motors = [self._require(name) for name in MOTOR_LEFT]
        self.right_motors = [self._require(name) for name in MOTOR_RIGHT]
        for m in self.left_motors + self.right_motors:
            m.setPosition(float("inf"))
            m.setVelocity(0.0)

        self.left_encoders = [self._require(name) for name in ENC_LEFT]
        self.right_encoders = [self._require(name) for name in ENC_RIGHT]
        self.accel = self.robot.getDevice("accelerometer")
        self.gyro = self.robot.getDevice("gyro")
        self.distance = self.robot.getDevice("front distance sensor")
        for sens in list(self.left_encoders) + list(self.right_encoders):
            sens.enable(self.timestep)
        if self.accel is not None:
            self.accel.enable(self.timestep)
        if self.gyro is not None:
            self.gyro.enable(self.timestep)
        if self.distance is not None:
            self.distance.enable(self.timestep)

        self.spawn_translation = list(
            spawn_translation
            if spawn_translation is not None
            else self.self_node.getField("translation").getSFVec3f()
        )
        self.spawn_rotation = list(
            spawn_rotation
            if spawn_rotation is not None
            else self.self_node.getField("rotation").getSFRotation()
        )
        self._prev_left = None
        self._prev_right = None
        self._last_cmd_left = 0.0
        self._last_cmd_right = 0.0
        self.terminated = False

        # Prime sensors (historical WebotsAdapter pattern)
        if self.robot.step(self.timestep) == -1:
            self.terminated = True

    def _require(self, name: str):
        dev = self.robot.getDevice(name)
        if dev is None:
            raise ValueError(f"Required Webots device not found: {name}")
        return dev

    def reset_physics_state(self) -> None:
        """Restore spawn pose + ODE state (historical _reset_robot pattern)."""
        for m in self.left_motors + self.right_motors:
            m.setVelocity(0.0)
        self.self_node.getField("translation").setSFVec3f(list(self.spawn_translation))
        self.self_node.getField("rotation").setSFRotation(list(self.spawn_rotation))
        if hasattr(self.self_node, "resetPhysics"):
            self.self_node.resetPhysics()
        self._prev_left = None
        self._prev_right = None
        self._last_cmd_left = 0.0
        self._last_cmd_right = 0.0
        # Settle contacts
        for _ in range(5):
            if self.robot.step(self.timestep) == -1:
                self.terminated = True
                break

    def apply_wheel_speeds(self, left_rad_s: float, right_rad_s: float) -> None:
        self._last_cmd_left = float(left_rad_s)
        self._last_cmd_right = float(right_rad_s)
        for m in self.left_motors:
            m.setVelocity(self._last_cmd_left)
        for m in self.right_motors:
            m.setVelocity(self._last_cmd_right)

    def apply_body_velocity(self, v_m_s: float, omega_rad_s: float) -> tuple[float, float]:
        left, right = self.kinematics.run(v_m_s, omega_rad_s)
        self.apply_wheel_speeds(left, right)
        return left, right

    def step_physics(self) -> bool:
        """Advance Webots ODE by one basicTimeStep. Returns False if quit."""
        if self.robot.step(self.timestep) == -1:
            self.terminated = True
            return False
        return True

    def read_sensors(self) -> SensorPacket:
        left_pos = [float(s.getValue()) for s in self.left_encoders]
        right_pos = [float(s.getValue()) for s in self.right_encoders]
        if self._prev_left is None:
            left_rate = right_rate = 0.0
        else:
            left_rate = float(
                sum((a - b) / self.dt for a, b in zip(left_pos, self._prev_left))
                / len(left_pos)
            )
            right_rate = float(
                sum((a - b) / self.dt for a, b in zip(right_pos, self._prev_right))
                / len(right_pos)
            )
        self._prev_left = left_pos
        self._prev_right = right_pos
        speed = self.kinematics.wheel_radius * (left_rate + right_rate) / 2.0

        if self.accel is None:
            accel_g = [0.0, 0.0, 0.0]
        else:
            accel_g = [float(v) / 9.80665 for v in self.accel.getValues()]
        if self.gyro is None:
            gyro_rad = [0.0, 0.0, 0.0]
        else:
            gyro_rad = [float(v) for v in self.gyro.getValues()]
        # Historical WebotsAdapter: publish deg/s on Donkey channels
        gyro_deg = [math.degrees(v) for v in gyro_rad]
        # Yaw rate about Y (NUE): gyro Y component
        gyro_yaw_rate = gyro_rad[1] if len(gyro_rad) > 1 else 0.0

        dist_cm = None
        if self.distance is not None:
            dist_cm = float(self.distance.getValue()) * 100.0

        pos = list(self.self_node.getPosition())
        rot = list(self.self_node.getField("rotation").getSFRotation())
        yaw = yaw_from_rotation(rot)
        vel = self.self_node.getVelocity()
        true_speed = math.hypot(float(vel[0]), float(vel[2]))

        return SensorPacket(
            sim_time_s=float(self.robot.getTime()),
            accel_g=accel_g,
            gyro_deg_s=gyro_deg,
            gyro_yaw_rate_rad_s=float(gyro_yaw_rate),
            left_rad_s=left_rate,
            right_rad_s=right_rate,
            speed_m_s=float(speed),
            distance_cm=dist_cm,
            true_position_m=pos,
            true_yaw_rad=float(yaw),
            true_speed_m_s=float(true_speed),
        )

    def quit(self, code: int = 0) -> None:
        self.robot.simulationQuit(int(code))

    def provenance(self) -> dict[str, Any]:
        return {
            "plant": "live_webots_ode",
            "python_pose_integration": False,
            "timestep_ms": self.timestep,
            "uses_webots_faithful_env": False,
            "uses_phase0_env": False,
            "kinematics": "parts.differential_drive.DifferentialDriveKinematics",
            "device_names": {
                "left_motors": list(MOTOR_LEFT),
                "right_motors": list(MOTOR_RIGHT),
            },
        }
