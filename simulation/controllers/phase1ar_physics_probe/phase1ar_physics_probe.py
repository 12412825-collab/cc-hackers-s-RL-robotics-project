"""Phase-1A-R Step 1: live Webots physics provenance probe.

Runs inside the historical FourWheelRobot as a temporary controller override.
Reads pose ONLY via Supervisor getPosition / rotation field (Webots state).
Commands historical wheel motors, steps the Webots simulation, re-reads pose.
Does NOT integrate pose in Python.
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone


MOTOR_NAMES = (
    "front left wheel motor",
    "front right wheel motor",
    "rear left wheel motor",
    "rear right wheel motor",
)
LEFT_MOTORS = ("front left wheel motor", "rear left wheel motor")
RIGHT_MOTORS = ("front right wheel motor", "rear right wheel motor")

# Non-zero, asymmetric wheel speeds to induce translation + yaw.
LEFT_VEL = 4.0  # rad/s
RIGHT_VEL = 2.0  # rad/s
N_STEPS = 80


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _yaw_from_rotation(axis_angle) -> float:
    """Extract yaw about +Y from Webots SFRotation [ax, ay, az, angle]."""
    ax, ay, az, angle = [float(v) for v in axis_angle]
    # Rotation matrix R from axis-angle; yaw = atan2(R[0,2], R[0,0]) in Y-up.
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    r00 = t * ax * ax + c
    r02 = t * ax * az - s * ay
    return math.atan2(r02, r00)


def main() -> int:
    from controller import Supervisor

    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    self_node = robot.getSelf()
    if self_node is None:
        print("FATAL: Supervisor.getSelf() returned None", file=sys.stderr)
        return 2

    motors = {}
    for name in MOTOR_NAMES:
        device = robot.getDevice(name)
        if device is None:
            print(f"FATAL: motor not found: {name}", file=sys.stderr)
            return 3
        device.setPosition(float("inf"))
        device.setVelocity(0.0)
        motors[name] = device

    gyro = robot.getDevice("gyro")
    accel = robot.getDevice("accelerometer")
    if gyro is not None:
        gyro.enable(timestep)
    if accel is not None:
        accel.enable(timestep)

    # Prime one physics step before snapshot.
    if robot.step(timestep) == -1:
        print("FATAL: simulation terminated on prime step", file=sys.stderr)
        return 4

    rot_field = self_node.getField("rotation")
    pos0 = list(self_node.getPosition())
    rot0 = list(rot_field.getSFRotation()) if rot_field is not None else None
    yaw0 = _yaw_from_rotation(rot0) if rot0 is not None else None

    for name in LEFT_MOTORS:
        motors[name].setVelocity(LEFT_VEL)
    for name in RIGHT_MOTORS:
        motors[name].setVelocity(RIGHT_VEL)

    for _ in range(N_STEPS):
        if robot.step(timestep) == -1:
            print("FATAL: simulation terminated mid-probe", file=sys.stderr)
            return 5

    pos1 = list(self_node.getPosition())
    rot1 = list(rot_field.getSFRotation()) if rot_field is not None else None
    yaw1 = _yaw_from_rotation(rot1) if rot1 is not None else None

    for motor in motors.values():
        motor.setVelocity(0.0)

    dx = pos1[0] - pos0[0]
    dy = pos1[1] - pos0[1]
    dz = pos1[2] - pos0[2]
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    dyaw = None
    if yaw0 is not None and yaw1 is not None:
        dyaw = math.atan2(math.sin(yaw1 - yaw0), math.cos(yaw1 - yaw0))

    # Motion must come from Webots node state, not Python integration.
    motion_confirmed = dist > 1e-4 or (dyaw is not None and abs(dyaw) > 1e-4)

    result = {
        "probe": "phase1ar_physics_probe",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "webots_basic_time_step_ms": timestep,
        "n_steps": N_STEPS,
        "motor_command_rad_s": {
            "left": LEFT_VEL,
            "right": RIGHT_VEL,
            "motors": list(MOTOR_NAMES),
        },
        "pose_source": "Supervisor.getSelf().getPosition() + rotation field",
        "python_pose_integration": False,
        "initial_position_m": pos0,
        "initial_rotation_axis_angle": rot0,
        "initial_yaw_rad": yaw0,
        "final_position_m": pos1,
        "final_rotation_axis_angle": rot1,
        "final_yaw_rad": yaw1,
        "delta_position_m": [dx, dy, dz],
        "distance_m": dist,
        "delta_yaw_rad": dyaw,
        "gyro_present": gyro is not None,
        "accelerometer_present": accel is not None,
        "live_webots_physics_confirmed": bool(motion_confirmed),
        "robot_name": (
            self_node.getDef()
            or (
                self_node.getField("name").getSFString()
                if self_node.getField("name") is not None
                else "unknown"
            )
        ),
    }

    out_dir = os.path.join(
        _repo_root(), "results", "adaptation_locus_phase1ar_live_webots", "runtime_probe"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "physics_provenance.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    print(json.dumps(result, indent=2))
    print(f"WROTE {out_path}")
    print(
        "LIVE_WEBOTS_PHYSICS_CONFIRMED:",
        "YES" if motion_confirmed else "NO",
    )

    # Quit the simulation so batch mode exits.
    robot.simulationQuit(0 if motion_confirmed else 1)
    return 0 if motion_confirmed else 1


if __name__ == "__main__":
    raise SystemExit(main())
