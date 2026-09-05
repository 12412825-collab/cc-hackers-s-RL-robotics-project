"""Phase-1A-R Step 1.5 — live Webots physics sanity diagnostic.

Runs Gate A–D on the historical FourWheelRobot plant.
Logs Webots simulation time, Supervisor pose, and wheel PositionSensor rates.
Does NOT integrate pose in Python. Does NOT use Phase-0 / W-1 plants.
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
ENCODER_NAMES = (
    "front left wheel sensor",
    "front right wheel sensor",
    "rear left wheel sensor",
    "rear right wheel sensor",
)
LEFT_MOTORS = ("front left wheel motor", "rear left wheel motor")
RIGHT_MOTORS = ("front right wheel motor", "rear right wheel motor")
LEFT_ENCODERS = ("front left wheel sensor", "rear left wheel sensor")
RIGHT_ENCODERS = ("front right wheel sensor", "rear right wheel sensor")

WHEEL_RADIUS_M = 0.0325  # from historical PROTO


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _out_dir() -> str:
    path = os.path.join(
        _repo_root(), "results", "adaptation_locus_phase1ar", "physics_sanity"
    )
    os.makedirs(path, exist_ok=True)
    return path


def _yaw_from_rotation(axis_angle) -> float:
    ax, ay, az, angle = [float(v) for v in axis_angle]
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    r00 = t * ax * ax + c
    r02 = t * ax * az - s * ay
    return math.atan2(r02, r00)


def _snapshot(robot, self_node, rot_field, encoders, prev_enc, dt):
    pos = list(self_node.getPosition())
    rot = list(rot_field.getSFRotation()) if rot_field is not None else None
    yaw = _yaw_from_rotation(rot) if rot is not None else None
    enc = {name: encoders[name].getValue() for name in encoders}
    rates = {}
    if prev_enc is not None and dt > 0:
        for name in encoders:
            rates[name] = (enc[name] - prev_enc[name]) / dt
    return {
        "sim_time_s": float(robot.getTime()),
        "position_m": pos,
        "rotation_axis_angle": rot,
        "yaw_rad": yaw,
        "encoder_rad": enc,
        "encoder_rate_rad_s": rates,
    }, enc


def _set_velocities(motors, left_vel: float, right_vel: float) -> None:
    for name in LEFT_MOTORS:
        motors[name].setVelocity(left_vel)
    for name in RIGHT_MOTORS:
        motors[name].setVelocity(right_vel)


def _run_segment(robot, self_node, rot_field, motors, encoders, timestep, left, right, n_steps, label):
    dt = timestep / 1000.0
    _set_velocities(motors, left, right)
    # one settle read after command apply
    if robot.step(timestep) == -1:
        raise RuntimeError(f"sim terminated at start of {label}")
    prev = None
    snap0, prev = _snapshot(robot, self_node, rot_field, encoders, None, dt)
    traces = [snap0]
    for i in range(n_steps):
        if robot.step(timestep) == -1:
            raise RuntimeError(f"sim terminated during {label} step {i}")
        snap, prev = _snapshot(robot, self_node, rot_field, encoders, prev, dt)
        traces.append(snap)

    first, last = traces[0], traces[-1]
    dx = last["position_m"][0] - first["position_m"][0]
    dy = last["position_m"][1] - first["position_m"][1]
    dz = last["position_m"][2] - first["position_m"][2]
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    dyaw = None
    if first["yaw_rad"] is not None and last["yaw_rad"] is not None:
        dyaw = math.atan2(
            math.sin(last["yaw_rad"] - first["yaw_rad"]),
            math.cos(last["yaw_rad"] - first["yaw_rad"]),
        )
    sim_dt = last["sim_time_s"] - first["sim_time_s"]
    # expected scale from mean |ω| * r * t (no-slip upper-ish scale)
    mean_cmd = 0.5 * (abs(left) + abs(right))
    expected_dist = WHEEL_RADIUS_M * mean_cmd * max(sim_dt, 0.0)

    # mean observed wheel rates over last half of segment
    half = traces[len(traces) // 2 :]
    obs_rates = {name: [] for name in encoders}
    for snap in half:
        for name, rate in snap["encoder_rate_rad_s"].items():
            obs_rates[name].append(rate)
    mean_obs = {
        name: (sum(vals) / len(vals) if vals else None) for name, vals in obs_rates.items()
    }

    return {
        "label": label,
        "command_left_rad_s": left,
        "command_right_rad_s": right,
        "n_controller_steps_after_prime": n_steps,
        "timestep_ms": timestep,
        "sim_time_start_s": first["sim_time_s"],
        "sim_time_end_s": last["sim_time_s"],
        "sim_duration_s": sim_dt,
        "initial_position_m": first["position_m"],
        "final_position_m": last["position_m"],
        "delta_position_m": [dx, dy, dz],
        "distance_m": dist,
        "initial_yaw_rad": first["yaw_rad"],
        "final_yaw_rad": last["yaw_rad"],
        "delta_yaw_rad": dyaw,
        "expected_no_slip_distance_m": expected_dist,
        "distance_over_expected": (dist / expected_dist) if expected_dist > 1e-9 else None,
        "mean_observed_wheel_rate_rad_s": mean_obs,
        "trace": traces,
    }


def main() -> int:
    from controller import Supervisor

    # Optional suite selection via env for isolated tests.
    suite = os.environ.get("PHASE1AR_SANITY_SUITE", "all").strip().lower()

    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    self_node = robot.getSelf()
    if self_node is None:
        print("FATAL: getSelf() is None", file=sys.stderr)
        return 2

    motors = {}
    for name in MOTOR_NAMES:
        m = robot.getDevice(name)
        if m is None:
            print(f"FATAL: missing motor {name}", file=sys.stderr)
            return 3
        m.setPosition(float("inf"))
        m.setVelocity(0.0)
        motors[name] = m

    encoders = {}
    for name in ENCODER_NAMES:
        s = robot.getDevice(name)
        if s is None:
            print(f"FATAL: missing encoder {name}", file=sys.stderr)
            return 4
        s.enable(timestep)
        encoders[name] = s

    rot_field = self_node.getField("rotation")

    # Prime sensors / settle contacts.
    if robot.step(timestep) == -1:
        return 5

    world_meta = {
        "probe": "phase1ar_physics_sanity",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "suite": suite,
        "coordinate_note": (
            "Webots default Y-up: vertical=Y; floor plane=X-Z; "
            "gravity acts along -Y."
        ),
        "wheel_radius_m": WHEEL_RADIUS_M,
        "basic_time_step_ms": timestep,
        "python_pose_integration": False,
        "pose_source": "Supervisor.getSelf().getPosition() + rotation field",
        "robot_name": self_node.getField("name").getSFString()
        if self_node.getField("name") is not None
        else "unknown",
    }

    segments = []
    # 4 s each at 50 ms -> 80 steps (matches original probe duration scale)
    n = 80
    plans = []
    if suite in ("all", "a", "zero"):
        plans.append((0.0, 0.0, n, "A_zero_command"))
    if suite in ("all", "b", "symmetric"):
        plans.append((1.0, 1.0, n, "B_symmetric_slow"))
    if suite in ("all", "c", "differential"):
        plans.append((1.5, 0.5, n, "C_differential"))
    if suite in ("all", "d", "reverse"):
        plans.append((-1.0, -1.0, n, "D_reverse"))
    if suite in ("all", "legacy"):
        plans.append((4.0, 2.0, n, "LEGACY_probe_4_2"))

    for left, right, steps, label in plans:
        # reset pose to initial translation if possible between segments
        trans_field = self_node.getField("translation")
        rot0 = self_node.getField("rotation")
        if trans_field is not None:
            trans_field.setSFVec3f([-3.0, 0.0325, 0.0])
        if rot0 is not None:
            rot0.setSFRotation([0.0, 1.0, 0.0, 0.0])
        _set_velocities(motors, 0.0, 0.0)
        # reset physics velocities
        if hasattr(self_node, "resetPhysics"):
            self_node.resetPhysics()
        # settle a few steps after reset
        for _ in range(10):
            if robot.step(timestep) == -1:
                return 6
        seg = _run_segment(
            robot, self_node, rot_field, motors, encoders, timestep, left, right, steps, label
        )
        segments.append(seg)
        # write per-segment trace immediately
        seg_path = os.path.join(_out_dir(), f"{label}.json")
        with open(seg_path, "w", encoding="utf-8") as f:
            json.dump(seg, f, indent=2)
            f.write("\n")
        print(
            f"{label}: dist={seg['distance_m']:.4f} m "
            f"expected~{seg['expected_no_slip_distance_m']:.4f} m "
            f"ratio={seg['distance_over_expected']} "
            f"dy={seg['delta_position_m'][1]:.4f} "
            f"sim_dt={seg['sim_duration_s']:.3f}s"
        )

    summary = {
        **world_meta,
        "segments": [
            {k: v for k, v in seg.items() if k != "trace"} for seg in segments
        ],
    }

    # Gate evaluation (pre-repair baseline uses same thresholds as acceptance)
    def gate_a():
        seg = next((s for s in segments if s["label"].startswith("A_")), None)
        if seg is None:
            return None
        dy = abs(seg["delta_position_m"][1])
        dist = seg["distance_m"]
        return {
            "pass": dist < 0.05 and dy < 0.05,
            "distance_m": dist,
            "delta_y_m": seg["delta_position_m"][1],
        }

    def gate_b():
        seg = next((s for s in segments if s["label"].startswith("B_")), None)
        if seg is None:
            return None
        ratio = seg["distance_over_expected"]
        ok = ratio is not None and 0.2 <= ratio <= 5.0 and abs(seg["delta_position_m"][1]) < 0.1
        return {"pass": bool(ok), "ratio": ratio, "distance_m": seg["distance_m"]}

    def gate_c():
        seg = next((s for s in segments if s["label"].startswith("C_")), None)
        if seg is None:
            return None
        dyaw = seg["delta_yaw_rad"]
        # left > right with +Z left ⇒ expect yaw sign consistent; require nonzero yaw
        ok = dyaw is not None and abs(dyaw) > 0.02 and abs(seg["delta_position_m"][1]) < 0.15
        # magnitude: not explosive
        ok = ok and seg["distance_m"] < 5.0 * max(seg["expected_no_slip_distance_m"], 1e-6)
        return {"pass": bool(ok), "delta_yaw_rad": dyaw, "distance_m": seg["distance_m"]}

    def gate_d_stability():
        bad = []
        for seg in segments:
            if abs(seg["delta_position_m"][1]) > 0.2:
                bad.append(f"{seg['label']}: vertical jump {seg['delta_position_m'][1]:.3f}m")
            if seg["distance_over_expected"] is not None and seg["distance_over_expected"] > 20:
                bad.append(
                    f"{seg['label']}: distance ratio {seg['distance_over_expected']:.1f}x"
                )
        return {"pass": len(bad) == 0, "issues": bad}

    def gate_e_clock():
        oks = []
        for seg in segments:
            expected = (seg["n_controller_steps_after_prime"]) * (timestep / 1000.0)
            # first sample already after one step in _run_segment setup; duration spans n steps
            # sim_duration should be ~ n * dt
            err = abs(seg["sim_duration_s"] - expected)
            oks.append(err < 0.05 + 1e-6)
        return {
            "pass": all(oks) if oks else None,
            "per_segment_sim_duration_s": [s["sim_duration_s"] for s in segments],
            "expected_per_segment_s": n * (timestep / 1000.0),
        }

    summary["gates"] = {
        "A_zero": gate_a(),
        "B_symmetric_scale": gate_b(),
        "C_differential": gate_c(),
        "D_no_explosion": gate_d_stability(),
        "E_sim_clock": gate_e_clock(),
    }

    out = os.path.join(_out_dir(), "summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(json.dumps(summary["gates"], indent=2))
    print("WROTE", out)

    robot.simulationQuit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
