"""Phase-1A-R Step 2 validation controller (runs inside Webots).

Nominal closed-loop, repeated reset, residual=0, mismatch=0, zero-intervention
equivalence vs historical VelocityDriveMode+kinematics cruise path.
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _out_dir() -> str:
    path = os.path.join(
        _repo_root(), "results", "adaptation_locus_phase1ar", "adapter_validation"
    )
    os.makedirs(path, exist_ok=True)
    return path


def _ensure_path() -> None:
    root = _repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    os.chdir(root)


def run_historical_cruise(backend, n_steps: int, cruise_v: float) -> dict:
    """Historical stack mode: VelocityDriveMode residual=0, steering=0, cruise throttle."""
    from parts.differential_drive import VelocityDriveMode

    drive = VelocityDriveMode(0.20, 1.50)
    throttle = cruise_v / 0.20
    traces = []
    backend.reset_physics_state()
    t0 = backend.robot.getTime()
    pos0 = list(backend.self_node.getPosition())
    for _ in range(n_steps):
        v, omega = drive.run(
            mode="local",
            user_steering=0.0,
            user_throttle=0.0,
            pilot_steering=0.0,
            pilot_throttle=throttle,
            residual_omega=0.0,
        )
        cmd_l, cmd_r = backend.kinematics.run(v, omega)
        backend.apply_wheel_speeds(cmd_l, cmd_r)
        if not backend.step_physics():
            break
        sens = backend.read_sensors()
        traces.append(
            {
                "sim_time_s": sens.sim_time_s,
                "position_m": sens.true_position_m,
                "yaw_rad": sens.true_yaw_rad,
                "cmd_left_rad_s": cmd_l,
                "cmd_right_rad_s": cmd_r,
                "omega_rad_s": omega,
                "v_m_s": v,
            }
        )
    pos1 = traces[-1]["position_m"] if traces else pos0
    return {
        "mode": "historical_velocity_drive_cruise",
        "n_steps": len(traces),
        "sim_duration_s": (traces[-1]["sim_time_s"] - traces[0]["sim_time_s"])
        if traces
        else 0.0,
        "start_position_m": pos0,
        "end_position_m": pos1,
        "delta_position_m": [pos1[i] - pos0[i] for i in range(3)],
        "mean_abs_yaw_rad": float(
            sum(abs(t["yaw_rad"]) for t in traces) / max(len(traces), 1)
        ),
        "mean_cmd_left": float(
            sum(t["cmd_left_rad_s"] for t in traces) / max(len(traces), 1)
        ),
        "mean_cmd_right": float(
            sum(t["cmd_right_rad_s"] for t in traces) / max(len(traces), 1)
        ),
        "trace": traces,
        "webots_time_start": t0,
    }


def main() -> int:
    _ensure_path()
    from research.adaptation_locus.live_webots.env import LiveWebotsEnv
    from research.adaptation_locus.live_webots.plant_backend import LiveWebotsBackend

    out = _out_dir()
    backend = LiveWebotsBackend()
    env = LiveWebotsEnv(backend=backend, max_steps=80, cruise_linear_velocity=0.12)

    results = {
        "probe": "phase1ar_live_adapter_validation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "plant": backend.provenance(),
        "adaptation_enabled": False,
        "residual_default": 0.0,
        "mismatch_nominal": True,
    }

    # --- Nominal closed-loop + multi-reset ---
    reset_poses = []
    nominal_logs = []
    for ep, seed in enumerate([0, 1, 2]):
        ctrl = env.reset(seed=seed, condition="nominal_A0_none")
        env.assert_controller_obs_firewall(ctrl)
        # Capture pose after reset
        sens = backend.read_sensors()
        reset_poses.append(
            {
                "episode": env.episode,
                "seed": seed,
                "position_m": sens.true_position_m,
                "yaw_rad": sens.true_yaw_rad,
                "estimator_params": env.estimator.get_params(),
                "residual_omega": env.residual.residual_omega_rad_s,
            }
        )
        done = False
        while not done:
            ctrl, info, done = env.step(0.0)
            env.assert_controller_obs_firewall(ctrl)
            # Residual must stay zero
            if abs(info["residual_omega_rad_s"]) > 1e-12:
                raise RuntimeError("residual non-zero under adaptation OFF")
            # Estimator params frozen
            if info["estimator_params"] != env.estimator.get_params():
                raise RuntimeError("estimator params mutated unexpectedly")
        metrics = env.get_metrics()
        log = env.get_log()
        nominal_logs.append({"seed": seed, "metrics": metrics, "n_log": len(log)})
        with open(os.path.join(out, f"nominal_seed{seed}.json"), "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "log": log}, f, indent=2)
            f.write("\n")

    results["repeated_reset"] = {
        "poses": reset_poses,
        "spawn_spread_m": max(
            math.dist(reset_poses[0]["position_m"], p["position_m"]) for p in reset_poses
        ),
        "pass": all(
            math.dist(reset_poses[0]["position_m"][:2], p["position_m"][:2]) < 0.05
            and abs(p["yaw_rad"] - reset_poses[0]["yaw_rad"]) < 0.1
            for p in reset_poses
        ),
    }
    results["nominal_closed_loop"] = {
        "episodes": nominal_logs,
        "pass": all(
            ep["metrics"]["mean_residual_abs"] < 1e-12
            and ep["metrics"]["mean_abs_true_yaw_rad"] < 0.5
            for ep in nominal_logs
        ),
    }

    # --- Zero mismatch hook identity ---
    env.mismatch.imu_bias_rad_s = 0.0
    env.mismatch.motor_delta = 0.0
    env.reset(seed=0, condition="mismatch_hooks_zero")
    for _ in range(40):
        _, info, done = env.step(0.0)
        if done:
            break
    zero_hook_metrics = env.get_metrics()
    results["zero_mismatch_hooks"] = {
        "metrics": zero_hook_metrics,
        "pass": zero_hook_metrics["mean_residual_abs"] < 1e-12,
    }

    # --- Zero-intervention equivalence ---
    hist = run_historical_cruise(backend, n_steps=80, cruise_v=0.12)
    with open(os.path.join(out, "historical_cruise.json"), "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)
        f.write("\n")

    env.reset(seed=0, condition="research_zero_intervention")
    # Force heading estimate ~0 and residual 0 → should behave like cruise
    env.estimator.unlock()
    env.estimator.reset(0.0)
    env.estimator.set_params(0.0, 0.85)
    env.estimator.lock()
    research_trace = []
    for _ in range(80):
        ctrl, info, done = env.step(0.0)
        research_trace.append(
            {
                "position_m": info["true_position_m"],
                "yaw_rad": info["true_yaw_rad"],
                "base_omega": ctrl.heading_est_rad,  # placeholder filled below
            }
        )
        if done:
            break
    # Re-run capturing commands from log
    research_log = env.get_log()
    with open(os.path.join(out, "research_zero_intervention.json"), "w", encoding="utf-8") as f:
        json.dump({"log": research_log, "metrics": env.get_metrics()}, f, indent=2)
        f.write("\n")

    # Compare mean |wheel cmds| and displacement scale
    hist_dist = math.dist(hist["start_position_m"], hist["end_position_m"])
    res_start = research_log[0]["true_position_m"]
    res_end = research_log[-1]["true_position_m"]
    res_dist = math.dist(res_start, res_end)
    mean_hist_cmd = 0.5 * (hist["mean_cmd_left"] + hist["mean_cmd_right"])
    mean_res_cmd = 0.5 * (
        sum(r["cmd_left_rad_s"] for r in research_log) / len(research_log)
        + sum(r["cmd_right_rad_s"] for r in research_log) / len(research_log)
    )
    # Equivalence: distances within factor 3 and same order; yaw small both
    equiv_pass = (
        hist_dist > 0.02
        and res_dist > 0.02
        and max(hist_dist, res_dist) / max(min(hist_dist, res_dist), 1e-6) < 3.0
        and hist["mean_abs_yaw_rad"] < 0.5
        and env.get_metrics()["mean_abs_true_yaw_rad"] < 0.5
    )
    results["zero_intervention_equivalence"] = {
        "historical_distance_m": hist_dist,
        "research_distance_m": res_dist,
        "historical_mean_abs_yaw_rad": hist["mean_abs_yaw_rad"],
        "research_mean_abs_yaw_rad": env.get_metrics()["mean_abs_true_yaw_rad"],
        "historical_mean_wheel_cmd": mean_hist_cmd,
        "research_mean_wheel_cmd": mean_res_cmd,
        "pass": bool(equiv_pass),
        "note": (
            "Historical mode: VelocityDriveMode cruise, residual=0, steering=0. "
            "Research mode: heading-P + residual=0 + mismatch=0 on same live ODE."
        ),
    }

    # Estimator params remain fixed across episode
    results["estimator_frozen"] = {
        "params": env.estimator.get_params(),
        "adaptation_enabled": env.estimator.adaptation_enabled,
        "pass": env.estimator.adaptation_enabled is False,
    }

    gates = {
        "live_ode_plant": results["plant"]["plant"] == "live_webots_ode",
        "no_python_pose_integration": results["plant"]["python_pose_integration"] is False,
        "nominal_closed_loop": results["nominal_closed_loop"]["pass"],
        "repeated_reset": results["repeated_reset"]["pass"],
        "zero_mismatch_hooks": results["zero_mismatch_hooks"]["pass"],
        "zero_intervention_equivalence": results["zero_intervention_equivalence"]["pass"],
        "estimator_frozen": results["estimator_frozen"]["pass"],
        "residual_zero": all(
            ep["metrics"]["mean_residual_abs"] < 1e-12 for ep in nominal_logs
        ),
    }
    results["gates"] = gates
    results["pass"] = all(gates.values())

    summary_path = os.path.join(out, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(json.dumps(gates, indent=2))
    print("PASS" if results["pass"] else "FAIL")
    print("WROTE", summary_path)

    env.close()
    backend.quit(0 if results["pass"] else 1)
    return 0 if results["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
