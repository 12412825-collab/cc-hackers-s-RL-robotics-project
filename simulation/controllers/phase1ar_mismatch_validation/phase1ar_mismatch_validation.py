"""Phase-1A-R Step 3 — live mismatch causal separation validation."""

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
        _repo_root(), "results", "adaptation_locus_phase1ar", "mismatch_validation"
    )
    os.makedirs(path, exist_ok=True)
    return path


def _ensure_path() -> None:
    root = _repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    os.chdir(root)


def _run_episode(env, seed: int, name: str, cfg, n_steps: int = 60, open_loop_omega=None):
    env.open_loop_omega = open_loop_omega
    ctrl = env.reset(seed=seed, condition=name, mismatch=cfg)
    env.assert_controller_obs_firewall(ctrl)
    first_info = None
    done = False
    while not done and env._step_count < n_steps:
        ctrl, info, done = env.step(0.0)
        env.assert_controller_obs_firewall(ctrl)
        if first_info is None:
            first_info = info
        # Adaptation guards
        assert abs(info["residual_omega_rad_s"]) < 1e-15
        assert info["estimator_params"] == env._phi0
    metrics = env.get_metrics()
    log = env.get_log()
    return {
        "name": name,
        "seed": seed,
        "mismatch": cfg.to_dict(),
        "first_intervention": first_info["intervention"] if first_info else None,
        "metrics": metrics,
        "log": log,
        "clip_fraction": metrics.get("clip_fraction", 0.0),
        "final_true_yaw_rad": metrics.get("final_true_yaw_rad"),
        "mean_abs_true_yaw_rad": metrics.get("mean_abs_true_yaw_rad"),
    }


def main() -> int:
    _ensure_path()
    from research.adaptation_locus.live_webots.env import LiveWebotsEnv
    from research.adaptation_locus.live_webots.mismatch import (
        DIAG_IMU_BIAS_RAD_S,
        DIAG_MOTOR_DELTA,
        diagnostic_suite,
        make_mismatch,
    )
    from research.adaptation_locus.live_webots.plant_backend import LiveWebotsBackend

    out = _out_dir()
    backend = LiveWebotsBackend()
    env = LiveWebotsEnv(backend=backend, max_steps=60, cruise_linear_velocity=0.12)

    seeds = [0, 1, 2]
    summary = {
        "probe": "phase1ar_mismatch_causal_validation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_imu_bias_rad_s": DIAG_IMU_BIAS_RAD_S,
        "diagnostic_motor_delta": DIAG_MOTOR_DELTA,
        "adaptation_off": True,
        "residual_zero": True,
        "note_units": (
            "M1 bias is on Gyro yaw-rate [rad/s] (historical stack has Gyro, "
            "not InertialUnit heading). Internals are radians / rad/s only."
        ),
        "plant": backend.provenance(),
        "episodes": {},
        "paired_checks": {},
        "gates": {},
    }

    # ---- D0–D4 closed-loop diagnostics ----
    for seed in seeds:
        suite = diagnostic_suite(seed=seed)
        for name, cfg in suite.items():
            key = f"{name}_seed{seed}"
            ep = _run_episode(env, seed, name, cfg, n_steps=60)
            summary["episodes"][key] = {
                k: v for k, v in ep.items() if k != "log"
            }
            with open(os.path.join(out, f"{key}.json"), "w", encoding="utf-8") as f:
                json.dump(ep, f, indent=2)
                f.write("\n")

    # ---- Paired causal checks (seed 0) ----
    # M1: open-loop omega=0 so first-step plant command identical; only obs differs.
    nom = make_mismatch("none", seed=0)
    imu_p = make_mismatch(
        "imu_bias", "diagnostic_pos", imu_bias_rad_s=+DIAG_IMU_BIAS_RAD_S, seed=0
    )
    imu_n = make_mismatch(
        "imu_bias", "diagnostic_neg", imu_bias_rad_s=-DIAG_IMU_BIAS_RAD_S, seed=0
    )

    ep0 = _run_episode(env, 0, "pair_imu_nominal", nom, n_steps=40, open_loop_omega=0.0)
    ep1 = _run_episode(env, 0, "pair_imu_pos", imu_p, n_steps=40, open_loop_omega=0.0)
    ep2 = _run_episode(env, 0, "pair_imu_neg", imu_n, n_steps=40, open_loop_omega=0.0)

    i0 = ep0["first_intervention"]
    i1 = ep1["first_intervention"]
    i2 = ep2["first_intervention"]

    m1_checks = {
        # Cross-run raw gyro may differ slightly after independent ODE resets;
        # causal claim is WITHIN-STEP: observed = raw + bias, gains untouched.
        "within_step_bias_pos_exact": abs(
            (i1["observed_imu_yaw_rate_rad_s"] - i1["raw_imu_yaw_rate_rad_s"])
            - DIAG_IMU_BIAS_RAD_S
        )
        < 1e-12,
        "within_step_bias_neg_exact": abs(
            (i2["observed_imu_yaw_rate_rad_s"] - i2["raw_imu_yaw_rate_rad_s"])
            + DIAG_IMU_BIAS_RAD_S
        )
        < 1e-12,
        "observed_minus_raw_pos": i1["observed_imu_yaw_rate_rad_s"]
        - i1["raw_imu_yaw_rate_rad_s"],
        "observed_minus_raw_neg": i2["observed_imu_yaw_rate_rad_s"]
        - i2["raw_imu_yaw_rate_rad_s"],
        "motor_gains_unchanged_under_imu": (
            abs(i1["motor_gain_left"] - 1.0) < 1e-15
            and abs(i1["motor_gain_right"] - 1.0) < 1e-15
            and abs(i2["motor_gain_left"] - 1.0) < 1e-15
        ),
        "requested_cmds_identical_open_loop_first_step": (
            abs(i1["requested_left_rad_s"] - i0["requested_left_rad_s"]) < 1e-9
            and abs(i1["requested_right_rad_s"] - i0["requested_right_rad_s"]) < 1e-9
        ),
        "spawn_pose_restored": (
            math.dist(i0["pre_physics_true_position_m"], [-3.0, i0["pre_physics_true_position_m"][1], 0.0])
            < 0.05
            and math.dist(i1["pre_physics_true_position_m"], [-3.0, i1["pre_physics_true_position_m"][1], 0.0])
            < 0.05
            and abs(i0["pre_physics_true_yaw_rad"]) < 0.05
            and abs(i1["pre_physics_true_yaw_rad"]) < 0.05
        ),
        "cross_run_raw_imu_delta": abs(
            i1["raw_imu_yaw_rate_rad_s"] - i0["raw_imu_yaw_rate_rad_s"]
        ),
        "phi_frozen": i1["estimator_params"] == i0["estimator_params"],
        "distinction": (
            "Direct intervention = observed IMU only (observed=raw+bias). "
            "Cross-run raw gyro micro-differences after independent resets are "
            "not M1 contamination. Later closed-loop trajectory changes are "
            "downstream controller reactions."
        ),
    }
    m1_checks["pass"] = all(
        [
            m1_checks["within_step_bias_pos_exact"],
            m1_checks["within_step_bias_neg_exact"],
            m1_checks["motor_gains_unchanged_under_imu"],
            m1_checks["requested_cmds_identical_open_loop_first_step"],
            m1_checks["spawn_pose_restored"],
            m1_checks["phi_frozen"],
        ]
    )
    summary["paired_checks"]["M1_imu"] = m1_checks
    with open(os.path.join(out, "pair_m1_seed0.json"), "w", encoding="utf-8") as f:
        json.dump({"nominal": ep0, "pos": ep1, "neg": ep2, "checks": m1_checks}, f)
        f.write("\n")

    # M2: open-loop so requested commands identical; only applied gains differ.
    mot_p = make_mismatch(
        "motor_asymmetry", "diagnostic_pos", motor_delta=+DIAG_MOTOR_DELTA, seed=0
    )
    mot_r = make_mismatch(
        "motor_asymmetry", "diagnostic_rev", motor_delta=-DIAG_MOTOR_DELTA, seed=0
    )
    mp0 = _run_episode(env, 0, "pair_motor_nominal", nom, n_steps=60, open_loop_omega=0.0)
    mp1 = _run_episode(env, 0, "pair_motor_pos", mot_p, n_steps=60, open_loop_omega=0.0)
    mp2 = _run_episode(env, 0, "pair_motor_rev", mot_r, n_steps=60, open_loop_omega=0.0)
    j0, j1, j2 = mp0["first_intervention"], mp1["first_intervention"], mp2["first_intervention"]

    def gain_ok(inter, delta):
        return (
            abs(inter["motor_gain_left"] - (1.0 + delta)) < 1e-12
            and abs(inter["motor_gain_right"] - (1.0 - delta)) < 1e-12
            and abs(
                inter["applied_left_rad_s"]
                - inter["motor_gain_left"] * inter["requested_left_rad_s"]
            )
            < 1e-9
            and abs(
                inter["applied_right_rad_s"]
                - inter["motor_gain_right"] * inter["requested_right_rad_s"]
            )
            < 1e-9
        )

    m2_checks = {
        # Within-step: M2 must not add IMU bias; raw vs observed identical in-step.
        "within_step_no_imu_bias": abs(j1.get("gyro_rate_bias_rad_s", j1.get("mismatch_imu_bias_rad_s", 0.0))) < 1e-15
        and abs(j1["observed_imu_yaw_rate_rad_s"] - j1["raw_imu_yaw_rate_rad_s"]) < 1e-15,
        "imu_bias_zero_under_motor": abs(
            j1.get("gyro_rate_bias_rad_s", j1.get("mismatch_imu_bias_rad_s", 0.0))
        )
        < 1e-15,
        "requested_identical_first_step": (
            abs(j1["requested_left_rad_s"] - j0["requested_left_rad_s"]) < 1e-9
            and abs(j1["requested_right_rad_s"] - j0["requested_right_rad_s"]) < 1e-9
        ),
        "gains_pos_ok": gain_ok(j1, +DIAG_MOTOR_DELTA),
        "gains_rev_ok": gain_ok(j2, -DIAG_MOTOR_DELTA),
        "yaw_drift_sign_reverses": (
            math.copysign(1.0, mp1["final_true_yaw_rad"] + 1e-18)
            != math.copysign(1.0, mp2["final_true_yaw_rad"] + 1e-18)
            or (
                abs(mp1["final_true_yaw_rad"]) < 1e-5
                and abs(mp2["final_true_yaw_rad"]) < 1e-5
            )
        ),
        "final_yaw_pos": mp1["final_true_yaw_rad"],
        "final_yaw_rev": mp2["final_true_yaw_rad"],
        "clip_fraction_pos": mp1["clip_fraction"],
        "clip_fraction_rev": mp2["clip_fraction"],
        "phi_frozen": j1["estimator_params"] == j0["estimator_params"],
        "cross_run_raw_imu_delta": abs(
            j1["raw_imu_yaw_rate_rad_s"] - j0["raw_imu_yaw_rate_rad_s"]
        ),
        "distinction": (
            "Direct intervention = applied motor gains only. Later IMU changes "
            "are downstream of changed ODE motion, not sensor-side contamination."
        ),
    }
    # If requested is nonzero, applied must differ when delta nonzero
    if abs(j1["requested_left_rad_s"]) > 1e-6 or abs(j1["requested_right_rad_s"]) > 1e-6:
        m2_checks["applied_differs_from_requested_when_delta"] = (
            abs(j1["applied_left_rad_s"] - j1["requested_left_rad_s"]) > 1e-9
            or abs(j1["applied_right_rad_s"] - j1["requested_right_rad_s"]) > 1e-9
        )
    else:
        m2_checks["applied_differs_from_requested_when_delta"] = (
            abs(j1["applied_left_rad_s"] - j0["applied_left_rad_s"]) > 1e-9
            or abs(j1["applied_right_rad_s"] - j0["applied_right_rad_s"]) > 1e-9
        )

    # Stronger yaw reverse check using mean yaw rate proxy over episode
    yaw1 = mp1["final_true_yaw_rad"]
    yaw2 = mp2["final_true_yaw_rad"]
    m2_checks["yaw_product_negative_or_tiny"] = (yaw1 * yaw2) < 0.0 or (
        abs(yaw1) < 1e-4 and abs(yaw2) < 1e-4
    )
    m2_checks["pass"] = all(
        [
            m2_checks["within_step_no_imu_bias"],
            m2_checks["imu_bias_zero_under_motor"],
            m2_checks["requested_identical_first_step"],
            m2_checks["gains_pos_ok"],
            m2_checks["gains_rev_ok"],
            m2_checks["applied_differs_from_requested_when_delta"],
            m2_checks["phi_frozen"],
            m2_checks["yaw_product_negative_or_tiny"],
            mp1["clip_fraction"] < 0.05,
            mp2["clip_fraction"] < 0.05,
        ]
    )
    summary["paired_checks"]["M2_motor"] = m2_checks
    with open(os.path.join(out, "pair_m2_seed0.json"), "w", encoding="utf-8") as f:
        json.dump({"nominal": mp0, "pos": mp1, "rev": mp2, "checks": m2_checks}, f)
        f.write("\n")

    # Closed-loop IMU downstream divergence (controller reacts to bias)
    cl0 = _run_episode(env, 0, "cl_imu_nom", nom, n_steps=60, open_loop_omega=None)
    cl1 = _run_episode(env, 0, "cl_imu_pos", imu_p, n_steps=60, open_loop_omega=None)
    summary["paired_checks"]["M1_closed_loop_downstream"] = {
        "nominal_final_yaw": cl0["final_true_yaw_rad"],
        "biased_final_yaw": cl1["final_true_yaw_rad"],
        "diverged": abs(cl1["final_true_yaw_rad"] - cl0["final_true_yaw_rad"]) > 1e-3,
        "note": "Expected: physical yaw diverges because controller reacts to biased obs.",
    }

    # Zero mismatch reproduces nominal adapter behavior (clip 0, residual 0)
    z = summary["episodes"]["D0_nominal_seed0"]
    summary["gates"] = {
        "m1_causal": m1_checks["pass"],
        "m2_causal": m2_checks["pass"],
        "m1_closed_loop_downstream": summary["paired_checks"]["M1_closed_loop_downstream"][
            "diverged"
        ],
        "adaptation_off": True,
        "residual_zero": abs(z["metrics"]["mean_residual_abs"]) < 1e-12,
        "clip_negligible": m2_checks["clip_fraction_pos"] < 0.05,
        "live_ode": backend.provenance()["plant"] == "live_webots_ode",
    }
    summary["pass"] = all(summary["gates"].values())

    # Compact intervention table for docs
    summary["intervention_table"] = {
        "M1_first_step": {
            "within_step_observed_equals_raw_plus_bias": m1_checks[
                "within_step_bias_pos_exact"
            ],
            "motor_gains_same": m1_checks["motor_gains_unchanged_under_imu"],
            "requested_cmds_same_open_loop": m1_checks[
                "requested_cmds_identical_open_loop_first_step"
            ],
            "spawn_restored": m1_checks["spawn_pose_restored"],
        },
        "M2_first_step": {
            "within_step_no_imu_bias": m2_checks["within_step_no_imu_bias"],
            "requested_same": m2_checks["requested_identical_first_step"],
            "applied_changed": m2_checks["applied_differs_from_requested_when_delta"],
            "yaw_sign_reverses": m2_checks["yaw_product_negative_or_tiny"],
        },
    }

    path = os.path.join(out, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(json.dumps(summary["gates"], indent=2))
    print("PASS" if summary["pass"] else "FAIL")
    print("WROTE", path)

    # Simple text plot/table artifact
    table_path = os.path.join(out, "paired_intervention_table.txt")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("M1 first-step (open-loop)\n")
        f.write(json.dumps(summary["intervention_table"]["M1_first_step"], indent=2))
        f.write("\n\nM2 first-step (open-loop)\n")
        f.write(json.dumps(summary["intervention_table"]["M2_first_step"], indent=2))
        f.write("\n\nM2 yaw pos/rev: ")
        f.write(f"{yaw1:.6f} / {yaw2:.6f}\n")
        f.write(f"clip_fraction pos/rev: {mp1['clip_fraction']:.4f} / {mp2['clip_fraction']:.4f}\n")
    env.close()
    backend.quit(0 if summary["pass"] else 1)
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
