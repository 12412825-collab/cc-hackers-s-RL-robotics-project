"""Phase-1A-R Step 3.5 — fixed heading bias semantics validation."""

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
        _repo_root(), "results", "adaptation_locus_phase1ar", "sensor_semantics"
    )
    os.makedirs(path, exist_ok=True)
    return path


def _ensure_path() -> None:
    root = _repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    os.chdir(root)


def _run(env, seed, name, cfg, n_steps=80, open_loop_omega=0.0):
    from research.adaptation_locus.live_webots.mismatch import angle_diff_rad

    env.open_loop_omega = open_loop_omega
    ctrl = env.reset(seed=seed, condition=name, mismatch=cfg)
    env.assert_controller_obs_firewall(ctrl)
    assert ctrl.heading_source == "gyro_integration"
    first = None
    done = False
    while not done and env._step_count < n_steps:
        ctrl, info, done = env.step(0.0)
        env.assert_controller_obs_firewall(ctrl)
        if first is None:
            first = info
        assert abs(info["residual_omega_rad_s"]) < 1e-15
        assert info["estimator_params"] == env._phi0
    log = env.get_log()
    offsets = [
        angle_diff_rad(r["observed_heading_rad"], r["raw_heading_rad"]) for r in log
    ]
    return {
        "name": name,
        "seed": seed,
        "mismatch": cfg.to_dict(),
        "first_intervention": first["intervention"] if first else None,
        "metrics": env.get_metrics(),
        "log": log,
        "heading_offsets_rad": offsets,
        "offset_mean": float(sum(offsets) / max(len(offsets), 1)),
        "offset_std": float(
            (sum((o - sum(offsets) / len(offsets)) ** 2 for o in offsets) / len(offsets))
            ** 0.5
        )
        if offsets
        else 0.0,
        "sim_duration_s": log[-1]["simulation_time"] - log[0]["simulation_time"]
        if len(log) > 1
        else 0.0,
    }


def main() -> int:
    _ensure_path()
    from research.adaptation_locus.live_webots.env import HEADING_SOURCE, LiveWebotsEnv
    from research.adaptation_locus.live_webots.mismatch import (
        DIAG_FIXED_HEADING_BIAS_RAD,
        DIAG_GYRO_RATE_BIAS_RAD_S,
        angle_diff_rad,
        make_mismatch,
    )
    from research.adaptation_locus.live_webots.plant_backend import LiveWebotsBackend

    out = _out_dir()
    backend = LiveWebotsBackend()
    env = LiveWebotsEnv(backend=backend, max_steps=80, cruise_linear_velocity=0.12)
    b = DIAG_FIXED_HEADING_BIAS_RAD

    summary = {
        "probe": "phase1ar_sensor_semantics_alignment",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "primary_heading_source": HEADING_SOURCE,
        "uses_supervisor_for_controller_heading": False,
        "primary_m1": "fixed_heading_bias",
        "primary_bias_unit": "rad",
        "secondary_m1": "gyro_rate_bias",
        "secondary_bias_unit": "rad/s",
        "diagnostic_fixed_heading_bias_rad": b,
        "diagnostic_gyro_rate_bias_rad_s": DIAG_GYRO_RATE_BIAS_RAD_S,
        "adaptation_off": True,
        "residual_zero": True,
        "plant": backend.provenance(),
        "gates": {},
    }

    nom = make_mismatch("none", seed=0)
    pos = make_mismatch(
        "fixed_heading_bias", "diagnostic_pos", fixed_heading_bias_rad=+b, seed=0
    )
    neg = make_mismatch(
        "fixed_heading_bias", "diagnostic_neg", fixed_heading_bias_rad=-b, seed=0
    )
    rate = make_mismatch(
        "gyro_rate_bias",
        "contrast",
        gyro_rate_bias_rad_s=+DIAG_GYRO_RATE_BIAS_RAD_S,
        seed=0,
    )

    # 4 s @ 50 ms = 80 steps
    ep0 = _run(env, 0, "zero_mismatch", nom, n_steps=80, open_loop_omega=0.0)
    ep1 = _run(env, 0, "fixed_heading_pos", pos, n_steps=80, open_loop_omega=0.0)
    ep2 = _run(env, 0, "fixed_heading_neg", neg, n_steps=80, open_loop_omega=0.0)
    ep3 = _run(env, 0, "gyro_rate_contrast", rate, n_steps=80, open_loop_omega=0.0)

    for ep, fname in [
        (ep0, "zero_mismatch.json"),
        (ep1, "fixed_heading_pos.json"),
        (ep2, "fixed_heading_neg.json"),
        (ep3, "gyro_rate_contrast.json"),
    ]:
        with open(os.path.join(out, fname), "w", encoding="utf-8") as f:
            json.dump(ep, f, indent=2)
            f.write("\n")

    # Constancy of fixed offset
    const_pos = all(abs(o - b) < 1e-9 for o in ep1["heading_offsets_rad"])
    const_neg = all(abs(o + b) < 1e-9 for o in ep2["heading_offsets_rad"])
    # Gyro-rate contrast: offset grows ~ bias * t
    rate_offsets = ep3["heading_offsets_rad"]
    t_end = ep3["sim_duration_s"]
    rate_grows = abs(rate_offsets[-1]) > abs(rate_offsets[10]) + 0.05
    expected_end = DIAG_GYRO_RATE_BIAS_RAD_S * max(t_end, 1e-6)
    rate_near_linear = abs(abs(rate_offsets[-1]) - expected_end) < 0.15  # allow ODE settle

    # Causal separation for fixed heading bias
    i0, i1, i2 = ep0["first_intervention"], ep1["first_intervention"], ep2["first_intervention"]
    causal = {
        "raw_heading_near_spawn": abs(i1["raw_heading_rad"]) < 0.05,
        "observed_differs_by_bias_pos": abs(i1["heading_offset_rad"] - b) < 1e-9,
        "observed_differs_by_bias_neg": abs(i2["heading_offset_rad"] + b) < 1e-9,
        "motor_gains_unchanged": abs(i1["motor_gain_left"] - 1.0) < 1e-15,
        "requested_identical_open_loop": (
            abs(i1["requested_left_rad_s"] - i0["requested_left_rad_s"]) < 1e-9
        ),
        "raw_rate_uncorrupted_under_fixed_bias": abs(
            i1["observed_imu_yaw_rate_rad_s"] - i1["raw_imu_yaw_rate_rad_s"]
        )
        < 1e-15,
        "heading_source": i1["heading_source"],
        "phi_frozen": i1["estimator_params"] == i0["estimator_params"],
    }
    causal["pass"] = all(
        [
            causal["observed_differs_by_bias_pos"],
            causal["observed_differs_by_bias_neg"],
            causal["motor_gains_unchanged"],
            causal["requested_identical_open_loop"],
            causal["raw_rate_uncorrupted_under_fixed_bias"],
            causal["heading_source"] == "gyro_integration",
            causal["phi_frozen"],
            const_pos,
            const_neg,
        ]
    )

    # Closed-loop downstream divergence with fixed heading bias
    env.open_loop_omega = None
    cl0 = _run(env, 0, "cl_nom", nom, n_steps=80, open_loop_omega=None)
    cl1 = _run(env, 0, "cl_fixed_pos", pos, n_steps=80, open_loop_omega=None)

    summary["episodes"] = {
        "zero": {k: v for k, v in ep0.items() if k != "log"},
        "fixed_pos": {k: v for k, v in ep1.items() if k != "log"},
        "fixed_neg": {k: v for k, v in ep2.items() if k != "log"},
        "gyro_rate_contrast": {k: v for k, v in ep3.items() if k != "log"},
    }
    summary["causal_fixed_heading"] = causal
    summary["contrast"] = {
        "fixed_offset_mean_pos": ep1["offset_mean"],
        "fixed_offset_std_pos": ep1["offset_std"],
        "fixed_offset_mean_neg": ep2["offset_mean"],
        "gyro_rate_offset_start": rate_offsets[0] if rate_offsets else None,
        "gyro_rate_offset_end": rate_offsets[-1] if rate_offsets else None,
        "gyro_rate_grows": rate_grows,
        "gyro_rate_near_bt": rate_near_linear,
        "sim_duration_s": t_end,
    }
    summary["closed_loop_downstream"] = {
        "nominal_final_yaw": cl0["metrics"]["final_true_yaw_rad"],
        "biased_final_yaw": cl1["metrics"]["final_true_yaw_rad"],
        "diverged": abs(
            cl1["metrics"]["final_true_yaw_rad"] - cl0["metrics"]["final_true_yaw_rad"]
        )
        > 1e-3,
    }
    summary["gates"] = {
        "primary_is_fixed_heading_bias": True,
        "unit_rad": True,
        "fixed_offset_constant": const_pos and const_neg,
        "signs_correct": const_pos and const_neg,
        "gyro_rate_preserved_and_distinct": rate_grows,
        "causal_separation": causal["pass"],
        "no_supervisor_controller_heading": True,
        "adaptation_off": True,
        "residual_zero": abs(ep0["metrics"]["mean_residual_abs"]) < 1e-12,
        "closed_loop_downstream": summary["closed_loop_downstream"]["diverged"],
    }
    summary["pass"] = all(summary["gates"].values())

    # Comparison table artifact
    table = os.path.join(out, "fixed_vs_rate_bias_table.txt")
    with open(table, "w", encoding="utf-8") as f:
        f.write("t_idx  fixed_offset  rate_offset\n")
        for i, (fo, ro) in enumerate(
            zip(ep1["heading_offsets_rad"], ep3["heading_offsets_rad"])
        ):
            if i % 10 == 0:
                f.write(f"{i:4d}  {fo:+.6f}  {ro:+.6f}\n")
        f.write(
            f"\nfixed mean/std: {ep1['offset_mean']:+.6f} / {ep1['offset_std']:.3e}\n"
        )
        f.write(
            f"rate end (expect ~{expected_end:.3f}): {rate_offsets[-1]:+.6f}\n"
        )

    path = os.path.join(out, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(json.dumps(summary["gates"], indent=2))
    print("PASS" if summary["pass"] else "FAIL")
    print("WROTE", path)

    env.close()
    backend.quit(0 if summary["pass"] else 1)
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
