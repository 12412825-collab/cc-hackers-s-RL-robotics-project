# Phase-1A-R Step 3.5 — Sensor Semantics Alignment

**Date:** 2026-09-06  
**Branch:** `research/adaptation-locus-phase1ar-live-webots`  
**Verdict:** **PASS**

---

## Purpose

Align the **primary** Phase-1A-R observation mismatch with Phase-0 / Phase-1A **fixed heading bias** semantics (`rad`), without using Supervisor ground truth for controller-facing heading.

Step 3 remains valid: it causally isolated the initial **gyro-rate bias** (`rad/s`) implementation. That mismatch is retained under the explicit name `gyro_rate_bias` as a secondary / Phase-1B candidate — **not** the primary decision-matrix M1.

---

## Cross-phase semantic mapping

| Phase | Primary observation mismatch | Unit | Meaning |
| ----- | ---------------------------- | ---- | ------- |
| Phase-0 | IMU heading path (rate→integrated heading) | deg/s on rate → deg heading | accumulating heading error from rate bias* |
| Phase-1A | same Webots-faithful path | rad/s on rate | accumulating |
| Phase-1A-R before 3.5 | `gyro_rate_bias` | rad/s | accumulating drift |
| Phase-1A-R after 3.5 | **`fixed_heading_bias`** | **rad** | **fixed offset** |

\*Phase-0 preregistration injects bias on measured rate then integrates; the **effective controller-facing quantity** is a drifting heading. Step 3.5 makes the primary live intervention an explicit **constant heading offset**, which is the intended Phase-0/1A scientific object for “heading bias” in Adaptation Locus terms.

---

## Primary heading source (Option B)

| Item | Value |
|------|-------|
| Choice | **Integrated historical Webots Gyro** |
| Why not InertialUnit | Absent from `FourWheelRobot.proto` |
| Why not Supervisor | Forbidden for controller-facing heading |
| Integration | `θ_t = wrap(θ_{t-1} + ω_t · dt)` with `dt = basicTimeStep/1000` (50 ms) |
| Initial θ | Known spawn heading `0.0` rad from experiment config (world `rotation 0 1 0 0`) — **not** runtime Supervisor readout |
| Noise | No research noise added; PROTO Gyro has no R2025a noise field |
| Research glue | `GyroHeadingIntegrator` + heading-space φ fusion |

`heading_source` logged as `"gyro_integration"` on every controller observation.

---

## Mismatch types (explicit names)

### PRIMARY — `fixed_heading_bias` [rad]

```text
raw Gyro ω
→ integrate → raw_heading
→ observed_heading = wrap(raw_heading + b_θ)
→ research estimator φ (heading-space)
→ controller
```

Episode-constant `b_θ`. Does not alter motors, PROTO, or raw Gyro samples.

### SECONDARY — `gyro_rate_bias` [rad/s] (Step 3 preserved)

```text
raw Gyro ω
→ observed_ω = raw + b_ω
→ integrate biased rate for controller path
```

Accumulating heading error. Retained for contrast / future Phase-1B.

### M2 — `motor_asymmetry` (unchanged)

---

## Validation evidence

Live summary: `results/adaptation_locus_phase1ar/sensor_semantics/summary.json`

| Check | Result |
|-------|--------|
| +0.10 rad offset constant over ~4 s | PASS (std ~1e-16) |
| −0.10 rad offset constant | PASS |
| gyro-rate contrast grows (~0.005 → ~0.40) | PASS |
| causal: observed−raw = ±b; gains 1.0; rate channel clean | PASS |
| Supervisor not in controller heading | PASS |
| adaptation OFF / residual 0 | PASS |
| closed-loop downstream divergence | PASS |

---

## Firewall

ControllerObservation contains sensor-derived headings only. Privileged `true_yaw_rad` remains eval/log only.
