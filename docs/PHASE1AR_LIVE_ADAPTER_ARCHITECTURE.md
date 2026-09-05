# Phase-1A-R Step 2 — Live Adapter Architecture

**Date:** 2026-09-06  
**Branch:** `research/adaptation-locus-phase1ar-live-webots`  
**Plant:** live Webots R2025a ODE (`coordinateSystem "NUE"`, P-1AR-01)

---

## A. System diagram

```text
Webots ODE (FourWheelRobot + floor)
        │
        ▼
Historical devices (H)
  Gyro / Accel / PositionSensors / DistanceSensor / RotationalMotors
        │
        ▼
LiveWebotsBackend (R wrapper around H semantics)
  units: gyro rad/s→deg/s channels; distance m→cm; encoders → rad/s
        │
        ├──► mismatch.observe_imu_*  (R hook; Step-2 value = 0)
        │
        ▼
HeadingEstimator φ (N)
  φ = {imu_bias_hat_rad_s, fusion_weight}
  heading_est ← integrate(w·(imu−b̂)+(1−w)·encoder_yaw_rate)
  adaptation OFF in Step 2
        │
        ▼
HeadingPController (N / C-1 spirit)
  ω_base = clip(−kp · heading_est, ±ω_max)
        │
        ▼
ResidualHook → VelocityDriveMode (H Scheme A)
  ω_final = ω_base + a·0.75     (Step-2: a = 0)
        │
        ▼
DifferentialDriveKinematics (H)
  (v, ω) → (ω_L, ω_R)
        │
        ├──► mismatch.apply_motor_gains (R hook; Step-2 δ = 0)
        │
        ▼
RotationalMotor.setVelocity → Supervisor.step(dt)
        │
        ▼
Webots ODE updates pose  (NOT Python x/y/yaw integration)

Supervisor true pose/yaw ──► privileged_eval_only logging / metrics
                             (firewall: excluded from ControllerObservation)
```

---

## B. Inheritance map

| Component | Source | Classification | Reason |
| --------- | ------ | -------------- | ------ |
| World / PROTO / motors / sensors | `simulation/worlds`, `FourWheelRobot.proto` | **H** | Historical NUS assets |
| `coordinateSystem "NUE"` | WorldInfo | **C** | P-1AR-01 R2025a correctness |
| Device names / units / resetPhysics | `simulation/webots_adapter.py` pattern | **H** | Same channels & reset |
| `DifferentialDriveKinematics` | `parts/differential_drive.py` | **H** | Unchanged |
| `VelocityDriveMode` residual-on-ω | `parts/differential_drive.py` | **H** | Scheme A |
| `SensorFusion` 9-D normalizer | `parts/sensors.py` | **H** (available) | Not a heading estimator; not on critical Step-2 loop |
| KerasPilot / ResidualPilot SAC | `manage.py` / `residual_rl.py` | **H** (not used) | Vision/SAC not Adaptation Locus base |
| Heading P-controller | `live_webots/controller.py` | **N** | Historical Webots path had no heading P |
| Heading estimator φ | `live_webots/estimator.py` | **N** | Online φ for future A1; low-dimensional |
| Residual / mismatch / episode API | `live_webots/*` | **R** | Thin research hooks |
| Live validation controller | `phase1ar_live_adapter` | **R** | Batch episode driver |

---

## C. Privileged-state firewall

| Signal | Source | Controller input? | Logging? |
|--------|--------|-------------------|----------|
| true position / yaw / speed | Supervisor | **No** | Yes (`privileged_eval_only`) |
| raw IMU / encoders / distance | Webots devices | **Yes** | Yes |
| heading_est / φ | Estimator | **Yes** | Yes |
| tracking error for metrics | derived from true yaw | **No** (metrics only) | Yes |

`ControllerObservation` deliberately omits true yaw/position. Validation calls `assert_controller_obs_firewall`.

---

## D. Parameter ownership (future)

| Condition | May change | Must freeze |
|-----------|------------|-------------|
| **A0** | nothing | φ, residual, base kp, motor map |
| **A1** | φ only (`imu_bias_hat`, optionally fusion weight) | residual, kp, motor gains |
| **A2** | residual action / low-dim residual params | φ, kp, calibration |

Step 2: all adaptation **OFF**; residual **0**; mismatch **0**.

---

## E. Zero-intervention equivalence

Compared on the **same live ODE**:

1. **Historical mode:** `VelocityDriveMode` cruise (`steering=0`, `residual=0`)
2. **Research mode:** heading-P + residual=0 + mismatch=0

Measured (validation summary):

| Metric | Historical | Research |
|--------|----------:|---------:|
| distance (80 steps) | ~0.324 m | ~0.340 m |
| mean \|yaw\| | ~1.7e-5 rad | ~1.0e-5 rad |
| mean wheel cmd | 3.692307… | 3.692307… |

Wheel commands match to numerical noise; trajectories agree within the Step-2 equivalence gate.

---

## Mass / friction

See `docs/PHASE1AR_OPEN_PHYSICS_ISSUES.md`. PROTO is live mass source of truth; friction untouched.
