# Phase-1A Webots Reconstruction Audit

**Date:** 2026-09-05 (Asia/Singapore)  
**Parent tip:** `research/adaptation-locus-phase0` @ `1515a6b` (frozen results `273a4c4`)  
**Branch:** `research/adaptation-locus-phase1a-webots`  
**Mode:** read-only reconstruction before Phase-1A behavior changes.

---

## 1. Executive finding

The historical NUS Webots stack is a **DonkeyCar Vehicle loop synchronized to Webots `robot.step()`**. It provides:

- `FourWheelRobot` proto + `four_wheel_track.wbt`
- `WebotsAdapter` sensor/actuator I/O
- `VelocityDriveMode` + `DifferentialDriveKinematics` (`v`/`ω` → wheel rad/s)
- optional 9-D `SensorFusion` normalize
- optional vision residual SAC (`ResidualPilot`)

It does **not** provide Phase-0’s Adaptation Locus apparatus:

- classical heading P-controller
- online IMU-bias estimator (A1)
- observation-only / actuator-only mismatch injection
- Gym-style episode reset / PRR metrics
- headless batch experiment driver

**Live Webots Python `controller` API is unavailable in this agent environment** (`ImportError` for `controller`; no `WEBOTS_HOME` / `webots` binary). Phase-1A therefore inherits the **historical Webots plant parameters and control equations** via a Webots-faithful offline plant that reuses `DifferentialDriveKinematics` / `VelocityDriveMode` / `myconfig` Webots constants, and exposes a live `WebotsAdapter` backend hook for later execution when Webots is installed. This is documented as Amendment W-1 (not research tuning).

---

## 2. Control-path diagram (historical Webots / DonkeyCar)

```text
Webots ODE (FourWheelRobot)
  │  Supervisor ground truth: pos, vel, CTE  [logging]
  ▼
Devices: Camera, Accel, Gyro, DistanceSensor, PositionSensors
  ▼
simulation/webots_adapter.py :: WebotsAdapter.run / _outputs
  → cam/image_array
  → enc/left_speed, enc/right_speed, enc/speed   [rad/s, m/s]
  → imu/acl_* [g], imu/gyr_* [deg/s]
  → obs/distance [cm]
  → pos/*
  ▼
parts/sensors.py :: SensorFusion (optional) → 9-D normalize
  ← NOT a heading estimator
  ▼
(optional) KerasPilot → pilot/angle, pilot/throttle
(optional) ResidualPilot → residual/angular_velocity [rad/s]
  ▼
parts/differential_drive.py :: VelocityDriveMode
  → v [m/s], ω [rad/s]  (ω += residual)
  ▼
WebotsAdapter + DifferentialDriveKinematics
  → left/right wheel ω [rad/s] → RotationalMotor.setVelocity
  → Supervisor.step(WEBOTS_TIMESTEP_MS)
```

**Missing vs Phase-0 target chain:** no `estimator → classical heading controller` stage on the Webots path today.

---

## 3. Stage inventory

| Stage | Source | Symbol / lines | Units | Rate | Historical? | Phase-1A glue? |
|-------|--------|----------------|-------|------|-------------|----------------|
| World clock | `simulation/worlds/four_wheel_track.wbt` | `basicTimeStep` 50 | ms | 20 Hz | Yes | Reuse |
| Robot plant | `simulation/protos/FourWheelRobot.proto` | mass/wheels/motors | SI | — | Yes | Reuse params |
| Controller entry | `simulation/controllers/donkey_webots/donkey_webots.py` | launches `manage.py drive` | — | — | Yes | Not for batch |
| Adapter | `simulation/webots_adapter.py` | `WebotsAdapter` 20–260 | SI + deg/s | sync step | Yes | Mismatch hooks |
| Kinematics | `parts/differential_drive.py` | `DifferentialDriveKinematics` 86–109 | m/s, rad/s → rad/s | — | Yes | Reuse |
| Drive mode | `parts/differential_drive.py` | `VelocityDriveMode` 25–63 | residual on ω | — | Yes | Reuse residual path |
| Sensor normalize | `parts/sensors.py` | `SensorFusion` | 9-D [-1,1] | cfg 20 Hz | Yes | Not A1 |
| Vision residual | `parts/residual_rl.py` | `ResidualPilot` | ±0.75 rad/s | — | Yes | **Not** primary A2 |
| Wiring | `manage.py` | ~305–313, 467–542, 952–971 | — | — | Yes | Episode driver new |
| Config | `myconfig.py` | Webots/RL 765–961 | — | — | Yes | Document freeze |
| Phase-0 lab | `research/adaptation_locus/*` | `Phase0CorridorEnv` | cm, deg, dt=0.1 | 10 Hz | Phase-0 | Mapping only |

---

## 4. Frozen Webots nominal plant (do not tune for effect size)

| Parameter | Value | Source |
|-----------|------:|--------|
| Body mass | 0.60 kg (+ 4×0.05 wheel ≈ **0.80 kg** total) | proto / `ROBOT_MASS` |
| Wheel radius \(R\) | **0.0325 m** | proto / `WHEEL_RADIUS` |
| Wheel separation (track) | **0.130 m** | proto / `WHEEL_SEPARATION` |
| Wheelbase | **0.150 m** | proto |
| Max wheel speed | **12.0 rad/s** | proto / `MAX_WHEEL_SPEED` |
| Motor torque | **0.12 N·m** (provisional) | proto |
| Max linear velocity | **0.20 m/s** | `MAX_LINEAR_VELOCITY` |
| Max angular velocity | **1.50 rad/s** | `MAX_ANGULAR_VELOCITY` |
| Timestep | **50 ms** (20 Hz) | world / `WEBOTS_TIMESTEP_MS` |
| Gyro noise | 0.005 (proto) | Gyro node |
| Accel noise | 0.01 (proto) | Accelerometer |
| Distance sensor | sonar LUT 0.02–2.0 m → ×100 cm | adapter |
| Residual angular scale | **0.75 rad/s** (raw × scale) | `RESIDUAL_ANGULAR_SCALE` |
| `SIMULATOR` default | `'none'` (path dormant) | `myconfig.py` |
| `RESIDUAL_RL` default | `False` | `myconfig.py` |

---

## 5. Sensor channel units (WebotsAdapter)

| Channel | Unit | Transform |
|---------|------|-----------|
| `enc/left_speed`, `enc/right_speed` | rad/s | Δθ/dt mean of side |
| `enc/speed` | m/s | \(R(\omega_L+\omega_R)/2\) |
| `imu/gyr_*` | **deg/s** | `math.degrees(gyro)` |
| `imu/acl_*` | g | `/9.80665` |
| `obs/distance` | cm | raw m × 100 |
| Supervisor pose | m | ground truth |

---

## 6. Natural residual authority (Webots path)

Historical Webots residual (scheme A):

1. raw \(a \in [-1,1]\)
2. \(\omega_{\mathrm{res}} = a \times 0.75\) rad/s
3. \(\omega = \mathrm{clip}(\omega_{\mathrm{base}}+\omega_{\mathrm{res}}, \pm 1.50)\)

Phase-0 residual was heading degrees (±10° amended). Phase-1A primary A2 must translate the **same low-dimensional residual policy** into this **angular-velocity residual** space, not swap in large vision SAC.

---

## 7. Mismatch injection points (planned)

| Mismatch | Must affect | Must not affect | Injection site |
|----------|-------------|-----------------|----------------|
| IMU heading bias | observed gyro / fused heading | true Webots/plant yaw dynamics | after true rate, before estimator |
| Motor gain asymmetry | applied left/right wheel commands | IMU bias / calibration params | after kinematics, before plant integrate |

---

## 8. Mapping table: Phase-0 → Webots

| Phase-0 concept | Webots equivalent | Existing or new | Notes |
| --------------- | ----------------- | --------------- | ----- |
| True heading | Supervisor yaw / plant integrated yaw | Existing GT / new compute | Webots axes must be documented |
| IMU observation | `imu/gyr_*` deg/s | Existing | Bias injection new |
| Encoder observation | `enc/left_speed`, `enc/right_speed` | Existing | Yaw proxy new |
| Motor gain \(g_L,g_R\) | scale wheel ω commands | **New** | After kinematics |
| Steering/yaw authority | `MAX_ANGULAR_VELOCITY` 1.50 rad/s | Existing | Freeze |
| Residual action | `residual_omega` rad/s (±0.75) | Existing path | Low-dim A2 new |
| Base controller | heading P in Phase-0 | **New** classical part | Vision Keras ≠ Phase-0 base |
| Estimator A1 | none on Webots path | **New** (port Phase-0) | Bias-only primary |
| Tracking error | true heading abs | New metric | Align with Phase-0 score |
| Success criterion | segment complete | New corridor episode | Webots world is open track |
| dt | 0.1 s Phase-0 | **0.05 s** Webots | Document |
| Track width | 0.18 m Phase-0 | **0.130 m** Webots | Inherit Webots |
| Speed | 0.20 m/s Phase-0 base | 0.20 m/s max Webots | Match nominal cruise |

---

## 9. Environment availability

| Check | Result |
|-------|--------|
| `webots` on PATH | Not found |
| `WEBOTS_HOME` | Unset |
| `import controller` | Fail |
| Offline reuse of kinematics/config | Available |

**Implication:** final Phase-1A decision runs in this environment use the **Webots-faithful plant** (Amendment W-1). Live Webots backend is implemented behind the same interface for machines with Webots installed.

---

## 10. Audit conclusion

Reuse the historical Webots **geometry, units, kinematics, residual-on-ω path, and sensor channel semantics**. Add minimal Phase-1A glue for mismatch injection, classical heading base control, A1/A2, and episode metrics. Do not replace the robot proto or invent a new robot identity. Do not treat vision SAC as the Phase-1A residual locus.

**No production Webots files were modified to produce this audit.**
