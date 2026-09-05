# Phase-1A-R Step 1 — Live Webots Runtime Audit

**Date:** 2026-09-05 (Asia/Singapore)  
**Branch:** `research/adaptation-locus-phase1ar-live-webots`  
**Parent tip:** Phase-1A annotated tip `75b2f3f`  
**Scope:** runtime / physics provenance only. No scientific decision runs.

---

## Environment

| Item | Value |
|------|-------|
| OS | Windows 10 (build 26200), win32 |
| Webots version | **R2025a** (`resources/version.txt`; `webots --version`) |
| Executable | `C:\Program Files\Webots\msys64\mingw64\bin\webots.exe` |
| Companion | `...\bin\webotsw.exe`, `webots-bin.exe`, `webots-controller.exe` |
| WEBOTS_HOME | `C:\Program Files\Webots` (set in environment) |
| Python (host) | 3.11.9 (`...\Python311\python.exe`) |
| Python (controller runtime.ini) | repo `.venv\Scripts\python.exe` (3.11.9) |
| Controller API | Available when `WEBOTS_HOME\lib\controller\python` is on `PYTHONPATH`, or when launched by Webots |
| `import controller` without path | fails in bare shell; **succeeds** under Webots-launched controller |

### Install discovery (bounded)

Checked / found:

- `C:\Program Files\Webots\` — **present** (full install tree)
- `WEBOTS_HOME` env — **set**
- `webots.exe` under `msys64\mingw64\bin\` — **found**
- `C:\Program Files (x86)\Webots\` — missing
- user AppData Programs / Roaming Webots — missing
- PATH did not contain `webots` shim; executable resolved via `WEBOTS_HOME`

---

## Historical assets

| Asset | Path | Role | Loaded successfully |
| ----- | ---- | ---- | ------------------- |
| Primary world | `simulation/worlds/four_wheel_track.wbt` | Historical NUS track world | YES (batch/pause load) |
| Probe world | `simulation/worlds/four_wheel_track_phase1ar_probe.wbt` | Same assets; controller override for provenance only | YES |
| Robot PROTO | `simulation/protos/FourWheelRobot.proto` | Four-wheel differential robot + physics | YES (with noted warnings) |
| Mirror PROTO/world | `WebotsRobotProject/...` | Duplicate historical tree | Present (not used for probe) |
| Default controller | `simulation/controllers/donkey_webots/` | Launches `manage.py drive` | Process starts under Webots |
| Probe controller | `simulation/controllers/phase1ar_physics_probe/` | Step-1 motor+Supervisor provenance | YES |
| Adapter | `simulation/webots_adapter.py` | DonkeyCar ↔ Webots devices | Not exercised in Step 1 probe |
| Config | `myconfig.py` Webots block | Timestep 50 ms; 4-motor names; residual scale 0.75 | Documented |

### Robot / devices (from PROTO)

- **Name:** `four_wheel_robot`
- **Supervisor:** TRUE
- **Default controller:** `donkey_webots`
- **Timestep:** `WorldInfo.basicTimeStep = 50` ms
- **Motors:** `front/rear` × `left/right wheel motor` (`RotationalMotor`, maxVelocity 12, maxTorque 0.12)
- **Encoders:** matching `PositionSensor` on each wheel joint
- **IMU:** `Accelerometer` + `Gyro` (no InertialUnit node)
- **Other:** `Camera`, `DistanceSensor` (sonar)
- **Physics:** body mass 0.60 kg; each wheel mass 0.05 kg; HingeJoint endPoint Solids with `physics Physics`
- **Geometry params:** wheelRadius 0.0325 m; wheelWidth 0.026; wheelSeparation PROTO field unused in tree; wheelbase PROTO field unused in tree

---

## Runtime test

### Command (recorded)

```text
C:\Program Files\Webots\msys64\mingw64\bin\webots.exe
  --mode=fast --batch --stdout --stderr --minimize --no-rendering
  <repo>\simulation\worlds\four_wheel_track_phase1ar_probe.wbt
```

Launcher:

```bash
python -m research.adaptation_locus.run_phase1ar_runtime_probe
```

### Results (from `results/.../runtime_probe/physics_provenance.json`)

| Quantity | Value |
|----------|-------|
| world | `four_wheel_track_phase1ar_probe.wbt` (historical PROTO + track) |
| controller | `phase1ar_physics_probe` |
| timestep | 50 ms |
| motor command | left wheels +4.0 rad/s; right wheels +2.0 rad/s |
| number of steps | 80 (4.0 s sim time) |
| initial position [m] | `[-3.0, 0.0325, -0.024525]` |
| initial yaw [rad] | `0.0` |
| final position [m] | `[-3.210, 0.295, -81.229]` |
| final yaw [rad] | `1.019` |
| Δ distance [m] | `81.205` |
| Δ yaw [rad] | `1.019` |
| pose source | `Supervisor.getSelf().getPosition()` + `rotation` field |
| Python pose integration | **false** |

Also verified: historical `four_wheel_track.wbt` loads under Webots batch/pause (PROTO warnings only; controller process starts).

---

## Physics provenance

> Was robot motion produced by a live Webots physics process?

### **YES**

Evidence:

1. Pose before/after was read exclusively from the live Webots Supervisor node state.
2. Wheel `RotationalMotor.setVelocity` commands were applied to historical device names.
3. Motion advanced only via `Supervisor.step(timestep)` (Webots ODE clock).
4. Probe code does **not** contain `x += ...` / `yaw += ...` plant integration.
5. Phase-0 / Phase-1A faithful plants were **not** used.

**Caveat (not a NO):** displacement magnitude (~81 m in 4 s) is physically extreme relative to commanded wheel speed × radius (~0.1 m/s scale). This strongly suggests incomplete contact/friction / ContactProperties on the floor–wheel interface under default Webots contact. Provenance of motion is still Webots physics; **stability of the historical plant for science is not yet validated** and is a Step-2 concern.

---

## Correctness fixes applied in Step 1

| Fix | Reason | Scientific params changed? |
|-----|--------|----------------------------|
| Removed obsolete `noise` fields from `Accelerometer` / `Gyro` in both PROTO copies | R2025a rejects these fields (`Skipped unknown 'noise' field`) | No (fields were already skipped at load; historical intended noise values documented in PROTO comments) |

Remaining non-fatal PROTO warnings (not repaired in Step 1):

- `wheelSeparation` / `wheelbase` PROTO parameters have no matching `IS` field (geometry hard-coded in joint anchors).

---

## Explicit non-claims

- Step 1 does **not** produce Phase-1A-R scientific evidence.
- Step 1 does **not** re-run Phase-1A W-1 plant.
- Step 1 does **not** freeze residual bounds, mismatches, or GO criteria.

---

## Artifacts

- `docs/PHASE1AR_LIVE_WEBOTS_RUNTIME_AUDIT.md` (this file)
- `simulation/controllers/phase1ar_physics_probe/`
- `simulation/worlds/four_wheel_track_phase1ar_probe.wbt`
- `research/adaptation_locus/run_phase1ar_runtime_probe.py`
- `results/adaptation_locus_phase1ar_live_webots/runtime_probe/physics_provenance.json`
- `results/adaptation_locus_phase1ar_live_webots/runtime_probe/webots_probe_log.txt`

---

## Step 1 decision

**Status: PASS WITH CAVEAT**

**LIVE WEBOTS PHYSICS CONFIRMED: YES**

Blocker for later scientific steps (not for Step 1 pass): validate / correct wheel–floor contact so the historical robot remains on the track under modest motor commands, without research tuning aimed at Adaptation Locus effect sizes.
