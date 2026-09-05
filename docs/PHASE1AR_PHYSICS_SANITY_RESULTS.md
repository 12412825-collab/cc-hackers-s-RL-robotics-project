# Phase-1A-R Step 1.5 — Physics Sanity Results

**Date:** 2026-09-06 (Asia/Singapore)  
**Branch:** `research/adaptation-locus-phase1ar-live-webots`  
**Diagnosis:** `docs/PHASE1AR_PHYSICS_SANITY_DIAGNOSIS.md`  
**Diagnostic commit:** `322f126`  
**Repair commit:** `3a45db8` (P-1AR-01)

---

## Verdict

**PASS**

**LIVE WEBOTS PHYSICS SANE ENOUGH FOR RESEARCH: YES**

All acceptance gates A–E pass after a single correctness repair (coordinate frame). No ContactProperties tuning was required. No A1/A2 / residual / mismatch / GO changes.

---

## Root cause of the ~81 m anomaly

Webots R2025a defaults to `coordinateSystem "ENU"` (**Z-up**).  
Historical NUS worlds/PROTO are authored for **Y-up (NUE)** floors and joints.

Without an explicit `coordinateSystem "NUE"`, gravity acted along **−Z** while the floor slab was thin in **Y**, so it did not support the robot. Zero-command traces fitted:

\[
a_Z = -9.81\ \mathrm{m/s^2},\quad Y \approx \mathrm{const}
\]

i.e. free-fall along −Z — not plausible wheeled travel.

---

## Correctness repairs

### P-1AR-01 — `coordinateSystem "NUE"`

| Field | Value |
|-------|-------|
| ID | P-1AR-01 |
| Old behavior | Inherit R2025a default ENU → freefall −Z (~80–100 m / 4 s) |
| Root cause | Missing coordinate-system declaration on Y-up historical worlds |
| Files | `simulation/worlds/four_wheel_track.wbt`, `four_wheel_track_phase1ar_probe.wbt`, `four_wheel_track_phase1ar_sanity.wbt`, `WebotsRobotProject/worlds/four_wheel_track.wbt` |
| Old value | (absent; default `"ENU"`) |
| New value | `coordinateSystem "NUE"` |
| Why correctness | Restores authored Y-up frame so floor opposes gravity; not friction/gain/effect-size tuning |
| Historical NUS intent | Geometry/joints imply Y-up; field omitted under older defaults |
| Scientific protocol changed? | **No** |

No P-1AR-02 ContactProperties repair applied (defaults adequate after P-1AR-01).

**Scientific tuning: NO**

---

## Scale reference (unchanged historical radius)

- wheel radius \(r = 0.0325\) m  
- track width (anchors) = 0.130 m  
- ω = 1 rad/s → \(v = r\omega = 0.0325\) m/s → ~0.13 m in 4 s  

---

## Post-repair Gate results

Command:

```text
webots --mode=fast --batch --stdout --stderr --minimize --no-rendering
  simulation/worlds/four_wheel_track_phase1ar_sanity.wbt
```

| Gate | Criterion | Result | Evidence |
|------|-----------|--------|----------|
| A | zero-cmd stable | **PASS** | dist 0.0034 m, Δy 0.0034 m, wheel rates 0 |
| B | symmetric scale ~ rωt | **PASS** | dist 0.110 m vs expected 0.130 m (ratio **0.84**) |
| C | differential yaw plausible | **PASS** | Δyaw +0.045 rad; left>right; dist ratio 0.77 |
| D | no freefall/explosion | **PASS** | no vertical jumps; ratios ≪ 20 |
| E | sim clock | **PASS** | each segment **4.000 s** for 80×50 ms |

### Segment table (post P-1AR-01)

| Test | cmd L/R (rad/s) | sim Δt (s) | Δx,Δy,Δz (m) | dist (m) | expected (m) | ratio | Δyaw (rad) |
|------|----------------:|-----------:|--------------|---------:|-------------:|------:|-----------:|
| A zero | 0 / 0 | 4.000 | ~0, 0.003, 0 | 0.003 | 0 | — | 0 |
| B sym | 1 / 1 | 4.000 | −0.107, 0.022, 0 | 0.110 | 0.130 | 0.84 | ~0 |
| C diff | 1.5 / 0.5 | 4.000 | −0.100, −0.009, −0.004 | 0.100 | 0.130 | 0.77 | +0.045 |
| D rev | −1 / −1 | 4.000 | +0.112, 0.012, −0.002 | 0.113 | 0.130 | 0.87 | −0.017 |
| Legacy | 4 / 2 | 4.000 | −0.274, −0.003, −0.003 | 0.274 | 0.390 | 0.70 | +0.040 |

Wheel PositionSensor mean rates ≈ commanded ω (rad/s confirmed).

Forward travel under +ω is primarily **−X** in this Y-up frame (convention note only).

---

## Artifacts

- `results/adaptation_locus_phase1ar/physics_sanity/A_zero_command.json`
- `.../B_symmetric_slow.json`
- `.../C_differential.json`
- `.../D_reverse.json`
- `.../LEGACY_probe_4_2.json`
- `.../summary.json` (post-repair gates)
- `.../webots_sanity_log_post_P1AR01.txt`
- Controller: `simulation/controllers/phase1ar_physics_sanity/`

Baseline (pre-repair, freefall) retained in git history at diagnostic commit `322f126`.

---

## Remaining risks (non-blocking)

1. Default ContactProperties (friction≈1, bounce≈0.5) — not validated against hardware; adequate for sanity, UNKNOWN vs real robot.
2. Chassis mass 0.60 kg vs `myconfig.ROBOT_MASS=0.800` — historical inconsistency; not changed here.
3. PROTO unused `wheelSeparation`/`wheelbase` IS warnings remain (cosmetic).
4. Slip ratios ~0.7–0.85 vs ideal no-slip — expected ODE/contact; not tuned away.

---

## Step 2 authorization

**Allowed to enter Step 2** from a physics-sanity perspective: **YES**

Do not auto-start Step 2 in this commit; await explicit instruction.
