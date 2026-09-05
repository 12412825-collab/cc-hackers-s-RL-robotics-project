# Phase-1A-R Step 1.5 — Physics Sanity Diagnosis

**Date:** 2026-09-06 (Asia/Singapore)  
**Branch:** `research/adaptation-locus-phase1ar-live-webots`  
**Parent:** Step 1 commit `9756f60`  
**Status:** DIAGNOSIS BEFORE REPAIR (no plant tuning yet)

---

## 1. Physical scale audit (historical assets)

| Quantity | Source | Value | Units |
|----------|--------|------:|-------|
| wheel radius | `FourWheelRobot.proto` `wheelRadius` | 0.0325 | m |
| wheel diameter | 2r | 0.065 | m |
| wheel width | `wheelWidth` | 0.026 | m |
| track width (PROTO field) | `wheelSeparation` | 0.130 | m (PROTO param **unused** / no `IS`) |
| track width (anchors) | wheel `|z|=0.065` | 0.130 | m |
| wheelbase (PROTO field) | `wheelbase` | 0.150 | m (unused / no `IS`) |
| wheelbase (anchors) | `|x|=0.075` | 0.150 | m |
| body box | PROTO visual/collision | 0.255 × 0.055 × 0.170 | m |
| robot chassis mass | PROTO `Physics.mass` | 0.60 | kg |
| wheel mass (each) | PROTO | 0.05 | kg |
| density | `density -1` | automatic inertia from boundingObject | — |
| COM | `centerOfMass [0 0.025 0]` | relative | m |
| floor box | world | 100 × 0.05 × 100 | m |
| floor translation | world | `(0, -0.026, 0)` | m |
| robot spawn | world | `(-3.0, 0.0325, 0)` | m |
| maxVelocity | PROTO | 12.0 | rad/s |
| maxTorque | PROTO | 0.12 | N·m |
| basicTimeStep | WorldInfo | 50 | ms |

`myconfig.py` agrees: `WHEEL_RADIUS=0.0325`, `WHEEL_SEPARATION=0.130`, `WEBOTS_TIMESTEP_MS=50`.

All values are SI as required by Webots.

### Nominal no-slip scale

\[
v = r\,\omega
\]

| ω (rad/s) | v = rω (m/s) | distance in 4 s (m) |
|----------:|-------------:|--------------------:|
| 1.0 | 0.0325 | 0.13 |
| 2.0 | 0.065 | 0.26 |
| 4.0 | 0.130 | 0.52 |
| mean(4,2)=3 | 0.0975 | 0.39 |

Step-1 probe (~81 m in 4 s) is **orders of magnitude** above this scale.

---

## 2. Wheel joint audit

| Wheel | Joint | Axis | Radius | Motor | Velocity limit | Torque limit |
| ----- | ----- | ---- | ------ | ----- | -------------- | ------------ |
| front left | HingeJoint | (0,0,1) | 0.0325 | front left wheel motor | 12 rad/s | 0.12 N·m |
| front right | HingeJoint | (0,0,1) | 0.0325 | front right wheel motor | 12 rad/s | 0.12 N·m |
| rear left | HingeJoint | (0,0,1) | 0.0325 | rear left wheel motor | 12 rad/s | 0.12 N·m |
| rear right | HingeJoint | (0,0,1) | 0.0325 | rear right wheel motor | 12 rad/s | 0.12 N·m |

- Velocity control: `setPosition(inf)` + `setVelocity` (rad/s).
- Visual + collision cylinders use `rotation 1 0 0 1.5708` so cylinder axis ‖ Z; hinge axis ‖ Z → consistent for **Y-up** rolling about ±X.
- Live encoder rates under command ≈ commanded ω → **not** a rad/s↔m/s unit bug.

---

## 3. Collision geometry audit

| Body | boundingObject | Notes |
|------|----------------|-------|
| floor | Box 100×0.05×100 | Present; thin axis = **Y** |
| chassis | Box 0.255×0.055×0.170 at y=0.0275 | Present |
| each wheel | Cylinder r=0.0325, h=0.026 (rotated) | Matches visual radius |

Intended Y-up contact: wheel bottom at spawn Y≈0, floor top ≈ −0.001 m → ~1 mm gap (OK for Y-up).

Under default R2025a **ENU (Z-up)**, this floor slab is **not** a horizontal supporting surface against gravity (see root cause).

---

## 4. Physics nodes

| Solid | Physics | Mass | density | inertia |
|-------|---------|-----:|---------|---------|
| Robot chassis | yes | 0.60 | −1 (auto) | auto from BO |
| Each wheel | yes | 0.05 | −1 (auto) | auto from BO |
| Floor | **none** (static) | — | — | correct pattern for static floor |

No obviously massless dynamic bodies. Masses are small-robot plausible (UNKNOWN vs real hardware).

---

## 5. ContactProperties

WorldInfo has **empty** `contactProperties`.

Webots therefore applies default material pair `"default"`/`"default"`:

- coulombFriction ≈ 1
- bounce ≈ 0.5
- softERP/softCFM defaults

No custom wheel–ground materials in the historical NUS world.

---

## 6. Coordinate / initial-pose investigation (highest priority)

### Webots R2025a WorldInfo default

```text
coordinateSystem  "ENU"   # X East, Y North, Z Up
gravity           9.81    # along the down axis
```

Historical world **does not set** `coordinateSystem`, therefore inherits **ENU (Z-up)**.

Historical geometry is **Y-up / NUE-style**:

- floor thin dimension = Y
- robot spawn height on Y
- wheel hinge axis = Z (roll in X for Y-up)

### Interpretation of Step-1 final pose `z ≈ -81`

| Hypothesis | Evidence | Verdict |
|------------|----------|---------|
| A. Fast ground travel | wheel rates match cmds but zero-cmd also moves ~99 m with ω=0 | Rejected as primary |
| B. Free-fall / missing support | zero-cmd: a_z = **−9.81** m/s² exactly; y constant | **Accepted** |
| C. Axis misread only | Y stays flat while Z accelerates at g | Partially: gravity axis ≠ geometry up-axis |
| D. Wrong Supervisor node | name `four_wheel_robot`; consistent across tests | Rejected |

### Live baseline (pre-repair) — Gate suite

Simulation clock: **4.000 s** per 80 steps @ 50 ms → Gate E OK.

| Test | Δposition [m] | Δy [m] | distance [m] | wheel rates | notes |
|------|---------------|-------:|-------------:|-------------|-------|
| A zero | (0, 0, −99.08) | 0 | 99.08 | all 0 | freefall −Z |
| B sym ±1 | mostly −Z ~99 | +0.05 | 99.04 | ≈±1 | same |
| C diff | mostly −Z ~99 | +0.04 | 99.04 | ≈1.5/0.5 | same |
| D rev | mostly −Z ~99 | +0.03 | 99.07 | ≈−1 | same |
| Legacy 4/2 | mostly −Z ~99 | +0.22 | 98.91 | ≈4/2 | matches Step-1 class |

Quadratic fit on A_zero Z(t):

\[
z(t) = z_0 + v_0 t + \tfrac12 a t^2,\quad a = -9.81\ \mathrm{m/s^2}
\]

**Conclusion:** motion is gravitational free-fall along **−Z** because ENU down-axis is Z, while the “floor” does not oppose that axis.

---

## 7. Root-cause ranking (evidence-ordered)

1. **Missing `coordinateSystem "NUE"` on historical worlds under Webots R2025a default ENU** — **PRIMARY**. Explains zero-command freefall, ~g acceleration, ~80–100 m / 4 s scale, and Step-1 81 m anomaly.
2. Floor collision thin-axis aligned to Y (correct only for NUE) — consequence of (1), not an independent missing BO.
3. No custom ContactProperties — secondary; relevant after gravity/frame is corrected.
4. Unused PROTO `wheelSeparation`/`wheelbase` IS fields — cosmetic warnings; not causal.
5. Motor unit error — **ruled out** (encoder rates ≈ commands).
6. Simulation time mismatch — **ruled out** (exactly 4.0 s).
7. Python pose integration / W-1 plant — **ruled out** (Supervisor-only logging).

---

## 8. Proposed correctness repair (not yet applied in this document)

**P-1AR-01 (planned):** set

```text
WorldInfo {
  coordinateSystem "NUE"
  ...
}
```

on historical / probe / sanity worlds so gravity down-axis = **−Y**, matching the Y-up floor and robot PROTO.

Why correctness (not tuning):

- Restores the coordinate convention the historical NUS assets were authored for.
- Does not change mass, friction targets for A1/A2, residual bounds, or mismatch severities.
- Required for any wheel–ground contact to oppose gravity.

Optional follow-ups **only if still failing after P-1AR-01**:

- P-1AR-02: explicit ContactProperties for wheel/floor (only if default contact still non-physical).
- P-1AR-03: floor height micro-adjustment if residual penetration remains.

---

## 9. Gate status before repair

| Gate | Result |
|------|--------|
| A zero stable | **FAIL** |
| B symmetric scale | **FAIL** (~762×) |
| C differential | **FAIL** (dominated by freefall) |
| D no explosion | **FAIL** |
| E sim clock | **PASS** |

**Pre-repair decision: BLOCKED for research use until coordinate-frame correctness repair.**
