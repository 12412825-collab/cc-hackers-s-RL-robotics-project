# Phase-1A-R Step 3 — Mismatch Validation

**Date:** 2026-09-06  
**Branch:** `research/adaptation-locus-phase1ar-live-webots`  
**Verdict:** **PASS**  
**Live summary:** `results/adaptation_locus_phase1ar/mismatch_validation/summary.json`

---

## M1 definition — IMU yaw-rate bias

| Item | Value |
|------|-------|
| Type | Observation mismatch |
| Injection | `MismatchLayer.apply_imu_bias` after raw Webots Gyro Y read |
| Formula | `observed_ω = raw_ω + b` with `b` in **rad/s** |
| Does not modify | Webots physical state, motor gains, residual, φ, PROTO |
| Logs | `raw_imu_yaw_rate_rad_s`, `mismatch_imu_bias_rad_s`, `observed_imu_yaw_rate_rad_s`, privileged `true_yaw_rad` |

Diagnostic magnitude (not paper freeze): `b = ±0.10 rad/s`.

### Noise audit

- PROTO Gyro/Accel: no research noise added in Step 3.
- Historical `noise` fields were removed for R2025a compatibility (Step 1); no external stochastic sensor noise layer.
- Cross-run raw gyro may differ slightly after independent ODE resets; causal tests use **within-step** identities.

---

## M2 definition — motor gain asymmetry

| Item | Value |
|------|-------|
| Type | Dynamics / actuator mismatch |
| Injection | `MismatchLayer.apply_motor_gains` immediately before `setVelocity` |
| Formula | `gL=1+δ`, `gR=1-δ`; `applied = clip(g · requested, ±max_wheel_speed)` |
| Does not modify | wheel radius, mass, friction, PROTO joints, IMU bias, φ |
| Logs | requested L/R, gains, applied L/R, clip flags |

Diagnostic magnitude (not paper freeze): `δ = ±0.05`.

---

## Intervention vs downstream effect

**M1:** Direct effect is observation shift only. Closed-loop physical yaw divergence is a **downstream** controller reaction (confirmed in live paired closed-loop run).

**M2:** Direct effect is applied wheel commands only. Later IMU changes are **downstream** of ODE motion.

---

## Paired-run evidence (seed 0)

### M1 first step (open-loop ω=0)

| Check | Result |
|-------|--------|
| `observed − raw = +0.10` | PASS |
| `observed − raw = −0.10` | PASS |
| motor gains remain 1.0 | PASS |
| requested cmds identical vs nominal | PASS |
| φ frozen | PASS |
| spawn restored | PASS |

### M2 first step (open-loop cruise)

| Check | Result |
|-------|--------|
| within-step IMU bias = 0 | PASS |
| requested cmds identical vs nominal | PASS |
| gains `(1±δ)` and applied = g·requested | PASS |
| yaw sign reverses for ±δ | PASS (`yaw_pos≈-0.0049`, `yaw_rev≈+0.0287`) |
| clip fraction | **0.0** / **0.0** |
| φ frozen | PASS |

### Closed-loop M1 downstream

Nominal final yaw ≈ 0.0346 vs biased ≈ 0.0362 → diverged (downstream effect).

---

## Tests

Unit + live:

```text
pytest tests/test_adaptation_locus_phase1ar_mismatch.py
pytest tests/test_adaptation_locus_phase1ar_live_adapter.py
```

Live gates in `summary.json`: all true.

---

## Saturation audit

Diagnostic motor mismatch (`δ=0.05`) under cruise: **clip_fraction = 0**.  
No confound from silent saturation at this diagnostic severity.

---

## Remaining confounds

1. Cross-run raw gyro not bitwise identical after independent resets (documented; not treated as M1 leak).
2. Diagnostic severities are **not** preregistered paper values.
3. M1 acts on Gyro **rate**; no InertialUnit absolute yaw sensor in historical PROTO.
4. Closed-loop M1 physical-yaw effect size is small over 60 steps with encoder fusion — mechanism still isolates observation path.

---

## Scientific protocol

Step 3 did **not** freeze residual bounds, final severities, learning rates, budgets, or GO criteria.  
Adaptation remained OFF; residual remained 0.
