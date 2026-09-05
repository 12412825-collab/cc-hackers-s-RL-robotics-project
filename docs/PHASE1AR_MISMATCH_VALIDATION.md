# Phase-1A-R Step 3 — Mismatch Validation

**Date:** 2026-09-06  
**Branch:** `research/adaptation-locus-phase1ar-live-webots`  
**Verdict:** **PASS**  
**Live summary:** `results/adaptation_locus_phase1ar/mismatch_validation/summary.json`

---

## Addendum — Step 3.5 (do not rewrite Step 3)

Step 3 validated **causal isolation** of the initial observation mismatch implementation, which is a **gyro yaw-rate bias** (`rad/s`).

Step 3.5 **does not invalidate** that result. It:

1. Renames that implementation to the explicit type `gyro_rate_bias` (secondary).
2. Introduces primary `fixed_heading_bias` (`rad`) for Phase-1A-R scientific replication alignment.

See `docs/PHASE1AR_SENSOR_SEMANTICS_ALIGNMENT.md`.

Historical Step-3 traces under `results/.../mismatch_validation/` remain the gyro-rate causal evidence.

---

## M1 definition — IMU yaw-rate bias (Step 3 object)

| Item | Value |
|------|-------|
| Type | Observation mismatch (**now named `gyro_rate_bias`**) |
| Injection | `MismatchLayer.apply_gyro_rate_bias` / legacy `apply_imu_bias` after raw Webots Gyro Y read |
| Formula | `observed_ω = raw_ω + b` with `b` in **rad/s** |
| Does not modify | Webots physical state, motor gains, residual, φ, PROTO |
| Logs | `raw_imu_yaw_rate_rad_s`, rate-bias fields, `observed_imu_yaw_rate_rad_s`, privileged `true_yaw_rad` |

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
| Does not modify | wheel radius, mass, friction, PROTO joints, observation bias, φ |
| Logs | requested L/R, gains, applied L/R, clip flags |

Diagnostic magnitude (not paper freeze): `δ = ±0.05`.

---

## Intervention vs downstream effect

**M1 (gyro-rate):** Direct effect is observation-rate shift only. Closed-loop physical yaw divergence is a **downstream** controller reaction.

**M2:** Direct effect is applied wheel commands only. Later IMU changes are **downstream** of ODE motion.

---

## Paired-run evidence (seed 0)

### M1 first step (open-loop ω=0)

| Check | Result |
|-------|--------|
| `observed − raw = ±0.10` | PASS |
| motor gains remain 1.0 | PASS |
| requested cmds identical vs nominal | PASS |
| φ frozen | PASS |

### M2 first step (open-loop cruise)

| Check | Result |
|-------|--------|
| within-step rate bias = 0 | PASS |
| requested cmds identical | PASS |
| gains `(1±δ)` applied correctly | PASS |
| yaw sign reverses for ±δ | PASS |
| clip fraction | **0.0** |

---

## Saturation audit

Diagnostic motor mismatch (`δ=0.05`): **clip_fraction = 0**.

---

## Scientific protocol

Step 3 did **not** freeze residual bounds, final severities, learning rates, budgets, or GO criteria.  
Adaptation remained OFF; residual remained 0.
