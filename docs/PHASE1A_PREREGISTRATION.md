# Phase-1A Preregistration — Webots Replication

**Written:** 2026-09-05 (Asia/Singapore)  
**Status:** PREREGISTERED BEFORE FINAL DECISION RUNS  
**Branch:** `research/adaptation-locus-phase1a-webots`  
**Parent:** Phase-0 tip `1515a6b` / frozen results `273a4c4` / prereg `974bba0`  
**Audit:** `docs/PHASE1A_WEBOTS_AUDIT.md`  
**Amendments:** `docs/PHASE1A_AMENDMENTS.md` (W-1, C-1, R-1A, M-1A, A2-1A)

---

## 1. Research question

Does the Phase-0 **Mismatch Source × Adaptation Locus** interaction replicate under the repository’s original higher-fidelity Webots robotics stack (parameters, kinematics, residual-on-ω path)?

This is a **cross-environment replication** study. We do not expand mismatch families.

---

## 2. Inherited hypotheses (unchanged)

### H1 — Observation mismatch (IMU bias)

Estimator adaptation (A1) should outperform residual adaptation (A2).

### H2 — Dynamics mismatch (motor asymmetry)

Residual adaptation (A2) should outperform estimator-only adaptation (A1).

### H3 — Adaptation Mislocalization (provisional label)

Wrong-locus adaptation may recover poorly, require more effort, or transfer worse. Not an established literature term.

---

## 3. Webots implementation

### Plant

Primary decision plant: **Webots-faithful plant** (Amendment W-1) reusing:

- `parts.differential_drive.DifferentialDriveKinematics`
- `parts.differential_drive.VelocityDriveMode` residual-on-ω semantics
- Webots geometry / limits from historical config/proto
- dt = 0.05 s

Live Webots backend is optional and not required for the headline decision in this environment.

### Base controller (A0/A1/A2 shared)

Classical heading P → base ω (Amendment C-1). Vision KerasPilot is **not** the Phase-1A base.

### Episode task

Straight-corridor style segment (Webots-faithful kinematics): travel `segment_length_m` while regulating heading, with corridor lateral bound. Metrics use **true** yaw.

---

## 4. Exact mismatch definitions

### M1 — IMU heading bias (observation only)

\[
\omega^{\mathrm{imu}}_{\mathrm{obs}} = \omega^{\mathrm{true}} + b_{\mathrm{imu}} + \epsilon
\]

True plant yaw integration **does not** include \(b_{\mathrm{imu}}\).

| Label | \(b_{\mathrm{imu}}\) (rad/s) |
|-------|-----------------------------:|
| 0 | 0.0 |
| small | 0.05 |
| medium | 0.15 |
| large | 0.30 |

### M2 — Motor gain asymmetry (actuator only)

\[
\omega_L^{\mathrm{applied}} = g_L\,\omega_L^{\mathrm{cmd}},\quad
\omega_R^{\mathrm{applied}} = g_R\,\omega_R^{\mathrm{cmd}}
\]

with \(g_L=1+\delta\), \(g_R=1-\delta\). Sensors/calibration unchanged by this mismatch.

| Label | \(\delta\) |
|-------|-----------:|
| 0 | 0.0 |
| small | 0.02 |
| medium | 0.04 |
| large | 0.06 |

---

## 5. Adaptation conditions

### A0 — None

Frozen estimator (`bias_hat=0`, fixed fusion weight), residual \(a=0\).

### A1 — Estimator (realistic)

Adapt **only** \(\hat{b}_{\mathrm{imu}}\) via recursive innovation vs encoder yaw proxy (Phase-0 A1-1 spirit). Cannot change residual/controller/motor gains.

Oracle A1 (privileged true rate) is diagnostic only.

### A2 — Residual

Frozen estimator. Adapt 2-parameter linear residual \(a=\tanh(\theta_0+\theta_1\tilde{\psi})\) mapped to

\[
\omega_{\mathrm{res}}=a\cdot 0.75\ \mathrm{rad/s}
\]

(Amendment R-1A / A2-1A). Cannot change calibration.

---

## 6. Budgets

| Quantity | Value |
|----------|------:|
| Online steps per cell | 2000 |
| dt | 0.05 s |
| Eval interval | 200 steps |
| Eval episodes | 4 |
| Recovery threshold (PRR) | 0.8 |
| Seeds (smoke) | {0} |
| Seeds (debug) | {0,1,2} |
| Seeds (decision) | {0,1,2,3,4} |

---

## 7. Metrics

Aligned with Phase-0:

- success, mean/final |heading_true|, lateral error
- control effort, residual magnitude, oscillation
- param magnitude, samples-to-recovery
- Performance Recovery Ratio vs nominal A0 and shifted A0
- Webots diagnostics: true yaw, observed yaw, wheel cmd vs applied

Performance score (higher better):

\[
P = -(\text{mean\_abs\_heading\_deg} + 0.1\cdot\text{mean\_abs\_lateral\_cm})
\]

---

## 8. Decision criteria

### STRONG REPLICATION / GO

At medium severity, stable across ≥4/5 seeds:

- IMU: A1 clearly better than A2 (and improves vs A0 on ≥1 primary metric)
- Motor: A2 clearly better than A1 (and improves vs A0)

### PARTIAL REPLICATION / CONDITIONAL GO

Crossing weaker, but correct-locus efficiency/effort/transfer advantages persist on one or both sides.

### FAIL TO REPLICATE

Ordering reverses, one method dominates both mismatches, effects vanish, or isolation fails.

---

## 9. Explicit non-goals

No new mismatch families; no large SAC headline; no plant tuning for Phase-0 effect sizes; no post-hoc GO-rule edits.

---

## 10. Commit note

This file must be committed **before** final decision-stage results are inspected and before `PHASE1A_RESULTS.md` is finalized.
