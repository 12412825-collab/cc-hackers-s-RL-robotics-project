# Phase-0 Preregistration — Adaptation Locus Kill Test

**Written:** 2026-09-05 (Asia/Singapore)  
**Status:** PREREGISTERED BEFORE FINAL LARGE RUNS  
**Branch:** `research/adaptation-locus-phase0`  
**Parent audit:** `docs/ADAPTATION_LOCUS_SYSTEM_AUDIT.md`

This document freezes the scientific protocol. Hypotheses and kill criteria must not be altered based on intermediate smoke results. Implementation details may be clarified only if they do not change the hypotheses or decision rules.

---

## 1. Research question

When a robotic system experiences structured mismatch, where should adaptation occur?

We study the interaction:

**Mismatch Source × Adaptation Locus**

Mismatch sources (Phase-0 only):

1. Observation / estimation mismatch — IMU heading bias
2. Dynamics / actuator mismatch — left–right motor gain asymmetry

Adaptation loci:

1. A0 — none (frozen controller)
2. A1 — estimator / calibration adaptation only
3. A2 — residual control policy adaptation only

We are **not** testing “RL beats PID.”

---

## 2. Frozen hypotheses

### H1 — Observation mismatch

For IMU heading bias, **A1 (estimator adaptation)** should achieve faster or cleaner recovery than **A2 (residual policy adaptation)**.

Expected signals (any subset is supportive; all need not hold):

- fewer online samples to recover a fixed fraction of nominal performance
- lower steady-state / final heading error
- smaller residual / compensatory control action
- better transfer to nearby unseen bias severity

### H2 — Dynamics mismatch

For left–right motor gain asymmetry, **A2 (residual adaptation)** should outperform **A1 (estimator-only adaptation)**.

### H3 — Adaptation mislocalization (provisional term)

Wrong-locus adaptation may still recover reward or trajectory success, but with measurable costs:

- more online samples
- larger parameter changes
- larger residual / control effort
- worse robustness across severity
- worse transfer under an additional shift

“Adaptation Mislocalization” is a **provisional study label**, not an established literature term.

---

## 3. Laboratory

- Environment: Phase-0 research wrapper around the historical corridor laboratory (`LibraryCorridorEnv` defaults preserved for controller/reward).
- Historical Webots / visual SAC stack is **out of scope**.
- Scientific correction relative to historical env (explicitly declared):
  - IMU bias corrupts **observations / estimates**, not true heading dynamics.
  - Motor asymmetry corrupts **true kinematics / speeds**, not sensor bias parameters.
  - Controllers act on **estimated** heading, not privileged true heading (except optional oracle A1-oracle diagnostic).

---

## 4. Exact mismatch definitions

### 4.1 Observation: IMU heading bias

True heading evolves only from control and kinematics (no bias-on-dynamics term).

Measured IMU-integrated heading:

\[
\hat{\psi}^{\mathrm{imu}}_{t} = \hat{\psi}^{\mathrm{imu}}_{t-1} + (\dot{\psi}^{\mathrm{true}}_{t} + b_{\mathrm{imu}})\Delta t + \epsilon_t
\]

with small sensor noise \(\epsilon_t \sim \mathcal{N}(0, \sigma_{\mathrm{imu}}^2)\).

**Severities (deg/s):**

| Label | \(b_{\mathrm{imu}}\) |
|-------|----------------------|
| 0 | 0.0 |
| small | 0.5 |
| medium | 1.5 |
| large | 3.0 |

Secondary observation mismatch (optional, not required for kill decision): encoder scale — **deferred unless IMU path is blocked**. Primary remains IMU bias.

### 4.2 Dynamics: motor gain asymmetry

\[
g_L = 1 + \delta,\qquad g_R = 1 - \delta
\]

**Severities:**

| Label | \(\delta\) |
|-------|------------|
| 0 | 0.0 |
| small | 0.05 |
| medium | 0.15 |
| large | 0.30 |

### 4.3 Explicitly excluded in Phase-0

Battery degradation, friction changes, delay, dropout, vision noise, multi-mismatch compounds (except the preregistered transfer probe), Webots physics.

---

## 5. Adaptation algorithms

### A0 — No adaptation

- Frozen base P controller (`kp=1.5`, clip ±30°).
- Residual action fixed at 0 (or frozen pretrained residual with **no updates**; primary Phase-0 A0 uses residual=0 to isolate locus effects).
- Sensor calibration frozen at nominal (bias estimate = 0; fusion weight fixed).

### A1 — Estimator adaptation (realistic)

Adaptable parameters only:

- \(\hat{b}_{\mathrm{imu}}\) IMU bias estimate
- optional scalar fusion weight \(w \in [0,1]\) between encoder-derived heading rate and IMU rate

**Method:** recursive / online least-squares bias estimate comparing IMU-integrated heading increment to encoder-derived yaw-rate proxy from left–right wheel distance difference. Fusion:

\[
\hat{\psi}_t = \hat{\psi}_{t-1} + \big(w\,(\omega_{\mathrm{imu}}-\hat{b}) + (1-w)\,\omega_{\mathrm{enc}}\big)\Delta t
\]

with \(w\) adapted by a simple gradient / RLS step that reduces encoder–IMU residual (no neural estimator).

**Forbidden:** motor gains, base `kp`, residual network weights.

**A1-oracle (diagnostic only):** may use true heading to estimate bias. Reported separately; never used for GO/STOP primary decision.

### A2 — Residual policy adaptation

- Freeze sensor calibration and fusion (`\hat{b}=0`, fixed \(w\)).
- Freeze base P controller.
- Allow only residual SAC actor/critic to update online (reuse `LibrarySACAgent` machinery).
- Residual remains bounded: \(a \in [-1,1]\) → ±3° (historical hard bound).

**Forbidden:** changing IMU bias estimate, fusion weight, motor model, base gains.

---

## 6. Budgets (comparable experience)

| Quantity | Value |
|----------|-------|
| Online interaction budget per (mismatch, locus, severity, seed) | **2000** env steps |
| Episode horizon | 200 steps (historical) |
| Eval frequency during adaptation | every **200** steps |
| Eval episodes per checkpoint | **4** fixed IC replicas |
| Observation access | estimated / sensor-derived features only (same 5-dim schema) |
| A1 vs A2 information | same raw encoder, IMU, US channels; no privileged state for primary arms |
| Seeds (debug) | `{0, 1, 2}` |
| Seeds (decision) | `{0, 1, 2, 3, 4}` (≥5) |

Nominal reference: mismatch severity 0, A0, same seeds.

Shifted unadapted reference: same mismatch, A0.

---

## 7. Metrics (primary for decision)

Do **not** optimize only cumulative RL reward. Collect:

1. Episode success (segment complete without hard fail)
2. Mean absolute true heading error (deg)
3. Final absolute true heading error (deg)
4. Mean absolute lateral error (cm)
5. Cumulative control effort \(\sum |u_{\mathrm{base}}+u_{\mathrm{res}}|\)
6. Mean / RMS residual action magnitude
7. Oscillation count (sign changes of residual or total correction)
8. Online interaction count
9. Samples-to-recovery: steps until Performance Recovery Ratio ≥ 0.8 (if never, censored at budget)
10. Adaptation parameter magnitude (A1: \(|\hat{b}|\), \(|\Delta w|\); A2: mean \|Δθ\| proxy via residual RMS growth / weight L2 delta)
11. Wall-clock adaptation seconds (secondary)

### Performance Recovery Ratio (PRR)

Let \(P\) be a performance score (higher better). Default Phase-0 score:

\[
P = -(\text{mean\_abs\_heading} + 0.1\cdot\text{mean\_abs\_lateral})
\]

Then

\[
\mathrm{PRR} = \frac{P_{\mathrm{adapted}} - P_{\mathrm{A0,shifted}}}{P_{\mathrm{nominal}} - P_{\mathrm{A0,shifted}} + \varepsilon}
\]

with \(\varepsilon=10^{-6}\). Clip reporting to a finite window for plots but store raw values.

### Transfer probe

Adapt at **medium** severity; evaluate without further adaptation at **small** and **large** of the **same** mismatch family. Wrong-locus compensation is expected to transfer worse (H3).

---

## 8. Experiment matrix

See `docs/PHASE0_EXPERIMENT_MATRIX.md`.

Cells: 2 mismatch families × {0,small,medium,large} × {A0,A1,A2} × seeds.  
Plus transfer evals from medium-trained adapters.

---

## 9. Statistical treatment

- Report every seed (no hiding failures).
- Report mean ± std (or 95% CI if needed) across decision seeds.
- Prefer paired comparisons within seed across loci.
- No significance fishing; Phase-0 cares about **direction and stability**, not \(p<0.05\).

---

## 10. Kill criteria (decision rules)

### GO

Strong interaction evidence:

- Under IMU bias: A1 clearly more efficient/robust than A2 (and A0)
- Under motor asymmetry: A2 clearly more efficient/robust than A1 (and A0)

“Clearly” means consistent direction on ≥2 primary metrics (e.g., PRR and residual effort / samples-to-recovery) across ≥4/5 seeds at medium and large severities.

### CONDITIONAL GO

Both adaptive loci recover performance under both mismatches, but wrong-locus shows measurable costs (effort, samples, transfer, parameter movement) supporting H3.

### STOP / RETHINK

Any of:

- one method dominates every mismatch
- locus does not matter
- effects unstable across seeds
- effects depend entirely on arbitrary reward shaping (i.e., tracking metrics disagree with reward)
- simulator cannot distinguish mechanisms (unit tests fail separation)

Do not scale experiments under STOP.

---

## 11. Engineering invariants (automated tests)

1. IMU bias changes observations / estimates, not true dynamics parameters.
2. Motor asymmetry changes dynamics, not IMU observation bias.
3. A1 cannot modify controller gains or residual weights.
4. A2 cannot modify sensor calibration parameters.
5. Adaptation parameters are logged over time.
6. Experiments are config-driven and reproducible from one command.

---

## 12. Deliverables

- `docs/ADAPTATION_LOCUS_SYSTEM_AUDIT.md`
- `docs/PHASE0_PREREGISTRATION.md` (this file; committed before final runs)
- `docs/PHASE0_EXPERIMENT_MATRIX.md`
- `docs/PHASE0_RESULTS.md` (after runs)
- `results/adaptation_locus_phase0/`
- research package + unit tests + run script

---

## 13. Preregistration commit note

This file must be committed **before** final decision runs are executed and before `PHASE0_RESULTS.md` is finalized. Smoke tests may precede that commit; decision runs must not.
