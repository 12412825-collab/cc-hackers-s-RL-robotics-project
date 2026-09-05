# Phase-0 Results — Adaptation Locus Kill Test

**Decision stage completed:** 2026-09-05  
**Branch:** `research/adaptation-locus-phase0`  
**Preregistration commit:** `974bba0`  
**Implementation commit (pre-decision):** `7d40c61`  
**Protocol amendments:** `docs/PHASE0_PROTOCOL_AMENDMENT.md` (A2-1, M-1, R-1)  
**Raw outputs:** `results/adaptation_locus_phase0/decision/`  
**Command:**

```bash
python -m research.adaptation_locus.run_phase0 --stage decision --config research/adaptation_locus/configs/phase0.json --output results/adaptation_locus_phase0/decision
```

---

## 1. Verdict

**GO**

Automated decision rule (`run_phase0.decide`) returned GO with:

- `h1_supported = True`
- `h2_supported = True`

Medium-severity mean Performance Recovery Ratio (PRR):

| Mismatch | A0 | A1 (estimator) | A2 (residual) |
|----------|---:|---------------:|--------------:|
| IMU bias | 0.000 | **+0.885 ± ~0.22** | **−3.173 ± ~0.28** |
| Motor asymmetry | 0.000 | −0.005 ± ~0.04 | **+0.212 ± ~0.003** |

(Individual seeds reported in §3; ± values are approximate seed std from per-seed rows.)

---

## 2. Strongest empirical findings

1. **Mismatch × locus interaction is present and stable across 5 seeds at medium severity.**
   - Observation mismatch (IMU bias): estimator adaptation (A1) recovers toward nominal; residual adaptation (A2) **worsens** tracking and inflates residual effort.
   - Dynamics mismatch (motor asymmetry): residual adaptation (A2) improves heading / success; estimator-only adaptation (A1) is near zero / slightly harmful.

2. **Wrong-locus residual adaptation under IMU bias is not “free recovery” — it is actively harmful** (mean PRR ≈ −3.2, residual magnitude ≈ 0.42, control effort ≈ 336 vs A0 ≈ 200). This is the cleanest Phase-0 signal for provisional **Adaptation Mislocalization** (study label, not literature term).

3. **Under motor asymmetry, A2 raises success rate from 0.75 → 1.0** at medium severity across all seeds, while A1 does not.

---

## 3. Per-seed medium severity table (decision)

### IMU bias

| Seed | A0 perf | A1 perf | A2 perf | A1 PRR | A2 PRR | A0 effort | A1 effort | A2 effort |
|-----:|--------:|--------:|--------:|-------:|-------:|----------:|----------:|----------:|
| 0 | −1.380 | −1.197 | −1.997 | +0.951 | −3.189 | 206.6 | 124.2 | 343.6 |
| 1 | −1.395 | −1.220 | −1.981 | +0.844 | −2.816 | 203.9 | 65.1 | 338.5 |
| 2 | −1.389 | −1.289 | −1.974 | +0.494 | −2.904 | 199.6 | 155.8 | 334.6 |
| 3 | −1.351 | −1.181 | −1.946 | +1.036 | −3.621 | 193.3 | 70.4 | 328.7 |
| 4 | −1.368 | −1.169 | −1.971 | +1.100 | −3.337 | 197.3 | 81.3 | 333.2 |

All IMU cells: success_rate = 1.0 (discrimination is via heading/effort, not binary success).

### Motor asymmetry

| Seed | A0 perf | A1 perf | A2 perf | A1 PRR | A2 PRR | A0 succ | A1 succ | A2 succ |
|-----:|--------:|--------:|--------:|-------:|-------:|--------:|--------:|--------:|
| 0 | −8.377 | −7.984 | −6.840 | +0.055 | +0.214 | 0.75 | 0.75 | 1.00 |
| 1 | −8.353 | −8.417 | −6.829 | −0.009 | +0.213 | 0.75 | 0.75 | 1.00 |
| 2 | −8.337 | −8.762 | −6.851 | −0.059 | +0.208 | 0.75 | 0.50 | 1.00 |
| 3 | −8.325 | −8.294 | −6.793 | +0.004 | +0.215 | 0.75 | 0.75 | 1.00 |
| 4 | −8.348 | −8.478 | −6.823 | −0.018 | +0.213 | 0.75 | 0.75 | 1.00 |

---

## 4. Hypotheses

| Hypothesis | Decision | Notes |
|------------|----------|-------|
| H1 (IMU → estimator) | **Supported** | A1 ≫ A2 on PRR; A1 also lower control effort than A0/A2 |
| H2 (motor → residual) | **Supported** | A2 ≫ A1 on PRR and success; stable across seeds |
| H3 (mislocalization) | **Partially supported** | Wrong-locus A2 under IMU is harmful (effort↑, performance↓). Wrong-locus A1 under motor fails to recover (PRR≈0) rather than compensating with large residual. Transfer: A2 adapted on medium IMU transfers poorly to small/large vs A1 |

---

## 5. Transfer probe (adapt @ medium → eval @ small/large)

Mean performance score across 5 seeds:

| Family | Locus | Eval small | Eval large |
|--------|-------|-----------:|-----------:|
| IMU | A1 | −1.241 | −1.448 |
| IMU | A2 | −1.568 | −2.631 |
| Motor | A1 | −4.376 | −10.942 |
| Motor | A2 | −2.784 | −10.143 |

Interpretation: correct-locus adapters transfer better within-family for the milder shift; large motor asymmetry remains hard for both (near authority limits).

---

## 6. Plots

Under `results/adaptation_locus_phase0/decision/plots/`:

1. `severity_vs_performance.png`
2. `samples_vs_recovery.png`
3. `severity_vs_control_magnitude.png`
4. `mismatch_x_locus_interaction.png` (**primary interaction visual**)

---

## 7. Confounds / limitations (reviewer view)

1. **Plant closure amendment:** historical `LibraryCorridorEnv` did not integrate yaw from wheel differentials; Phase-0 env does. Motor mismatch meaning changed vs hackathon code.
2. **Residual bound amendment (R-1):** Phase-0 uses ±10° residual authority (historical train hard-code ±3°) so dynamics recovery is physically possible.
3. **Motor severity amendment (M-1):** δ reduced so disturbances sit near base yaw authority (~3 deg/s).
4. **A2 is low-dimensional residual policy, not from-scratch SAC** (amendment A2-1) for budget parity with A1.
5. **IMU mismatch rarely breaks binary success** under fusion with encoders; H1 relies on continuous tracking/effort metrics.
6. **A2 under motor improves tracking but increases cumulative control effort** — recovery is not “effort-free.”
7. Lightweight corridor kinematics ≠ Webots physics; results are Phase-0 mechanism evidence only.

---

## 8. Kill-criteria mapping

| Criterion | Outcome |
|-----------|---------|
| GO interaction pattern | **Met** at medium severity, 5/5 seeds directional for both legs |
| One method dominates all mismatches | **Not observed** (A1 wins IMU, A2 wins motor) |
| Unstable across seeds | **Not for primary interaction**; A1 IMU PRR always > A2; A2 motor PRR always > A1 |
| Reward-only artifact | Unlikely: success/heading/effort agree with PRR direction |
| Simulator cannot distinguish | Unit tests + separated mismatch effects pass |

---

## 9. What should come next (if continuing)

1. **Phase-1 mechanism study:** privileged vs realistic estimator; quantify information used by A1.
2. **Matched-authority residual vs estimator** under identical parameter dimension and update rules.
3. **Webots confirmation** of the same 2×3 matrix (do not expand mismatch families yet).
4. Optional SAC diagnostic backend with longer budget — not for changing Phase-0 verdict.

Do **not** expand mismatch families until this GO is replicated under Webots or a second independent seed batch.
