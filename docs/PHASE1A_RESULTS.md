# Phase-1A Results — Webots Adaptation Locus Replication

**Decision completed:** 2026-09-05  
**Branch:** `research/adaptation-locus-phase1a-webots`  
**Frozen results commit:** `349e269`  
**Preregistration commit:** `8c11a37`  
**Implementation commit:** `6ac8d7b`  
**Parent Phase-0 frozen results:** `273a4c4`  
**Plant:** Webots-faithful (Amendment W-1) — inherits NUS `DifferentialDriveKinematics`, residual-on-ω path, and Webots proto/`myconfig` parameters. Live Webots ODE binary was unavailable in this environment.

**Command:**

```bash
python -m research.adaptation_locus.run_phase1a_webots --stage decision --config research/adaptation_locus/configs/phase1a_webots.json --output results/adaptation_locus_phase1a_webots/decision
```

---

## 1. Verdict

**STRONG REPLICATION**

Automated rule returned STRONG REPLICATION with `h1_supported=True` and `h2_supported=True` on 5 decision seeds at medium severity.

### Medium-severity mean PRR

| Mismatch | A0 | A1 | A2 |
|----------|---:|---:|---:|
| IMU bias | 0.000 | **+0.991** | **−0.089** |
| Motor asymmetry | 0.000 | −0.014 | **+0.912** |

---

## 2. Cross-environment comparison (Phase-0 vs Phase-1A)

| Finding | Phase-0 | Webots-faithful | Replicated? |
| ------- | ------: | --------------: | ----------- |
| IMU: A1 > A2 | A1 PRR +0.89 vs A2 −3.17 | A1 +0.99 vs A2 −0.09 | **Yes** (ordering) |
| Motor: A2 > A1 | A2 +0.21 vs A1 ≈0 | A2 +0.91 vs A1 ≈0 | **Yes** (ordering) |
| Sample-efficiency ordering | A1 fast on IMU; A2 needed on motor | Same qualitative pattern | **Yes** |
| Wrong-locus IMU residual | Harmful (large negative PRR, effort↑) | Mildly harmful (PRR −0.09, residual>0) | **Partial** (direction yes; magnitude smaller) |
| Wrong-locus motor estimator | ≈0 recovery | ≈0 / slightly negative | **Yes** |
| Control-effort ordering (IMU) | A1 ≪ A0 ≪ A2 | A1 ≪ A0 ≈ A2 | **Yes** for A1 advantage |
| Severity-response shape | Interaction at medium/large | Interaction present at medium; severity sweep logged | **Yes** (qualitative) |

Effect sizes differ (expected). Replication criterion is stable **ordering / interaction**, not identical numbers.

---

## 3. Per-seed medium table (decision)

### IMU bias

| Seed | A0 head | A1 head | A2 head | A1 PRR | A2 PRR | A0 succ | A1 succ | A2 succ |
|-----:|--------:|--------:|--------:|-------:|-------:|--------:|--------:|--------:|
| 0 | 15.09 | 0.27 | 16.36 | +0.992 | −0.085 | 0 | 1 | 0 |
| 1 | 15.07 | 0.41 | 16.38 | +0.983 | −0.088 | 0 | 1 | 0 |
| 2 | 15.04 | 0.20 | 16.40 | +0.997 | −0.091 | 0 | 1 | 0 |
| 3 | 15.04 | 0.34 | 16.36 | +0.987 | −0.088 | 0 | 1 | 0 |
| 4 | 15.02 | 0.22 | 16.39 | +0.996 | −0.092 | 0 | 1 | 0 |

### Motor asymmetry

| Seed | A0 head | A1 head | A2 head | A1 PRR | A2 PRR | A2 residual |
|-----:|--------:|--------:|--------:|-------:|-------:|------------:|
| 0 | 1.98 | 1.82 | 0.26 | +0.081 | +0.944 | 0.086 |
| 1 | 2.13 | 2.36 | 0.45 | −0.113 | +0.852 | 0.084 |
| 2 | 2.00 | 2.14 | 0.29 | −0.074 | +0.927 | 0.086 |
| 3 | 1.98 | 1.76 | 0.32 | +0.117 | +0.910 | 0.082 |
| 4 | 2.01 | 2.16 | 0.29 | −0.083 | +0.927 | 0.086 |

All motor cells: success_rate = 1.0 at medium (discrimination via heading/PRR).

---

## 4. Adaptation Mislocalization (secondary)

| Case | Observation |
|------|-------------|
| IMU × wrong locus (A2) | Negative PRR; nonzero residual; does not beat A0 |
| Motor × wrong locus (A1) | Near-zero / slightly negative PRR; fails to match A2 |
| Transfer | A2 adapted @ medium motor transfers better to small/large than A1; A1 IMU transfer to other severities is weak (bias estimate severity-specific) |

Provisional **Adaptation Mislocalization** label remains useful for IMU×A2 (harmful residual) and motor×A1 (failure-to-recover). Not claimed as literature terminology.

---

## 5. Methodological amendments (all pre-decision)

See `docs/PHASE1A_AMENDMENTS.md`:

| ID | Summary |
|----|---------|
| W-1 | Webots-faithful plant (live ODE unavailable) |
| C-1 | Classical heading P base controller |
| R-1A | Residual bound frozen at ±0.75 rad/s |
| M-1A | Severity schedule (IMU rad/s; motor δ) |
| A2-1A | Low-dimensional residual (not large SAC) |

---

## 6. Largest differences from Phase-0

1. **Units / plant:** metres, rad/s, 20 Hz, Webots track 0.130 m, residual on ω — not Phase-0 cm/deg corridor with ±10° residual.
2. **IMU mismatch severity impact:** Phase-1A medium IMU breaks A0 success (0.0); Phase-0 A0 often still succeeded on binary success.
3. **Motor A2 effect size larger** here (PRR ~0.91 vs ~0.21) under Webots residual authority.
4. **Wrong-locus IMU residual less catastrophic** than Phase-0 (PRR −0.09 vs −3.17) but still worse than A0/A1.
5. **Infrastructure:** no live Webots ODE in this run (W-1).

---

## 7. Biggest remaining confound

**Amendment W-1:** decision evidence is from a Webots-**parameterized kinematic plant** reusing historical NUS control equations, not the Webots ODE process. This is sufficient for mechanism replication under the inherited stack, but **not** a claim about Webots contact/friction/ODE numerics.

---

## 8. Phase-1B justification

**Conditionally justified** for scientific expansion of mismatch families **only after** either:

1. a live Webots ODE confirmation run of the same matrix, or  
2. explicit acceptance that W-1 plant is the Phase-1A reference.

Do **not** auto-start Phase-1B from this report.

---

## 9. Artifacts

- Plots: `results/adaptation_locus_phase1a_webots/decision/plots/`
- Summary: `results/adaptation_locus_phase1a_webots/decision/summary.json`
- Unit tests: `tests/test_adaptation_locus_phase1a_webots.py` (7 passed)
- Phase-0 results untouched: `results/adaptation_locus_phase0/`
