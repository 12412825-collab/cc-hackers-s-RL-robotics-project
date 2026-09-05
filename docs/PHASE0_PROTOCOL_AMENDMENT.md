# Phase-0 Protocol Amendment (non-hypothesis)

**Date:** 2026-09-05  
**Parent:** `docs/PHASE0_PREREGISTRATION.md` @ commit `974bba0`

## Amendment A2-1 — Residual adaptation parameterization

Hypotheses H1–H3 and kill criteria are unchanged.

For **comparable online budgets** with A1 (low-dimensional recursive estimation), Phase-0 primary **A2** is implemented as a **2-parameter linear residual policy**

\[
a = \tanh(\theta_0 + \theta_1 \tilde{\psi})
\]

updated by SGD toward canceling the normalized estimated heading feature, with residual still bounded in \([-1,1]\) (maps to ±3°).

Full Residual SAC (`SACResidualAdapter`) remains available as an optional diagnostic backend under `research/adaptation_locus/residual_adapt.py`, but is **not** used for the primary GO/STOP decision because from-scratch SAC cannot fairly share a 2000-step budget with recursive bias estimation.

Sensor calibration and base controller remain frozen under A2.

## Amendment M-1 — Motor asymmetry severities

Phase-0 uses δ ∈ {0.015, 0.03, 0.045} after plant closure. See preregistration §4.2.

## Amendment R-1 — Residual bound

Phase-0 `max_residual_deg = 10.0` (historical training env used 3.0) so residual adaptation has non-negligible yaw authority under dynamics mismatch. Still hard-bounded.
