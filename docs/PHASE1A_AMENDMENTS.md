# Phase-1A Amendments

All amendments are methodological / infrastructure. Hypotheses H1–H3 are unchanged from Phase-0.

## W-1 — Webots-faithful offline plant (2026-09-05, BEFORE decision runs)

**Reason:** Live Webots `controller` API and `webots` binary are unavailable in the Phase-1A execution environment. The historical DonkeyCar+Webots loop also lacks a Gym episode API for batch Adaptation Locus experiments.

**Action:** Implement `WebotsFaithfulPlant` that inherits:

- `DifferentialDriveKinematics` / `VelocityDriveMode` from `parts/differential_drive.py`
- Webots geometry and limits from `myconfig.py` / `FourWheelRobot.proto`
- residual-on-ω scheme A (±`RESIDUAL_ANGULAR_SCALE`)
- 50 ms timestep

**Not done:** tuning mass/friction/gains to match Phase-0 effect sizes.

**Live hook:** `LiveWebotsBackend` remains available for hosts with Webots installed; primary decision runs use the faithful plant under this amendment.

| Item | Old | New |
|------|-----|-----|
| Experiment plant | Live Webots ODE (unavailable) | Webots-faithful plant reusing NUS kinematics/params |

---

## C-1 — Classical heading base controller for replication (BEFORE decision runs)

**Reason:** Historical Webots path has no Phase-0-equivalent heading P-controller (vision KerasPilot ≠ Adaptation Locus base).

**Action:** Add a minimal heading P-controller producing base ω from estimated heading, using Webots max angular velocity scaling. Gains chosen for nominal stability feasibility check only (not effect-size matching).

| Item | Old | New |
|------|-----|-----|
| Base controller on Webots path | absent / vision optional | `omega_base = clip(-kp * heading_est_rad, ±ω_max)` with `kp=2.0` |

---

## R-1A — Residual bound freeze (BEFORE decision runs)

**Reason:** Phase-0 amended residual authority for recoverability. Webots natural residual is ±0.75 rad/s.

**Action:** Freeze Phase-1A residual as:

\[
\omega_{\mathrm{res}} = a \cdot 0.75,\quad a\in[-1,1]
\]

i.e. historical `RESIDUAL_ANGULAR_SCALE=0.75`. No post-hoc bound changes after decision runs.

| Item | Old (conceptual) | Frozen |
|------|------------------|--------|
| Residual physical bound | Phase-0 ±10° heading | ±0.75 rad/s on ω |

---

## M-1A — Severity schedule (BEFORE decision runs)

Chosen after inspecting Webots authority (ω_max=1.50 rad/s, residual ±0.75, track 0.130 m, cruise 0.12 m/s). Feasibility: motor δ such that uncommanded yaw is meaningful but not instantly saturated.

| Label | IMU bias (rad/s) | Motor δ |
|-------|-----------------:|--------:|
| 0 | 0.0 | 0.0 |
| small | 0.05 | 0.02 |
| medium | 0.15 | 0.04 |
| large | 0.30 | 0.06 |

IMU bias applied to gyro yaw-rate observation (converted consistently with deg/s channel semantics internally).

---

## A2-1A — Low-dimensional residual (inherited from Phase-0 A2-1)

Primary A2 remains a 2-parameter linear residual policy mapped to ω residual. Vision SAC / large SAC are diagnostics only, not headline.
