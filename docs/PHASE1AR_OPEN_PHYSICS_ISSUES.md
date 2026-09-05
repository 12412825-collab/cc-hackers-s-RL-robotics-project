# Phase-1A-R — Open Physics Issues (frozen, not auto-resolved)

**Branch:** `research/adaptation-locus-phase1ar-live-webots`  
**Recorded:** Step 2 (adapter integration)

---

## OPI-01 — Robot mass: PROTO vs myconfig

| Source | Value |
|--------|------:|
| `simulation/protos/FourWheelRobot.proto` chassis `Physics.mass` | 0.60 kg |
| PROTO wheel masses (4 × 0.05) | 0.20 kg |
| **PROTO total (approx)** | **0.80 kg** |
| `myconfig.ROBOT_MASS` | 0.800 kg |

### Does `myconfig.ROBOT_MASS` enter Webots physics?

**No.** Grep of the runtime path shows `ROBOT_MASS` is not read by `WebotsAdapter`, kinematics, or the live research adapter. Live ODE mass comes from the PROTO `Physics` nodes.

### Decision (Step 2)

- **Live physics source of truth:** PROTO masses.
- **myconfig.ROBOT_MASS:** metadata / documentation only for the Webots path.
- **Action:** freeze; do not auto-reconcile or “pick the more real” value.
- Numerical coincidence (0.60+0.20≈0.80) is noted but not treated as a coupled config.

---

## OPI-02 — Default ContactProperties

Step 1.5 showed default `"default"`/`"default"` contact is sufficient for sanity gates after P-1AR-01.

**Action:** do not tune friction/bounce for digital-twin fidelity in Phase-1A-R.

---

## OPI-03 — Unused PROTO fields

`wheelSeparation` / `wheelbase` PROTO parameters lack `IS` bindings (geometry hard-coded in joint anchors). Cosmetic R2025a warnings only.
