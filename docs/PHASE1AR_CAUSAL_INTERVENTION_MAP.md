# Phase-1A-R Causal Intervention Map

**Step:** 3 — mismatch injection & causal separation  
**Plant:** live Webots ODE (`live_webots_ode`)  
**Adaptation:** OFF (A0 only)

---

## Diagram

```text
Webots ODE state x_t
      │
      ├── sensors h(x_t)          [Gyro rad/s, encoders, …]
      │       │
      │       └── [M1 IMU BIAS]   observation path ONLY
      │               ↓           observed = raw + b   (b in rad/s)
      │          observation y_t
      │               ↓
      │            estimator φ    (FROZEN in Step 3)
      │               ↓
      │          base controller  (heading-P)
      │               ↓
      │         residual hook = 0
      │               ↓
      │        requested motors (ω_L, ω_R)
      │               │
      │               └── [M2 MOTOR GAIN]  actuator path ONLY
      │                         ↓          applied = (gL·ω_L, gR·ω_R)
      └──────────────── applied motors
                                ↓
                           Webots ODE
```

---

## Intervention vs downstream consequence

| Mismatch | Direct intervention location | Expected downstream (NOT contamination) |
|----------|------------------------------|-------------------------------------------|
| **M1** IMU bias | Research layer after raw Gyro read | Controller reacts → physical trajectory / later sensor values change |
| **M2** motor δ | Immediately before `Motor.setVelocity` | ODE motion changes → later Gyro/encoders change |

**Forbidden contamination**

- M1 must not change motor gains, residual, φ, PROTO physics.
- M2 must not change IMU bias parameter, raw IMU processing, φ, sensor model.

---

## Units

- Internals: **radians** / **rad/s** only.
- Historical stack exposes Gyro (rate), not InertialUnit heading.
- M1 bias is therefore on **yaw-rate** [rad/s]; heading estimate integrates the biased rate.
