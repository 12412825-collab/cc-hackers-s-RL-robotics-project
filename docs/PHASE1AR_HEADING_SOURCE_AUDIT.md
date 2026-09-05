# Phase-1A-R Step 3.5 — Heading source audit (pre-implementation)

**Date:** 2026-09-06  
**Branch:** `research/adaptation-locus-phase1ar-live-webots`

## Available absolute / rate orientation signals

| Signal | Present in historical PROTO/stack? | Controller-facing? | Notes |
|--------|------------------------------------|--------------------|-------|
| Webots `Gyro` | **Yes** (`name "gyro"`) | Yes (via adapter) | rad/s; Y component = yaw rate under NUE |
| Webots `Accelerometer` | Yes | Yes | not heading |
| Webots `InertialUnit` | **No** | — | absent from `FourWheelRobot.proto` |
| Compass | **No** | — | absent |
| Encoder-derived yaw rate | Yes (wheel PositionSensors) | Yes | `(r/B)·(ω_R−ω_L)` |
| Supervisor rotation / true yaw | Yes | **Eval only** | privileged; forbidden for controller heading |

## Decision

**Option B — integrated historical Gyro** as primary raw heading source.

Reason: no historical InertialUnit/Compass; must not use Supervisor GT for controller-facing heading.

Initial condition at episode reset: known spawn heading from experiment configuration (`0.0` rad matching world `rotation 0 1 0 0`). This is config, not runtime Supervisor readout.

## Step-3 legacy

`gyro_rate_bias` (rad/s) remains validated under Step 3 as secondary / Phase-1B candidate. Not primary Phase-1A-R M1.
