# DonkeyCar reuse boundary

This file is the short architectural reference for future work. Read it before
re-reading the large `manage.py` or `config.py` files.

## Retained and trusted

| DonkeyCar capability | Project use |
|---|---|
| `Vehicle` and Part lifecycle | deterministic ordered pipeline and shutdown |
| named memory channels | stable interface between simulation, policy, UI and recording |
| Web/joystick controllers | manual driving and data collection |
| `TubWriter` and Tub tools | image, action and telemetry recording |
| base Pilot loading/inference | behavioural-cloning baseline |
| run conditions | user/automatic mode switching |
| existing real drivetrain branches | transitional real-robot compatibility |

These are framework-level facilities and do not assume Ackermann steering.
DonkeyCar's official design explicitly supports custom Parts and differential
drivetrains, so `WebotsAdapter` follows the intended extension mechanism.

## Replaced or isolated

| Original capability | Replacement |
|---|---|
| DonkeySim/Ackermann dynamics | Webots 4WD skid-steer model |
| steering/throttle as robot command | physical `v` (m/s) and `omega` (rad/s) |
| one steering residual | angular-velocity residual only |
| single drive motor assumptions | four motors grouped into left/right sides |
| simulator-specific sensor names | DonkeyCar-compatible memory channels |

## Stable boundary

Policy and robot adapters exchange only:

```text
command:     linear/velocity, angular/velocity
observation: cam/image_array, enc/*, imu/*, obs/distance
truth/log:   pos/*
```

Code above this boundary should not import the Webots `controller` package.
Code below it should not know about Keras or SAC internals.

## Known exclusions

- DonkeySim is not used for dynamics validation.
- Old residual checkpoints are not compatible with rad/s residual semantics.
- Current offline SAC data is not true closed-loop reinforcement learning.
- Webots ground truth is for reward/evaluation, not policy input.

