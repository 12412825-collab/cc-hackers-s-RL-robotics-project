# Library Robot Minimal Residual SAC Architecture & Integration

## 1. Overview

The `library_residual` package implements a lightweight, low-compute **Residual Soft Actor-Critic (SAC)** heading controller designed to plug into the fixed-grid borrowing robot (`Library_robot`).

Unlike standard end-to-end vision-based RL models (e.g. MobileNet/DonkeyCar CNNs), this residual controller:
1. **Inputs zero camera frames**: Uses only 5 normalized float32 sensor telemetry features.
2. **Never controls high-level logic**: Does not plan routes, select target boxes, execute turns, or handle obstacle safety stops.
3. **Outputs a pure residual**: Generates a single bounded heading adjustment (`±10 PWM` max) that adds to the rule-based encoder+IMU PID controller on the Arduino Mega.
4. **Fails safe**: Any missing model, stale telemetry, non-linear turn phase, or invalid reading immediately falls back to a `0 PWM` residual correction.

---

## 2. Feature Schema (`library-observation-v1`)

The model expects a float32 vector of exactly 5 features, normalized to `[-1, 1]`:

| Index | Feature Name | Description | Raw Range | Normalization Range |
|-------|--------------|-------------|-----------|--------------------|
| 0 | `motion_direction` | Current motion vector (+1.0 FORWARD, -1.0 BACKWARD) | `[-1.0, 1.0]` | `[-1.0, 1.0]` |
| 1 | `segment_progress` | Ratio of completed distance over segment target | `[0.0, 1.0]` | `[0.0, 1.0]` |
| 2 | `fused_heading_error` | Wrapped target heading minus MPU6500+encoder fused heading | `[-180.0, 180.0]` deg | `[-45.0, 45.0]` deg |
| 3 | `left_right_encoder_error` | Cumulative left encoder distance minus right encoder distance | `[-50.0, 50.0]` cm | `[-10.0, 10.0]` cm |
| 4 | `front_ultrasonic_distance` | Front HC-SR04 ultrasonic distance | `[0.0, 400.0]` cm | `[0.0, 400.0]` cm |

---

## 3. Network Architecture

- **Actor**: 2-layer MLP (`5 -> 64 (ReLU) -> 64 (ReLU) -> mean (1) / log_std (1)`). Tanh-squashed output in `[-1.0, 1.0]`.
- **Critic**: Twin Q-networks (`(5 + 1) -> 64 (ReLU) -> 64 (ReLU) -> Q (1)`).
- **Parameters**: ~9,000 parameters total.
- **Compute footprint**: < 1.0 ms CPU inference on Raspberry Pi 5.

---

## 4. Safety & Operating Modes

`SafeResidualPolicy` provides three operational modes:

1. **`disabled`** (default):
   - Inference is disabled. Returns `residual_pwm = 0` and `apply_to_motor = False`.
2. **`shadow`**:
   - Computes model recommendations, checks latency and deadlines, logs evaluation telemetry, but sets `apply_to_motor = False`.
3. **`active`**:
   - Executes model inference. Authorizes sending `SET_RL_CORRECTION` to the Arduino Mega ONLY when:
     - The robot is in a linear motion phase (`FORWARD` or `BACKWARD`).
     - Front ultrasonic data is valid and finite.
     - Telemetry age is under 250 ms.
     - Model latency is under 50 ms.

---

## 5. Training & Export

### 5.1 Training in Simulation
```bash
python train_library_sac.py --total-steps 50000 --seed 42 --output models/library_sac/
```

### 5.2 Exporting Deployment Bundle
```bash
python tools/export_library_sac.py --input models/library_sac/checkpoint.pth --output models/library_sac/
```

### 5.3 Benchmarking Inference Speed
```bash
python tools/benchmark_library_sac.py --model models/library_sac/
```

The exported bundle contains:
- `actor.ts`: TorchScript deterministic actor.
- `checkpoint.pth`: Full training checkpoint (actor, critics, optimizers).
- `manifest.json`: Versioning and schema metadata.
- `normalization.json`: Per-feature normalization parameters.
