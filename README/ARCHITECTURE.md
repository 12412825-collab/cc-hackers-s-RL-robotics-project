# Architecture — Multi-Modal Residual RL

## 1. SAC (Soft Actor-Critic)

**SAC** is a maximum-entropy, off-policy deep RL algorithm for continuous control.

| Feature | Description |
|---------|-------------|
| Off-Policy | Learns from historical Tub data, no online interaction required |
| Max Entropy | Encourages exploration, auto-balances explore/exploit |
| Twin Q-Network | Dual critics reduce value overestimation bias |
| Auto Temperature | Automatically tunes entropy coefficient alpha |
| Multi-Modal | Image + Sensor fusion as state input |

**Losses:**

- **Critic**: `L_Q = MSE[Q(s,a), r + γ·(min(Q1',Q2')(s',a') − α·log_π(a'|s'))]`
- **Actor**: `L_π = mean[α·log_π(a|s) − min(Q1,Q2)(s,a)]`
- **Alpha**: `L_α = −log_α · (log_π(a|s) + H_target)`

---

## 2. Multi-Modal Fusion Architecture

### Overview

```
┌─────────────────────┐    ┌─────────────────────┐
│   VISUAL STREAM      │    │   SENSOR STREAM      │
│                       │    │                       │
│ Camera (120×160 BGR)  │    │ Encoder (speed,accel) │
│       │               │    │ IMU (6-DOF)           │
│       ▼               │    │ Obstacle (distance)   │
│ ImageProcessor        │    │ Line Track (IR array) │
│  BGR→RGB .copy()      │    │       │               │
│  Resize 224×224       │    │       ▼               │
│  ImageNet Norm        │    │ SensorFusion          │
│       │               │    │  normalize per sensor │
│       ▼               │    │  fill missing → 0     │
│ MobileNetV3-Small     │    │       │               │
│  (Frozen, ImageNet)   │    │  obs vector (9,)     │
│  → GlobalAvgPool      │    │       │               │
│  → Projection Head    │    │       ▼               │
│       │               │    │ SensorEncoder MLP     │
│       ▼               │    │  Linear(12,64)→ReLU   │
│ visual_features (50)  │    │  Linear(64,64)→ReLU   │
│                       │    │  Linear(64,32)         │
│                       │    │       │               │
│                       │    │ sensor_features (32)  │
└──────────┬────────────┘    └──────────┬────────────┘
           │                            │
           └──────────┬─────────────────┘
                      │
                      ▼
           ┌──────────────────┐
           │  torch.cat([v,s]) │
           │  fused (82-dim)   │
           └────────┬─────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   ┌─────────┐            ┌──────────┐
   │SACActor │            │SACCritic │
   │ μ,σ→π   │            │ Q1,Q2    │
   └────┬────┘            └──────────┘
        │
   residual ∈ [-1,1] × scale
```

### Image-Only Mode (Backward Compatible)

When `USE_MULTI_MODAL=False` or `sensor_dim=0`:

```
Camera → ImageProcessor → Backbone → visual_features (50)
→ SACActor → residual
```

No code changes needed — the original training and inference pipeline works unchanged.

### Sensor Encoder Design

```python
SensorEncoder(
    sensor_dim=9,
    hidden_dim=64,
    sensor_feature_dim=32,
    dropout=0.1
)
# Architecture:
#   Linear(9 → 64) → ReLU → Dropout(0.1)
#   → Linear(64 → 64) → ReLU → Dropout(0.1)
#   → Linear(64 → 32)
# Output: 32-dim sensor feature vector
```

**Design Rationale:**
- Small MLP (~5K params) — negligible overhead vs backbone (~2.5M)
- Dropout prevents sensor overfitting when some sensors are unavailable
- 32-dim output gives sensors ~40% contribution to fused decision
- Missing sensors → zero fill → encoder learns to handle missing data

---

## 3. Image Processing Pipeline

### Pretrained Backbone Path (default: mobilenet_v3_small)

```
Camera Output (OpenCV BGR, uint8, 120x160, [0,255])
    │
    ▼
BGR → RGB Conversion          # torchvision models expect RGB
    │                          # .copy() prevents negative numpy strides
    ▼
Resize to 224x224             # bilinear interpolation
    │
    ▼
ImageNet Normalization        # mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
    │
    ▼
Pretrained Backbone (FROZEN)  # MobileNetV3-Small / V2 / EfficientNet-B0
    │                          # Early layers frozen → feature extractor
    ▼
Global Average Pooling
    │
    ▼
Projection Head               # Linear(backbone_dim, 128) → ReLU → Dropout → Linear(128, 50)
    │
    ▼
Visual Feature Vector (50-dim)
    │
    ├───→ [optional] fuse with sensor_features (32-dim) → fused (82-dim)
    │
    └───→ SACActor  → residual action in [-1, 1]
    └───→ SACCritic → Q-value
```

### DonkeyCNN Path (baseline, no pretrain)

```
Camera Output (uint8, 120x160, [0,255])
    │
    ▼
Normalize to [0, 1]           # astype(float32) / 255.0
    │
    ▼
DonkeyCNN (5 conv layers)     # Trained from scratch
    │                          # Conv(24,5,2)→Conv(32,5,2)→Conv(64,5,2)
    │                          # →Conv(64,3,1)→Conv(64,3,1)
    │                          # →Flatten→Dense(100)→Dense(50)
    ▼
Feature Vector (50-dim)
```

---

## 4. Backbone Comparison

| Backbone | Params | Input Size | Output Dim | Pretrained | Use Case |
|----------|--------|-----------|------------|------------|----------|
| `donkey_cnn` | ~70K | 120×160 | 50 | None | Baseline ablation |
| `mobilenet_v3_small` | ~2.5M | 224×224 | 576→50 | ImageNet | **Default** — Embedded/RPi |
| `mobilenet_v2` | ~3.5M | 224×224 | 1280→50 | ImageNet | Better accuracy |
| `efficientnet_b0` | ~5.3M | 224×224 | 1280→50 | ImageNet | Best accuracy |

### Transfer Learning Strategy

| Phase | Backbone | Projection Head | SensorEncoder | SAC Heads | LR |
|-------|----------|----------------|---------------|-----------|-----|
| Phase 1 (warm-start) | **Frozen** | Trainable | Trainable | Trainable | 3e-4 |
| Phase 2 (fine-tune, optional) | Last N blocks unfrozen | Trainable | Trainable | Trainable | 1e-4 |

### Model Sizes (Multi-Modal)

| Component | Params | Trainable (Phase 1) |
|-----------|--------|---------------------|
| MobileNetV3-Small (frozen) | ~2.5M | 0 |
| Projection Head | ~76K | ~76K |
| SensorEncoder | ~6K | ~6K |
| SACActor | ~67K | ~67K |
| SACCritic | ~134K | ~134K |
| **Total** | **~2.78M** | **~283K (10%)** |

---

## 5. Vehicle Pipeline Integration

```
Camera → cam/image_array
    │
    ├──→ SensorFusion → sensor/observation
    │       │
    │       ├── Encoder (enc/speed)
    │       ├── IMU (imu/acl_*, imu/gyr_*)
    │       ├── Obstacle (obs/distance)
    │       └── Line Track (line/raw_values)
    │
    ├──→ KerasLinear (Base Pilot)  → pilot/angle, pilot/throttle
    │
    ├──→ ResidualPilot (SAC)       → residual/steering
    │       │
    │       ├── ImageProcessor: BGR→RGB, Resize, ImageNet Norm
    │       ├── PretrainedBackbone: frozen MobileNetV3 → vis_feat (50)
    │       ├── SensorEncoder: obs (12) → sen_feat (32) [optional]
    │       ├── Fusion: cat(vis_feat, sen_feat) → fused (82)
    │       └── SACActor: deterministic_action()
    │
    └──→ ResidualDriveMode
            steering = pilot/angle + residual/steering
```

### Memory Channels

| Channel | Source | Type | Dimension |
|---------|--------|------|-----------|
| `cam/image_array` | Camera | uint8 ndarray | (120, 160, 3) |
| `cam/image_array_trans` | ImageTransformations | float32 (optional) | varies |
| `pilot/angle` | BasePilot | float | scalar |
| `pilot/throttle` | BasePilot | float | scalar |
| `residual/steering` | ResidualPilot | float | in [-scale, +scale] |
| `sensor/observation` | SensorFusion | float32 ndarray | (9,) |
| `enc/speed` | Encoder | float | m/s |
| `imu/acl_x..z` | IMU | float | g |
| `imu/gyr_x..z` | IMU | float | deg/s |
| `obs/distance` | Obstacle | float | cm |
| `line/raw_values` | Line Tracking | list | [0/1] × 5 |

---

## 6. Training Flow

### Offline Training (Image-Only)

```
Tub Data (images + user/angle)
    │
    ▼
Load image (native 120×160)
    │
    ▼
Base Pilot inference → base_steering
    │
    ▼
residual_target = clip((human_steering − base_steering) / RESIDUAL_SCALE, −1, 1)
    │
    ▼
Store in ReplayBuffer: (img_CHW_float32, residual, 0, img_CHW_float32, False)
    │
    ▼
Training loop:
    Sample batch → ImageProcessor.process_batch() → resize + ImageNet norm
    → Backbone → visual_features (50)
    → SAC Update (critic + actor + alpha)
```

### Offline Training (Multi-Modal)

```
Tub Data (images + sensor data + user/angle)
    │
    ▼
Load image (native 120×160) + build sensor observation from tub record
    │
    ▼
Base Pilot inference → base_steering
    │
    ▼
residual_target = clip((human_steering − base_steering) / RESIDUAL_SCALE, −1, 1)
    │
    ▼
Store in ReplayBuffer: (img_CHW, sensor_obs(9,), residual, 0, img_CHW, sensor_obs_next(9,), False)
    │
    ▼
Training loop:
    Sample batch → process images + extract sensor tensors
    → Backbone → visual_features (50)
    → SensorEncoder → sensor_features (32)
    → fuse → fused_features (82)
    → SAC Update (critic + actor + alpha)
```

---

## 7. SensorFusion Class Structure

```
parts/sensors.py
├── SensorConfig       — @dataclass: all sensor parameters
├── EncoderSensor      — speed + acceleration (GPIO/Arduino/Sim)
├── IMUSensor          — 6-DOF accel+gyro (MPU6050/9250/ICM20948)
├── ObstacleSensor     — distance (Ultrasonic/TFMini/IR)
├── LineTrackingSensor — IR array (3–5 sensors)
├── SensorFusion       — collects all → unified observation vector
├── SensorBuffer       — rolling buffer for temporal smoothing
├── RunningNormalizer  — running mean/std for observation normalization
└── create_sensor_fusion() — factory from DonkeyCar config
```

### SensorFusion Part Interface

```python
# In manage.py, wire sensors:
V.add(fusion, inputs=[
    'enc/speed',
    'imu/acl_x', 'imu/acl_y', 'imu/acl_z',
    'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z',
    'obs/distance',
    'line/raw_values',
], outputs=['sensor/observation'])

# Then wire ResidualPilot with sensor input:
V.add(rl_pilot, inputs=['cam/image_array', 'sensor/observation'],
      outputs=['residual/steering'], run_condition='run_pilot')
```

---

## 8. ReplayBuffer — Multi-Modal Storage

```
┌──────────────────────────────────────┐
│         ReplayBuffer                  │
│                                       │
│  push(state, action, reward,          │
│       next_state, done,               │
│       sensor_state=?,                 │
│       sensor_next_state=?)            │
│                                       │
│  Internal: deque(maxlen=capacity)     │
│  Each entry: (img, act, rew, nxt,     │
│               done, sen_s, sen_n)     │
│                                       │
│  has_sensor_data: bool                │
│    True → sample() returns 7 tensors  │
│    False → sample() returns 5 tensors │
└──────────────────────────────────────┘
```

Images stored at native resolution (120×160). Resize + normalization at batch time → buffer is backbone-agnostic. Sensor vectors stored as `(sensor_dim,)` float32.

---

## 9. Checkpoint Format

```python
# Multi-modal checkpoint:
{
    'backbone_type': 'mobilenet_v3_small',
    'encoder': state_dict,          # visual backbone
    'sensor_encoder': state_dict,   # ★ NEW — None if image-only
    'actor': state_dict,
    'critic': state_dict,
    'critic_target': state_dict,
    'log_alpha': tensor,
    'sensor_dim': 12,               # ★ NEW
    'sensor_feature_dim': 32,       # ★ NEW
    'visual_feature_dim': 50,
    'fused_feature_dim': 82,        # ★ NEW
    'use_multi_modal': True,        # ★ NEW
}
```

**Backward compatibility:** Image-only checkpoints load into multi-modal models (sensor encoder uses random init with warning).

---

## 10. Dependencies

```
donkeycar >= 5.0.0
torch >= 1.9.0
torchvision >= 0.10.0     # pretrained backbones
numpy
# Optional (hardware):
# smbus2, RPi.GPIO, pyserial
```

---

## 11. References

- SAC: [Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL](https://arxiv.org/abs/1801.01290)
- MobileNetV3: [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
- EfficientNet: [Rethinking Model Scaling](https://arxiv.org/abs/1905.11946)
- DonkeyCar: https://docs.donkeycar.com/
