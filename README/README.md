# Digital Twin Based Multi-Sensor Autonomous Mobile Robot

> **Deep Learning Perception + Residual Reinforcement Learning Decision-Making**

---

## 项目定位 (Project Positioning)

本项目是一个**基于数字孪生的多传感器智能移动机器人**系统。核心思路是：

- **DL 负责感知**（Perception）：CNN 从视觉+传感器数据中提取环境特征
- **RL 负责决策**（Decision）：SAC 强化学习基于感知特征输出控制修正
- **Arduino 负责执行**（Execution）：接收高层决策，执行底层电机/PWM 控制
- **数字孪生驱动开发**（Digital Twin）：仿真→训练→验证→Sim-to-Real→部署

| 层级 | 功能 | 技术栈 | 运行平台 |
|------|------|--------|----------|
| 感知层 (Perception) | 视觉特征提取 + 多传感器融合 | MobileNetV3 + SensorFusion | Raspberry Pi |
| 决策层 (Decision) | 残差强化学习控制策略 | SAC (Soft Actor-Critic) | Raspberry Pi |
| 执行层 (Execution) | 电机控制、PWM、低层驱动 | Arduino Mega 2560 | Arduino |
| 仿真层 (Simulation) | 数字孪生虚拟环境 | DonkeySim / Webots | PC / Cloud |

---

## 系统架构 (System Architecture)

```
┌──────────────────────────────────────────────────────────────────┐
│                     DIGITAL TWIN (PC/Cloud)                       │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐  │
│  │ Simulation│ → │ Training │ → │Validation│ → │Sim-to-Real   │  │
│  │ (DonkeySim│   │ (SAC     │   │ (Metrics │   │(Domain       │  │
│  │  Webots)  │   │ Offline) │   │  Monitor)│   │ Adaptation)  │  │
│  └──────────┘   └──────────┘   └──────────┘   └──────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                  RASPBERRY PI (On-Board Computer)                  │
│                                                                   │
│  ┌──────────┐   ┌─────────────────┐   ┌──────────────────────┐   │
│  │  Camera  │   │  SensorFusion   │   │   Residual RL (SAC)  │   │
│  │ 120×160  │   │                 │   │                      │   │
│  │  BGR     │   │ ┌─────────────┐ │   │  ┌────────────────┐  │   │
│  │          │   │ │ Encoder      │ │   │  │ ImageProcessor │  │   │
│  └────┬─────┘   │ │ (Speed/Accel)│ │   │  │ (BGR→RGB,      │  │   │
│       │         │ ├─────────────┤ │   │  │  Resize,Norm)  │  │   │
│       │         │ │ IMU          │ │   │  └───────┬────────┘  │   │
│       │         │ │ (6-DOF)      │─┼───┼─→│       ▼          │   │
│       │         │ ├─────────────┤ │   │  ┌───────────────┐   │   │
│       │         │ │ Obstacle     │ │   │  │ MobileNetV3   │   │   │
│       │         │ │ (Distance)   │ │   │  │ Backbone      │   │   │
│       │         │ ├─────────────┤ │   │  │ (Frozen,      │   │   │
│       │         │ │ Line Track   │ │   │  │  ImageNet)    │   │   │
│       │         │ │ (IR Array)   │ │   │  └───────┬───────┘   │   │
│       │         │ └─────────────┘ │   │          │            │   │
│       │         │       │         │   │          ▼            │   │
│       │         │  observation    │   │  visual_features(50)  │   │
│       │         │  vector(12-dim) │   │          │            │   │
│       │         │       │         │   │          │            │   │
│       │         │       ▼         │   │  ┌───────▼────────┐   │   │
│       │         │  SensorEncoder  │   │  │ SensorEncoder  │   │   │
│       │         │  → sen_feat(32) │   │  │ → sen_feat(32) │   │   │
│       │         └───────┬─────────┘   │  └───┬───────┬────┘   │   │
│       │                 │             │      │       │        │   │
│       │                 │             │  fused_features(82)   │   │
│       │                 │             │      │                │   │
│       │                 │             │  ┌───▼────▼───┐       │   │
│       │                 │             │  │ SACActor   │       │   │
│       └─────────────────┼─────────────┼──┤ SACCritic  │       │   │
│                         │             │  └─────┬──────┘       │   │
│                         │             │        │              │   │
│                         │             │  residual/steering    │   │
│  ┌──────────┐           │             │        │              │   │
│  │KerasLinear│           │             │        │              │   │
│  │Base Pilot │───────────┼─────────────┼────────┤              │   │
│  │(BC model) │           │             │        │              │   │
│  └─────┬─────┘           │             │        │              │   │
│        │                 │             │        │              │   │
│   pilot/angle            │             │   final_steering =    │   │
│        │                 │             │   base + residual     │   │
│        └─────────────────┴─────────────┴────────┤              │   │
│                                                 ▼              │   │
│                                    ┌────────────────────┐     │   │
│                                    │ Action Smoothing   │     │   │
│                                    │ Safety Constraints │     │   │
│                                    └────────┬───────────┘     │   │
└─────────────────────────────────────────────┼──────────────────┘
                                              │ UART/I²C
                                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                  ARDUINO MEGA 2560 (Execution Layer)              │
│                                                                   │
│  ┌──────────────────┐  ┌─────────────┐  ┌────────────────────┐  │
│  │ Motor PWM Control │  │Servo Control│  │ Sensor Preprocess │  │
│  │ (DRV8833/L298N)  │  │ (MG996R)    │  │ (Encoder ticks→m/s)│  │
│  └──────────────────┘  └─────────────┘  └────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 数据流 (Data Flow)

```
┌─────────────────────────────── SENSING ───────────────────────────────┐
│                                                                        │
│  Camera ──→ cam/image_array (120×160 BGR uint8)                       │
│  Encoder ─→ enc/speed (m/s)                                           │
│  IMU ────→ imu/acl_x|y|z, imu/gyr_x|y|z (g, deg/s)                  │
│  Obstacle → obs/distance (cm)                                         │
│  Line ───→ line/raw_values (0/1 × 5)                                  │
│                                                                        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│   SensorFusion           │    │   ImageProcessor         │
│   (parts/sensors.py)     │    │   (residual_rl.py)       │
│                          │    │                          │
│  • Timestamp align       │    │  • BGR → RGB (.copy())   │
│  • Normalize per sensor  │    │  • Resize → 224×224      │
│  • Fill missing w/ zeros │    │  • ImageNet norm         │
│                          │    │  • → tensor (1,3,224,224)│
│  → obs (9,) float32      │    │                          │
└───────────┬──────────────┘    └───────────┬──────────────┘
            │                               │
            │                               ▼
            │              ┌────────────────────────────┐
            │              │  MobileNetV3-Small (Frozen) │
            │              │  → GlobalAvgPool            │
            │              │  → Projection Head          │
            │              │  → visual_features (50-dim) │
            │              └─────────────┬──────────────┘
            │                            │
            │    ┌───────────────────────┘
            │    │
            ▼    ▼
┌─────────────────────────────────────────┐
│         Feature Fusion                   │
│  [visual(50) | sensor_encoded(32)]       │
│  → fused_features (82-dim)               │
└───────────────────┬─────────────────────┘
                    │
      ┌─────────────┴─────────────┐
      ▼                           ▼
┌──────────┐              ┌──────────────┐
│ SACActor │              │  SACCritic   │
│  → μ, σ  │              │  → Q1, Q2    │
│  → π(a|s)│              │  → V(s)      │
└────┬─────┘              └──────────────┘
     │
     │ residual ∈ [-1, 1] × RESIDUAL_SCALE
     │
     ▼
┌────────────────────────────────────────┐
│   ResidualDriveMode (manage.py)        │
│   steering = pilot/angle +              │
│              residual/steering          │
└───────────────────┬────────────────────┘
                    │ UART → Arduino
                    ▼
┌────────────────────────────────────────┐
│   Arduino Mega 2560                    │
│   steering → Servo PWM (MG996R)        │
│   throttle → Motor PWM (DRV8833)       │
└────────────────────────────────────────┘
```

---

## 深度学习模块 — 感知 (DL Perception)

> **DL is for Perception, NOT control. RL makes decisions.**

### 预训练 Backbone (Transfer Learning)

| Backbone | 参数量 | 输入尺寸 | 输出维度 | 预训练权重 | 适用场景 |
|----------|--------|----------|----------|------------|----------|
| `mobilenet_v3_small` ★ | ~2.5M | 224² | 576→50 | ImageNet | **默认** — 嵌入式/RPi |
| `mobilenet_v2` | ~3.5M | 224² | 1280→50 | ImageNet | 更高精度 |
| `efficientnet_b0` | ~5.3M | 224² | 1280→50 | ImageNet | 最佳精度 |
| `donkey_cnn` (baseline) | ~70K | 120×160 | 50 | 无 | 消融实验/对比基线 |

### 图像处理管线

```
Camera (OpenCV BGR, uint8, 120×160, [0,255])
    │
    ▼
BGR → RGB (.copy() 防止负 stride)
    │
    ▼
Resize → 224×224 (bilinear)
    │
    ▼
ImageNet Normalization
  mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
    │
    ▼
MobileNetV3-Small (FROZEN) → GlobalAvgPool
    │                           Feature maps → (576,)
    ▼
Projection Head: Linear(576,128) → ReLU → Dropout → Linear(128,50)
    │
    ▼
visual_features (50-dim)
```

### 迁移学习策略

| 阶段 | Backbone | Projection Head | SAC Heads | 学习率 |
|------|----------|-----------------|-----------|--------|
| **Phase 1** (warm-start) | **Frozen** | Trainable | Trainable | 3e-4 |
| **Phase 2** (fine-tune) | Last N blocks unfrozen | Trainable | Trainable | 1e-4 |

---

## 多传感器融合 (Multi-Sensor Fusion)

### 传感器观测向量 (9-dim)

| 索引 | 传感器 | 物理量 | 单位 | 归一化 |
|------|--------|--------|------|--------|
| 0 | Encoder | 车速 (speed) | m/s | [-1,1] |
| 1 | Encoder | 加速度 (accel, derived) | m/s² | [-1,1] |
| 2 | IMU | 加速度 X | g | [-1,1] |
| 3 | IMU | 加速度 Y | g | [-1,1] |
| 4 | IMU | 加速度 Z | g | [-1,1] |
| 5 | IMU | 角速度 X (roll) | deg/s | [-1,1] |
| 6 | IMU | 角速度 Y (pitch) | deg/s | [-1,1] |
| 7 | IMU | 角速度 Z (yaw) | deg/s | [-1,1] |
| 8 | HC-SR04 | 前方障碍物距离 | cm | [1,-1] |

### 传感器编码器 (SensorEncoder)

```
sensor obs (9,) → Linear(9,64) → ReLU → Dropout
                → Linear(64,64) → ReLU → Dropout
                → Linear(64,32) → sensor_features (32,)
```

缺失传感器的通道填 0（鲁棒性）。所有值归一化到 [-1,1] 以稳定 SAC 训练。

### 硬件参数参考 (Hardware Specs for Simulation)

仿真/数字孪生环境需匹配以下实际硬件参数：

| 模块 | 型号 | 关键参数 | 接口 | Arduino Mega 2560 | Raspberry Pi |
|------|------|----------|------|-------------------|--------------|
| 摄像头 | USB Camera | 120×160, 20 FPS, BGR | USB | — | USB Port |
| 编码器 | Hall Encoder ×2 | 20 线/转, 轮径 65mm | GPIO 中断 | D2 (INT0), D3 (INT1) | — |
| IMU | MPU6050 | 6-DOF, ±2g, ±250°/s, DLP 5Hz | I²C (0x68) | SDA(A4), SCL(A5) | SDA(GPIO2), SCL(GPIO3) |
| 超声波 | HC-SR04 | 2–400cm, 5V, 15° 波束角 | Trig/Echo | Trig(D8), Echo(D9) | — |
| 电机驱动 | DRV8833 | 2 通道, 2.7–10.8V, 1.5A/通道 | PWM×4 | IN1(D5), IN2(D6), IN3(D9), IN4(D10) | — |
| 转向舵机 | MG996R | 4.8–7.2V, 10kg·cm, 0.17s/60° | PWM 50Hz | D7 (PCA9685 可选) | — |

> **仿真参数说明**：以上参数定义了传感器噪声范围、控制延迟和物理约束，在 DonkeySim/Webots 中构建数字孪生时对应配置。编码器线数和轮径决定 `MM_PER_TICK` 计算；HC-SR04 的 15° 波束角和 400cm 量程对应仿真中的 ray-cast 参数；舵机 0.17s/60° 的响应时间决定转向延迟。

---

## 残差强化学习 (Residual RL)

### SAC (Soft Actor-Critic) 算法

| 特性 | 描述 |
|------|------|
| Off-Policy | 从历史 Tub 数据学习，无需在线交互 |
| Max Entropy | 最大熵探索，自动平衡 explore/exploit |
| Twin Q | 双 Q 网络减少价值高估偏差 |
| Auto Alpha | 自动调节熵温度系数 |
| Multi-Modal | Image (50-dim) + Sensor (32-dim) → Fused (82-dim) |

### 训练模式

| 模式 | 描述 | 命令 |
|------|------|------|
| **Transfer Learning** | Frozen backbone, 训练 projection + SAC heads | `python train_residual.py --tubs data/ --base models/mypilot.h5` |
| **Fine-Tuning** | Unfreeze backbone, 全模型微调 | `python train_residual.py ... --unfreeze` |
| **Multi-Modal** | 图像 + 传感器联合输入 | 设置 `USE_MULTI_MODAL=True` |

### 控制公式

```
residual_raw = π_θ(image, sensor_obs)  ∈ [-1, 1]
residual = residual_raw × RESIDUAL_SCALE  ∈ [-0.3, 0.3]
steering = base_pilot.angle + residual
```

---

## 数字孪生工作流 (Digital Twin Workflow)

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│Simulation│ →  │ Training │ →  │Validation│ →  │Sim-to-  │ →  │Deployment│
│          │    │          │    │          │    │Real      │    │          │
│DonkeySim │    │SAC Off-  │    │Metrics:  │    │Domain    │    │RPi +     │
│Webots    │    │line RL   │    │Success   │    │Randomiz- │    │Arduino   │
│Gazebo    │    │(Residual │    │Rate, MSE │    │ation     │    │Real Car  │
│          │    │ Targets) │    │Collision │    │          │    │          │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
     │                                                               │
     └─────────────────── Feedback Loop ─────────────────────────────┘
              (Real-world data → improve sim → retrain → redeploy)
```

---

## Raspberry Pi ↔ Arduino 通信架构

```
┌──────────────────────┐         ┌──────────────────────────┐
│   Raspberry Pi        │         │   Arduino Mega 2560       │
│                       │         │                           │
│  ✓ DL Perception      │  UART   │  ✓ Motor PWM Control      │
│    (MobileNetV3)      │◄───────►│    (DRV8833 / L298N)      │
│                       │  I²C    │                           │
│  ✓ RL Decision        │         │  ✓ Servo Control          │
│    (SAC + Residual)   │         │    (MG996R Steering)      │
│                       │         │                           │
│  ✓ SensorFusion       │         │  ✓ Raw Sensor Read        │
│    (IMU/Obstacle/Line)│         │    (ADC for IR)           │
│                       │         │                           │
│  ✓ High-Level Logic   │         │  ✓ Encoder Tick Counting  │
│    (Path Planning)    │         │    (Interrupt-driven)     │
└──────────────────────┘         └──────────────────────────┘

通信协议: UART 115200 baud
  Pi → Arduino: steering (float), throttle (float)
  Arduino → Pi: encoder ticks, raw sensor values
```

---

## 项目目录结构 (Folder Structure)

```
cc+hacker final/
│
├── config.py                    # DonkeyCar 默认配置 (不改)
├── myconfig.py                  # ★ 用户配置 (所有开关)
├── manage.py                    # ★ 主入口 (Vehicle 管线注册)
├── train.py                     # Keras BC 训练 (不改)
├── train_residual.py            # ★ SAC 训练脚本
├── calibrate.py                 # 校准 (不改)
│
├── parts/
│   ├── __init__.py              # 模块初始化
│   ├── residual_rl.py           # ★ SAC + ResidualPilot + Multi-Modal
│   └── sensors.py               # ★ SensorFusion + 传感器框架
│
├── models/
│   ├── mypilot.h5               # Base KerasLinear (BC) 模型
│   └── residual_sac.pth         # SAC 残差模型 (支持 multi-modal)
│
├── data/
│   └── tub_*/                   # Tub 录制数据
│
├── arduino/                     # 📋 PLANNED — Arduino 固件
│   └── motor_control/           #    电机 PWM + 编码器读取
│
├── simulation/                  # 📋 PLANNED — 数字孪生仿真
│   └── donkey_gym/              #    DonkeySim 环境包装
│
└── README/
    ├── README.md                # 项目总览 (本文件)
    ├── ARCHITECTURE.md          # 详细架构文档
    └── USAGE.md                 # 使用指南
```

---

## 配置参考 (Configuration Quick Reference)

### 必须配置 (`myconfig.py`)

```python
# --- 总开关 ---
RESIDUAL_RL = True               # 启用残差 RL
USE_MULTI_MODAL = False          # 启用多模态传感器输入 (需 sensors.py)

# --- 残差控制 ---
RESIDUAL_SCALE = 0.3             # 残差缩放因子
RESIDUAL_MODEL_PATH = 'models/residual_sac.pth'

# --- Backbone ---
RESIDUAL_BACKBONE = 'mobilenet_v3_small'  # ★ 默认
RESIDUAL_FEATURE_DIM = 50
RESIDUAL_FREEZE_BACKBONE = True  # 迁移学习

# --- SAC 超参数 ---
RESIDUAL_HIDDEN_DIM = 256
RESIDUAL_LR_ACTOR = 3e-4
RESIDUAL_GAMMA = 0.99
RESIDUAL_BUFFER_SIZE = 100000
RESIDUAL_BATCH_SIZE = 256

# --- 传感器 (仅在 USE_MULTI_MODAL=True 时生效) ---
SENSOR_DIM = 9
SENSOR_FEATURE_DIM = 32
ENABLE_ENCODER = False
ENABLE_IMU = False
ENABLE_OBSTACLE = False
ENABLE_LINE_TRACKING = False
```

### 快速开始命令

```bash
# 1. 训练 Base Pilot (BC)
python train.py --tubs data/ --model models/mypilot.h5

# 2. 训练残差 RL (image-only)
python train_residual.py --tubs data/ --base models/mypilot.h5

# 3. 训练残差 RL (multi-modal) — 需要传感器 Tub 数据
#    先在 myconfig.py 中设置 USE_MULTI_MODAL=True
python train_residual.py --tubs data/ --base models/mypilot.h5

# 4. Fine-tune (unfreeze backbone)
python train_residual.py --tubs data/ --base models/mypilot.h5 --unfreeze

# 5. 部署驾驶
python manage.py drive --model models/mypilot.h5
```

---

## 当前实现状态 (Implementation Status)

### ✅ 已实现 (Done)

| 模块 | 文件 | 状态 |
|------|------|------|
| DonkeyCar Base (Vehicle/Config/Train) | `config.py`, `manage.py`, `train.py` | ✅ 完整 |
| KerasLinear Base Pilot (BC) | DonkeyCar 内置 | ✅ 完整 |
| SAC Agent (Twin Q + Auto Alpha) | `parts/residual_rl.py` | ✅ 完整 |
| Pluggable Backbone (MobileNetV2/V3/EfficientNet) | `parts/residual_rl.py` | ✅ 完整 |
| Transfer Learning (Frozen backbone + Projection Head) | `parts/residual_rl.py` | ✅ 完整 |
| BGR→RGB ImageProcessor (防负 stride) | `parts/residual_rl.py` | ✅ 完整 |
| ResidualPilot (DonkeyCar Part) | `parts/residual_rl.py` | ✅ 完整 |
| ResidualDriveMode (final_steering = base + residual) | `manage.py` | ✅ 完整 |
| Offline RL Training (train_residual.py) | `train_residual.py` | ✅ 完整 |
| SensorFusion 框架 (Encoder/IMU/Obstacle/Line) | `parts/sensors.py` | ✅ 框架完成 |
| SensorEncoder MLP (多模态融合) | `parts/residual_rl.py` | ✅ 完整 |
| Multi-Modal Replay Buffer | `parts/residual_rl.py` | ✅ 完整 |
| Tub 传感器数据加载 | `parts/residual_rl.py` | ✅ 框架完成 |

### 📋 计划中 (Planned)

| 模块 | 描述 | 优先级 |
|------|------|--------|
| **Arduino Mega 2560 固件** | 电机 PWM、编码器读取、传感器预处理、UART 通信协议 | 🔴 高 |
| **Digital Twin 仿真** | DonkeySim/Webots 集成、Domain Randomization、Sim-to-Real 验证 | 🟡 中 |
| **Action Smoothing** | EMA 平滑 + 限幅，防止 RL 输出突变导致抖动 | 🟡 中 |
| **Safety Constraints** | 最小障碍物距离 → 紧急刹车、最大转向角限制 | 🟡 中 |
| **RPi ↔ Arduino 通信** | UART/I²C 协议实现、丢包处理、Watchdog | 🔴 高 |
| **在线 Fine-Tuning** | 驾驶过程中持续更新 SAC（online RL with safety） | 🟢 低 |
| **多传感器时间戳同步** | 精确时间对齐、传感器延迟补偿 | 🟡 中 |
| **硬件传感器驱动** | GPIO 编码器、I²C IMU、超声波、IR 阵列的实际驱动 | 🔴 高 |

---

## 未来扩展 (Future Extensions)

1. **多任务 RL** — 同时学习 steering + throttle 残差（当前仅 steering）
2. **Curiosity-Driven RL** — 用 ICM (Intrinsic Curiosity Module) 鼓励探索新轨迹
3. **Multi-Agent RL** — 多车协作/竞争场景
4. **Online Domain Adaptation** — 实时调整仿真→现实 gap
5. **Vision Transformer Backbone** — 替换 CNN 为 ViT/DeiT 提升感知能力
6. **SLAM Integration** — 结合 ORB-SLAM3 提供全局定位
7. **Cloud Training → Edge Deployment** — AWS/GCP 训练 → Raspberry Pi 推理

---

## 依赖 (Dependencies)

```
donkeycar >= 5.0.0
torch >= 1.9.0
torchvision >= 0.10.0
numpy
# 可选 (仿真):
# gym, pybullet, webots
# 可选 (硬件):
# smbus2 (I²C), RPi.GPIO, pyserial (Arduino UART)
```

---

## 参考文献

- SAC: [Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL](https://arxiv.org/abs/1801.01290)
- MobileNetV3: [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
- EfficientNet: [Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946)
- DonkeyCar: [https://docs.donkeycar.com/](https://docs.donkeycar.com/)
- Digital Twin: [A Review of Digital Twin in Robotics](https://arxiv.org/abs/2203.08876)
