# Residual RL 使用指南 (Usage Guide)

## 快速开始 (Quick Start)

### 前提条件

1. 已有训练好的 KerasLinear 基础模型：`models/mypilot.h5`
2. 已有录制好的 Tub 数据：`data/` 目录
3. 安装 PyTorch：`pip install torch`

### Step 1 — 启用 Residual RL 配置

编辑 `myconfig.py`，将以下行反注释：

```python
# 启用 Residual RL
RESIDUAL_RL = True

# 残差缩放 (推荐 0.2~0.5)
RESIDUAL_SCALE = 0.3

# 模型路径
RESIDUAL_MODEL_PATH = 'models/residual_sac.pth'
```

### Step 2 — 训练残差模型

```bash
# 使用 data/ 下所有 tub 训练
python train_residual.py --tubs data/ --base models/mypilot.h5

# 使用指定 tub 训练
python train_residual.py --tubs data/tub_1 data/tub_2 --base models/mypilot.h5

# 自定义参数
python train_residual.py --tubs data/ --base models/mypilot.h5 \
    --epochs 200 --steps 2000 --output models/residual_v2.pth

# 从检查点继续训练
python train_residual.py --tubs data/ --base models/mypilot.h5 \
    --resume models/residual_sac.pth
```

### Step 3 — 部署 & 驾驶

```bash
# 启动带 Residual RL 的自动驾驶
python manage.py drive --model models/mypilot.h5
```

Vehicle 会自动加载 `models/residual_sac.pth`（如果存在）。

---

## 配置参考

### 完整配置项 (`myconfig.py`)

```python
# ===== Residual RL Configuration =====

# 总开关：True 启用残差 RL
RESIDUAL_RL = True

# 残差缩放：最终 steering = base + residual * scale
# - 0.1: 很小修正，几乎感觉不到
# - 0.3: 推荐值，明显修正但不危险
# - 0.5: 较大修正，RL 有很高权限
# - 1.0: RL 完全接管转向
RESIDUAL_SCALE = 0.3

# 模型路径
RESIDUAL_MODEL_PATH = 'models/residual_sac.pth'

# ---- SAC 超参数 (通常保持默认) ----
RESIDUAL_HIDDEN_DIM = 256
RESIDUAL_LR_ACTOR = 3e-4
RESIDUAL_LR_CRITIC = 3e-4
RESIDUAL_LR_ALPHA = 3e-4
RESIDUAL_GAMMA = 0.99
RESIDUAL_TAU = 0.005
RESIDUAL_BUFFER_SIZE = 100000
RESIDUAL_BATCH_SIZE = 256
RESIDUAL_LOG_FREQ = 200
```

---

## 调参指南

### 场景 1: 残差太小，没效果

```python
RESIDUAL_SCALE = 0.5    # 增大
```

### 场景 2: 车辆抖动/不稳定

```python
RESIDUAL_SCALE = 0.1    # 减小
RESIDUAL_TAU = 0.01     # 更平滑的目标网络更新
```

### 场景 3: 训练不收敛

```python
RESIDUAL_LR_ACTOR = 1e-4      # 降低学习率
RESIDUAL_LR_CRITIC = 1e-4
RESIDUAL_BATCH_SIZE = 512     # 增大 batch size
```

### 场景 4: 过拟合

```python
# 收集更多 Tub 数据
# 或减小训练轮数
python train_residual.py ... --epochs 50
```

---

## 故障排除 (Troubleshooting)

### ImportError: No module named 'torch'

```bash
pip install torch
```

### ImportError: cannot import name 'ResidualPilot'

确保 `parts/` 目录结构正确：
```
parts/
├── __init__.py
└── residual_rl.py
```

### 训练时 "No Tub directories found"

检查 `data/` 目录下是否有 Tub 子目录（含有 `record_*.json` 文件）。

### 驾驶时 "No Residual RL model loaded"

这只是一个 Warning。如果还没训练模型，ResidualPilot 会输出 0（即纯基础模型）。训练后：
```bash
python train_residual.py --tubs data/ --base models/mypilot.h5
```

---

## 目录结构 (完整)

```
cc+hacker final/
├── config.py                  # 默认配置 (不修改)
├── myconfig.py                # ★ 用户配置 (添加 RESIDUAL_RL 段)
├── manage.py                  # ★ 主入口 (添加 ResidualPilot 注册)
├── train.py                   # Keras 训练 (不修改)
├── train_residual.py          # ★ SAC 训练脚本
├── calibrate.py               # 校准 (不修改)
├── parts/
│   ├── __init__.py            # 模块初始化
│   └── residual_rl.py        # ★ SAC + ResidualPilot
├── models/
│   ├── mypilot.h5            # Base KerasLinear 模型
│   └── residual_sac.pth      # SAC 残差模型
├── data/
│   └── tub_*/                # Tub 录制数据
└── README/
    ├── README.md             # 概述
    ├── ARCHITECTURE.md       # 详细架构
    └── USAGE.md              # 本文件
```
