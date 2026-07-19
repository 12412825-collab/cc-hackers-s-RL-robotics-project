"""
Residual RL Module for DonkeyCar Autonomous Driving
====================================================

Architecture:
  - SAC (Soft Actor-Critic) algorithm for continuous steering residual
  - Pluggable CNN backbone with transfer learning support:
    * donkey_cnn       — original 5-conv from-scratch CNN (baseline)
    * mobilenet_v3_small — MobileNetV3-Small, ImageNet pretrained (default)
    * mobilenet_v2       — MobileNetV2, ImageNet pretrained
    * efficientnet_b0    — EfficientNet-B0, ImageNet pretrained
  - MULTI-MODAL fusion: visual features + sensor observation vector -> SAC
  - ResidualPilot: DonkeyCar Part, takes camera image + optional sensor obs,
    outputs residual steering
  - Final control: steering = base_pilot_steering + residual_steering * scale

Data Flow (Multi-Modal):
  Camera (120x160 BGR)
    -> ImageProcessor (BGR->RGB, Resize, ImageNet Norm)
    -> Backbone (MobileNetV3-Small) -> visual_features (50-dim)

  Sensors (Encoder/IMU/Obstacle/Line)
    -> SensorFusion -> observation vector (12-dim)
    -> SensorEncoder MLP -> sensor_features (32-dim)

  [visual_features | sensor_features] -> fused (82-dim)
    -> SACActor -> residual in [-1, 1]
    -> SACCritic -> Q-value

Design Principles:
  - Does NOT modify original KerasPilot or any DonkeyCar source files
  - Duck-typed Part: just needs run(img_arr, sensor_obs=None) -> residual
  - Compatible with entire original training pipeline (TubWriter unchanged)
  - Transfer learning: pretrained weights + freeze early layers
  - Multi-modal: sensor_dim=0 falls back to image-only (backward compatible)
"""

import os
import sys
import time
import logging
import numpy as np
from collections import deque
from typing import Tuple, Optional, Dict, List

# ---------------------------------------------------------------------------
# Check PyTorch / torchvision availability
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Normal
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import torchvision
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

logger = logging.getLogger(__name__)


# ===========================================================================
# Backbone Registry
# ===========================================================================

BACKBONE_REGISTRY = {}


def _register(name, input_size, feature_dim):
    def decorator(fn):
        BACKBONE_REGISTRY[name] = {
            'input_size': input_size,
            'feature_dim': feature_dim,
            'factory': fn,
        }
        return fn
    return decorator


# ===========================================================================
# 1. CNN Encoders
# ===========================================================================

class DonkeyCNN(nn.Module):
    """PyTorch reimplementation of DonkeyCar's default 5-layer CNN encoder.
    Trained FROM SCRATCH. Input: (B,3,120,160) in [0,1]. Output: (B,50)."""

    def __init__(self, input_shape=(3, 120, 160), feature_dim=50):
        super().__init__()
        c, h, w = input_shape

        self.conv1 = nn.Conv2d(c, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            dummy = F.relu(self.conv1(dummy))
            dummy = F.relu(self.conv2(dummy))
            dummy = F.relu(self.conv3(dummy))
            dummy = F.relu(self.conv4(dummy))
            dummy = F.relu(self.conv5(dummy))
            self.flat_size = dummy.view(1, -1).size(1)

        self.fc1 = nn.Linear(self.flat_size, 100)
        self.fc2 = nn.Linear(100, feature_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


# --- 1b. Pretrained Backbone Wrapper ---

class PretrainedBackbone(nn.Module):
    """Wraps a torchvision pretrained model as a feature extractor.
    Architecture: Backbone(frozen) -> GlobalAvgPool -> ProjectionHead -> feature
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(self, backbone_type='mobilenet_v3_small',
                 feature_dim=50, freeze_backbone=True, pretrained=True):
        super().__init__()

        if not TORCHVISION_AVAILABLE:
            raise ImportError(
                "torchvision is required for pretrained backbones. "
                "Install: pip install torchvision")

        self.backbone_type = backbone_type
        self.feature_dim = feature_dim

        self.backbone, self.backbone_dim, self.input_size = \
            self._build_backbone(backbone_type, pretrained)

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info(
                f"Backbone '{backbone_type}' frozen "
                f"({sum(p.numel() for p in self.backbone.parameters()):,} params)")

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Sequential(
            nn.Linear(self.backbone_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, feature_dim),
        )

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"PretrainedBackbone '{backbone_type}': "
            f"total={total:,} trainable={trainable:,} "
            f"backbone_dim={self.backbone_dim} feature_dim={feature_dim}")

    def _build_backbone(self, backbone_type, pretrained):
        weights = 'DEFAULT' if pretrained else None

        if backbone_type == 'mobilenet_v3_small':
            model = torchvision.models.mobilenet_v3_small(weights=weights)
            self._extracted_features = model.features
            self._extracted_avgpool = model.avgpool
            return self._extracted_features, 576, 224

        elif backbone_type == 'mobilenet_v2':
            model = torchvision.models.mobilenet_v2(weights=weights)
            self._extracted_features = model.features
            self._extracted_avgpool = nn.AdaptiveAvgPool2d(1)
            return self._extracted_features, 1280, 224

        elif backbone_type == 'efficientnet_b0':
            model = torchvision.models.efficientnet_b0(weights=weights)
            self._extracted_features = model.features
            self._extracted_avgpool = model.avgpool
            return self._extracted_features, 1280, 224

        else:
            raise ValueError(
                f"Unknown backbone_type: '{backbone_type}'. "
                f"Available: {self.available_backbones()}")

    @staticmethod
    def available_backbones():
        return ['mobilenet_v3_small', 'mobilenet_v2', 'efficientnet_b0']

    @staticmethod
    def get_imagenet_norm():
        return (
            torch.tensor(PretrainedBackbone.IMAGENET_MEAN).view(1, 3, 1, 1),
            torch.tensor(PretrainedBackbone.IMAGENET_STD).view(1, 3, 1, 1),
        )

    def forward(self, x):
        """x: (B, 3, H, W) already resized and normalized. Returns (B, feature_dim)."""
        x = self._extracted_features(x)
        x = self._extracted_avgpool(x)
        x = torch.flatten(x, 1)
        x = self.project(x)
        return x

    def unfreeze(self, num_layers_to_unfreeze=0):
        """Unfreeze backbone for fine-tuning. 0 = unfreeze ALL."""
        self.freeze_backbone = False
        if num_layers_to_unfreeze == 0:
            for param in self.backbone.parameters():
                param.requires_grad = True
            logger.info("Unfroze ALL backbone layers")
        else:
            children = list(self.backbone.children())
            for child in children[:-num_layers_to_unfreeze]:
                for param in child.parameters():
                    param.requires_grad = False
            for child in children[-num_layers_to_unfreeze:]:
                for param in child.parameters():
                    param.requires_grad = True
            logger.info(
                f"Unfroze last {num_layers_to_unfreeze}/{len(children)} blocks")


# ===========================================================================
# 2. Image Processor
# ===========================================================================

class ImageProcessor:
    """Unified image preprocessing for all backbones.
    - donkey_cnn: normalize [0,1], keep original size
    - pretrained: BGR->RGB, resize, ImageNet normalize
    """

    def __init__(self, backbone_type='mobilenet_v3_small',
                 input_shape=(3, 120, 160), device='cpu'):
        self.backbone_type = backbone_type
        self.input_shape = input_shape
        self.device = device

        if backbone_type == 'donkey_cnn':
            self.input_size = (input_shape[1], input_shape[2])
            self.use_imagenet_norm = False
            self.needs_resize = False
        else:
            info = BACKBONE_REGISTRY.get(backbone_type, {'input_size': 224})
            self.input_size = info['input_size']
            self.use_imagenet_norm = True
            self.needs_resize = (
                self.input_size != input_shape[1] or
                self.input_size != input_shape[2])

        if self.use_imagenet_norm:
            self._mean = torch.tensor(
                PretrainedBackbone.IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
            self._std = torch.tensor(
                PretrainedBackbone.IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def numpy_to_tensor(self, img_arr):
        """Raw camera image (uint8 HWC BGR) -> model-ready tensor (1,C,H',W')."""
        if img_arr.dtype == np.uint8:
            img = img_arr.astype(np.float32) / 255.0
        else:
            img = img_arr.astype(np.float32)

        if self.use_imagenet_norm and img.shape[-1] == 3:
            img = img[..., ::-1].copy()

        img = np.transpose(img, (2, 0, 1))
        tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)

        if self.needs_resize:
            tensor = F.interpolate(tensor,
                size=(self.input_size, self.input_size),
                mode='bilinear', align_corners=False)

        if self.use_imagenet_norm:
            tensor = (tensor - self._mean) / self._std

        return tensor

    def process_batch(self, img_batch):
        """Batch images (B,C,H,W) [0,1] -> preprocessed."""
        if self.needs_resize:
            img_batch = F.interpolate(img_batch,
                size=(self.input_size, self.input_size),
                mode='bilinear', align_corners=False)

        if self.use_imagenet_norm:
            self._mean = self._mean.to(img_batch.device)
            self._std = self._std.to(img_batch.device)
            img_batch = (img_batch - self._mean) / self._std

        return img_batch


# ===========================================================================
# 3. Sensor Encoder — multi-sensor observation -> feature
# ===========================================================================

class SensorEncoder(nn.Module):
    """MLP that encodes multi-sensor observation vector into compact features
    for fusion with visual features.

    Architecture:
      Sensor Vector (sensor_dim, e.g. 12)
        -> Linear(sensor_dim, 64) -> ReLU -> Dropout
        -> Linear(64, 64) -> ReLU -> Dropout
        -> Linear(64, sensor_feature_dim)  -> sensor_features (e.g. 32-dim)

    The sensor observation vector comes from SensorFusion (parts/sensors.py).
    """

    def __init__(self, sensor_dim=12, hidden_dim=64,
                 sensor_feature_dim=32, dropout=0.1):
        super().__init__()
        self.sensor_dim = sensor_dim
        self.sensor_feature_dim = sensor_feature_dim

        self.net = nn.Sequential(
            nn.Linear(sensor_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, sensor_feature_dim),
        )

        total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"SensorEncoder: sensor_dim={sensor_dim} -> "
            f"sensor_feature_dim={sensor_feature_dim} ({total:,} params)")

    def forward(self, sensor_vec):
        """sensor_vec: (B, sensor_dim) -> (B, sensor_feature_dim)."""
        return self.net(sensor_vec)


# ===========================================================================
# 4. SAC Networks
# ===========================================================================

class SACActor(nn.Module):
    """Gaussian policy network. Input: fused features (B, feature_dim).
    Output: action mean (B,1), log_std (B,1)."""

    def __init__(self, feature_dim=50, action_dim=1, hidden_dim=256,
                 log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, features):
        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, features):
        mean, log_std = self.forward(features)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic_action(self, features):
        mean, _ = self.forward(features)
        return torch.tanh(mean)


class SACCritic(nn.Module):
    """Twin Q-network. Input: fused features (B, feature_dim) + action (B,1).
    Output: Q1 (B,1), Q2 (B,1)."""

    def __init__(self, feature_dim=50, action_dim=1, hidden_dim=256):
        super().__init__()
        self.fc1_1 = nn.Linear(feature_dim + action_dim, hidden_dim)
        self.fc1_2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1 = nn.Linear(hidden_dim, 1)

        self.fc2_1 = nn.Linear(feature_dim + action_dim, hidden_dim)
        self.fc2_2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2 = nn.Linear(hidden_dim, 1)

    def forward(self, features, action):
        x = torch.cat([features, action], dim=-1)
        q1 = F.relu(self.fc1_1(x))
        q1 = F.relu(self.fc1_2(q1))
        q1 = self.q1(q1)
        q2 = F.relu(self.fc2_1(x))
        q2 = F.relu(self.fc2_2(q2))
        q2 = self.q2(q2)
        return q1, q2


# ===========================================================================
# 5. SAC Agent — pluggable backbone + optional multi-modal + training
# ===========================================================================

class SACAgent:
    """Soft Actor-Critic agent for residual steering learning.

    Key features:
      - Pluggable backbone (donkey_cnn / mobilenet_v3_small / mobilenet_v2 / efficientnet_b0)
      - Transfer learning with frozen pretrained backbone
      - Multi-modal fusion: visual + sensor observation -> SAC heads
      - sensor_dim=0 falls back to image-only (backward compatible)
      - Twin Q-networks + auto entropy tuning + soft target updates
    """

    def __init__(self,
                 input_shape=(3, 120, 160),
                 action_dim=1,
                 hidden_dim=256,
                 backbone_type='mobilenet_v3_small',
                 feature_dim=50,
                 freeze_backbone=True,
                 sensor_dim=0,
                 sensor_feature_dim=32,
                 sensor_hidden_dim=64,
                 lr_actor=3e-4,
                 lr_critic=3e-4,
                 lr_alpha=3e-4,
                 gamma=0.99,
                 tau=0.005,
                 target_entropy=None,
                 device=None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install: pip install torch")

        self.backbone_type = backbone_type
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Multi-modal config
        self.sensor_dim = sensor_dim
        self.sensor_feature_dim = sensor_feature_dim
        self.use_multi_modal = (sensor_dim > 0)
        self.visual_feature_dim = feature_dim

        # --- Build visual encoder ---
        if backbone_type == 'donkey_cnn':
            self.encoder = DonkeyCNN(input_shape, feature_dim=feature_dim).to(self.device)
        else:
            self.encoder = PretrainedBackbone(
                backbone_type=backbone_type,
                feature_dim=feature_dim,
                freeze_backbone=freeze_backbone,
                pretrained=True,
            ).to(self.device)

        # --- Build sensor encoder (optional) ---
        if self.use_multi_modal:
            self.sensor_encoder = SensorEncoder(
                sensor_dim=sensor_dim,
                hidden_dim=sensor_hidden_dim,
                sensor_feature_dim=sensor_feature_dim,
            ).to(self.device)
            self.fused_feature_dim = feature_dim + sensor_feature_dim
            logger.info(
                f"SACAgent: Multi-modal ON — "
                f"visual={feature_dim} + sensor={sensor_feature_dim} "
                f"-> fused={self.fused_feature_dim}")
        else:
            self.sensor_encoder = None
            self.fused_feature_dim = feature_dim
            logger.info(
                f"SACAgent: Image-only mode — feature_dim={feature_dim}")

        # --- Image processor ---
        self.img_processor = ImageProcessor(
            backbone_type=backbone_type,
            input_shape=input_shape,
            device=self.device)

        # --- SAC Heads (use fused dim) ---
        self.actor = SACActor(
            feature_dim=self.fused_feature_dim,
            action_dim=action_dim, hidden_dim=hidden_dim).to(self.device)
        self.critic = SACCritic(
            feature_dim=self.fused_feature_dim,
            action_dim=action_dim, hidden_dim=hidden_dim).to(self.device)
        self.critic_target = SACCritic(
            feature_dim=self.fused_feature_dim,
            action_dim=action_dim, hidden_dim=hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # --- Optimizers ---
        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        if self.use_multi_modal:
            encoder_params += list(self.sensor_encoder.parameters())

        self.actor_optimizer = optim.Adam(
            encoder_params + list(self.actor.parameters()), lr=lr_actor)
        self.critic_optimizer = optim.Adam(
            encoder_params + list(self.critic.parameters()), lr=lr_critic)

        # --- Auto entropy tuning ---
        self.target_entropy = target_entropy if target_entropy is not None else -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
        self.alpha = self.log_alpha.exp().item()
        self.training = True

        # Log
        total = sum(p.numel() for p in self.encoder.parameters())
        trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        logger.info(
            f"SACAgent: backbone={backbone_type}, visual_dim={feature_dim}, "
            f"sensor_dim={sensor_dim}, fused_dim={self.fused_feature_dim}, "
            f"encoder: total={total:,}, trainable={trainable:,}")

    # ---- Preprocessing & Encoding ----

    def preprocess(self, img_arr):
        """Raw camera image -> model-ready tensor."""
        return self.img_processor.numpy_to_tensor(img_arr)

    def encode(self, img_tensor, sensor_vec=None):
        """Encode image (and optionally sensor) into fused features.

        Args:
            img_tensor: (B,3,H,W) preprocessed
            sensor_vec: (B,sensor_dim) or None
        Returns:
            fused_features: (B, fused_feature_dim)
        """
        vis_features = self.encoder(img_tensor)

        if self.use_multi_modal:
            if sensor_vec is None:
                B = img_tensor.size(0)
                sensor_vec = torch.zeros(B, self.sensor_dim,
                    device=img_tensor.device, dtype=img_tensor.dtype)
            sen_features = self.sensor_encoder(sensor_vec)
            return torch.cat([vis_features, sen_features], dim=-1)
        return vis_features

    # ---- Action Selection ----

    def select_action(self, img_arr, sensor_obs=None, deterministic=False):
        """Select residual steering from camera image + optional sensor data.

        Args:
            img_arr: camera image (H,W,C) uint8
            sensor_obs: optional sensor observation vector (sensor_dim,)
            deterministic: True=inference, False=exploration
        Returns:
            residual_steering: float32 scalar in [-1, 1]
        """
        img_tensor = self.preprocess(img_arr)
        sensor_tensor = None
        if self.use_multi_modal and sensor_obs is not None:
            sensor_tensor = torch.from_numpy(
                sensor_obs.astype(np.float32)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.encode(img_tensor, sensor_tensor)
            if deterministic:
                action = self.actor.deterministic_action(features)
            else:
                action, _ = self.actor.sample(features)
        return action.cpu().numpy().squeeze()

    # ---- SAC Update ----

    def update(self, replay_buffer, batch_size=256):
        """Perform one SAC update step. Returns loss dict."""
        if len(replay_buffer) < batch_size:
            return {}

        batch = replay_buffer.sample(batch_size)

        if replay_buffer.has_sensor_data:
            states, sensor_states, actions, rewards, \
                next_states, sensor_next_states, dones = batch
            sensor_states = sensor_states.to(self.device) if sensor_states is not None else None
            sensor_next_states = sensor_next_states.to(self.device) if sensor_next_states is not None else None
        else:
            states, actions, rewards, next_states, dones = batch
            sensor_states = None
            sensor_next_states = None

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        states = self.img_processor.process_batch(states)
        next_states = self.img_processor.process_batch(next_states)

        features = self.encode(states, sensor_states)
        with torch.no_grad():
            next_features = self.encode(next_states, sensor_next_states)

        # --- Critic Update ---
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_features)
            q1_next, q2_next = self.critic_target(next_features, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + self.gamma * (1 - dones) * q_next

        q1, q2 = self.critic(features, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor Update ---
        new_actions, log_probs = self.actor.sample(features.detach())
        q1_new, q2_new = self.critic(features.detach(), new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Alpha Update ---
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        # --- Soft Update Target Networks ---
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha,
        }

    # ---- Save / Load ----

    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.',
                    exist_ok=True)
        save_dict = {
            'backbone_type': self.backbone_type,
            'encoder': self.encoder.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha,
            'sensor_dim': self.sensor_dim,
            'sensor_feature_dim': self.sensor_feature_dim,
            'visual_feature_dim': self.visual_feature_dim,
            'fused_feature_dim': self.fused_feature_dim,
            'use_multi_modal': self.use_multi_modal,
        }
        if self.use_multi_modal and self.sensor_encoder is not None:
            save_dict['sensor_encoder'] = self.sensor_encoder.state_dict()
        torch.save(save_dict, path)
        logger.info(f"Saved SAC model to {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        saved_backbone = checkpoint.get('backbone_type', 'donkey_cnn')
        if saved_backbone != self.backbone_type:
            logger.warning(
                f"Checkpoint backbone '{saved_backbone}' != "
                f"current '{self.backbone_type}'. Skipping encoder load.")
        else:
            self.encoder.load_state_dict(checkpoint['encoder'])

        if self.use_multi_modal and 'sensor_encoder' in checkpoint:
            self.sensor_encoder.load_state_dict(checkpoint['sensor_encoder'])
            logger.info("Loaded sensor encoder weights")
        elif self.use_multi_modal:
            logger.warning(
                "Checkpoint is image-only but current model is multi-modal. "
                "Sensor encoder will use random init.")

        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha = self.log_alpha.exp().item()
        logger.info(f"Loaded SAC model from {path}")


# ===========================================================================
# 6. Replay Buffer (with optional multi-sensor support)
# ===========================================================================

class ReplayBuffer:
    """Standard replay buffer for off-policy RL.

    Stores images as float32 (C, H_orig, W_orig) [0,1].
    Optionally stores sensor observation vectors for multi-modal training.

    Multi-Modal:
      - sensor_dim > 0 enables sensor storage
      - push() accepts optional sensor_state / sensor_next_state
      - sample() returns sensor data as additional tensors
      - has_sensor_data flag indicates whether stored transitions have sensors
    """

    def __init__(self, capacity=100000, sensor_dim=0):
        self.buffer = deque(maxlen=capacity)
        self.sensor_dim = sensor_dim
        self.has_sensor_data = False

    def push(self, state, action, reward, next_state, done,
             sensor_state=None, sensor_next_state=None):
        if sensor_state is not None and not self.has_sensor_data:
            self.has_sensor_data = True
        self.buffer.append((
            state, action, reward, next_state, done,
            sensor_state, sensor_next_state))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        sensor_states, sensor_next_states = [], []

        for i in indices:
            item = self.buffer[i]
            states.append(item[0])
            actions.append(item[1])
            rewards.append(item[2])
            next_states.append(item[3])
            dones.append(item[4])

            if self.has_sensor_data:
                ss = item[5] if item[5] is not None else np.zeros(self.sensor_dim, dtype=np.float32)
                sn = item[6] if item[6] is not None else np.zeros(self.sensor_dim, dtype=np.float32)
                sensor_states.append(ss)
                sensor_next_states.append(sn)

        if self.has_sensor_data:
            return (
                torch.FloatTensor(np.array(states)),
                torch.FloatTensor(np.array(sensor_states)),
                torch.FloatTensor(np.array(actions)),
                torch.FloatTensor(np.array(rewards)).unsqueeze(-1),
                torch.FloatTensor(np.array(next_states)),
                torch.FloatTensor(np.array(sensor_next_states)),
                torch.FloatTensor(np.array(dones)).unsqueeze(-1),
            )
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(-1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(-1),
        )

    def __len__(self):
        return len(self.buffer)


# ===========================================================================
# 7. ResidualPilot — DonkeyCar Part (inference only)
# ===========================================================================

class ResidualPilot:
    """DonkeyCar Part: Camera Image + optional Sensor Obs -> Residual Steering.

    Vehicle pipeline:
        # Image-only:
        V.add(rl, inputs=['cam/image_array'], outputs=['residual/steering'],
              run_condition='run_pilot')
        # Multi-modal:
        V.add(rl, inputs=['cam/image_array', 'sensor/observation'],
              outputs=['residual/steering'], run_condition='run_pilot')
    """

    def __init__(self, cfg):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. pip install torch")

        self.cfg = cfg
        self.residual_scale = getattr(cfg, 'RESIDUAL_SCALE', 0.3)
        self.model_path = getattr(cfg, 'RESIDUAL_MODEL_PATH', None)
        self.backbone_type = getattr(cfg, 'RESIDUAL_BACKBONE', 'mobilenet_v3_small')
        self.input_shape = (cfg.IMAGE_DEPTH, cfg.IMAGE_H, cfg.IMAGE_W)

        self.sensor_dim = getattr(cfg, 'SENSOR_DIM', 0)
        self.use_multi_modal = getattr(cfg, 'USE_MULTI_MODAL', False)

        self.agent = SACAgent(
            input_shape=self.input_shape,
            action_dim=1,
            hidden_dim=getattr(cfg, 'RESIDUAL_HIDDEN_DIM', 256),
            backbone_type=self.backbone_type,
            feature_dim=getattr(cfg, 'RESIDUAL_FEATURE_DIM', 50),
            freeze_backbone=getattr(cfg, 'RESIDUAL_FREEZE_BACKBONE', True),
            sensor_dim=self.sensor_dim if self.use_multi_modal else 0,
            sensor_feature_dim=getattr(cfg, 'SENSOR_FEATURE_DIM', 32),
            sensor_hidden_dim=getattr(cfg, 'SENSOR_HIDDEN_DIM', 64),
            lr_actor=getattr(cfg, 'RESIDUAL_LR_ACTOR', 3e-4),
            lr_critic=getattr(cfg, 'RESIDUAL_LR_CRITIC', 3e-4),
            gamma=getattr(cfg, 'RESIDUAL_GAMMA', 0.99),
            tau=getattr(cfg, 'RESIDUAL_TAU', 0.005),
        )

        if self.model_path and os.path.exists(self.model_path):
            self.agent.load(self.model_path)
            logger.info(f"Residual RL model loaded from {self.model_path}")
        else:
            logger.warning(
                "No Residual RL model loaded. ResidualPilot outputs zero. "
                "Train with: python train_residual.py")

        self.frame_count = 0
        self.log_freq = getattr(cfg, 'RESIDUAL_LOG_FREQ', 200)

    def run(self, img_arr, sensor_obs=None):
        """DonkeyCar Part interface. Returns residual_steering in [-scale, +scale].

        Args:
            img_arr: camera image (H,W,C) uint8
            sensor_obs: optional sensor observation (sensor_dim,) from SensorFusion
        """
        if img_arr is None:
            return 0.0
        try:
            raw_residual = self.agent.select_action(
                img_arr,
                sensor_obs=sensor_obs if self.use_multi_modal else None,
                deterministic=True)
            residual = float(raw_residual) * self.residual_scale

            self.frame_count += 1
            if self.frame_count % self.log_freq == 0:
                logger.debug(
                    f"ResidualPilot frame {self.frame_count}: residual={residual:.4f}")
            return residual
        except Exception as e:
            logger.error(f"ResidualPilot error: {e}")
            return 0.0

    def shutdown(self):
        pass


# ===========================================================================
# 8. ResidualTrainer — offline training orchestrator
# ===========================================================================

class ResidualTrainer:
    """SAC trainer for Residual RL.

    Supports:
      1. Offline pre-training from Tub data (residual targets from base model)
      2. Optional sensor data loading from Tub for multi-modal training
      3. Online fine-tuning
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.backbone_type = getattr(cfg, 'RESIDUAL_BACKBONE', 'mobilenet_v3_small')
        self.sensor_dim = getattr(cfg, 'SENSOR_DIM', 0)
        self.use_multi_modal = getattr(cfg, 'USE_MULTI_MODAL', False)

        self.agent = SACAgent(
            input_shape=(cfg.IMAGE_DEPTH, cfg.IMAGE_H, cfg.IMAGE_W),
            action_dim=1,
            hidden_dim=getattr(cfg, 'RESIDUAL_HIDDEN_DIM', 256),
            backbone_type=self.backbone_type,
            feature_dim=getattr(cfg, 'RESIDUAL_FEATURE_DIM', 50),
            freeze_backbone=getattr(cfg, 'RESIDUAL_FREEZE_BACKBONE', True),
            sensor_dim=self.sensor_dim if self.use_multi_modal else 0,
            sensor_feature_dim=getattr(cfg, 'SENSOR_FEATURE_DIM', 32),
            sensor_hidden_dim=getattr(cfg, 'SENSOR_HIDDEN_DIM', 64),
            lr_actor=getattr(cfg, 'RESIDUAL_LR_ACTOR', 3e-4),
            lr_critic=getattr(cfg, 'RESIDUAL_LR_CRITIC', 3e-4),
            gamma=getattr(cfg, 'RESIDUAL_GAMMA', 0.99),
            tau=getattr(cfg, 'RESIDUAL_TAU', 0.005),
        )

        effective_sensor_dim = self.sensor_dim if self.use_multi_modal else 0
        self.replay_buffer = ReplayBuffer(
            capacity=getattr(cfg, 'RESIDUAL_BUFFER_SIZE', 100000),
            sensor_dim=effective_sensor_dim)
        self.batch_size = getattr(cfg, 'RESIDUAL_BATCH_SIZE', 256)
        self.residual_scale = getattr(cfg, 'RESIDUAL_SCALE', 0.3)

    def load_tub_data(self, tub_paths, base_model_path=None):
        """Load DonkeyCar Tub data into replay buffer.

        For each record:
          1. Load image (store as float32 CHW [0,1])
          2. Optionally load sensor data from tub
          3. Compute residual = (human - base) / scale
          4. Push to replay buffer

        Sensor data loaded from: enc/speed, imu/acl_*, obs/distance, line/raw_values
        """
        from donkeycar.parts.datastore import Tub
        from donkeycar.utils import load_image

        base_model = None
        if base_model_path and os.path.exists(base_model_path):
            import donkeycar as dk
            base_model = dk.utils.get_model_by_type(
                self.cfg.DEFAULT_MODEL_TYPE, self.cfg)
            base_model.load(base_model_path)
            logger.info(f"Loaded base model from {base_model_path}")

        total_records = 0
        records_with_sensors = 0

        for tub_path in tub_paths:
            tub = Tub(tub_path)
            records = tub.get_records()
            logger.info(f"Loading {len(records)} records from {tub_path}")

            has_sensor_data = False
            if records and self.use_multi_modal:
                first_keys = list(records[0].keys())
                if any(k in first_keys for k in ['enc/speed', 'imu/acl_x', 'obs/distance']):
                    has_sensor_data = True
                    logger.info(f"  Tub has sensor data")

            for i, record in enumerate(records):
                try:
                    img_path = os.path.join(tub_path, record['cam/image_array'])
                    img_arr = load_image(img_path)
                    human_steering = float(record['user/angle'])

                    base_steering = 0.0
                    if base_model is not None:
                        base_steering, _ = base_model.run(img_arr)

                    residual = np.clip(
                        (human_steering - base_steering) / self.residual_scale, -1.0, 1.0)

                    img = img_arr.astype(np.float32) / 255.0
                    img = np.transpose(img, (2, 0, 1))

                    sensor_state = None
                    sensor_next = None
                    if has_sensor_data:
                        try:
                            sensor_state = self._build_sensor_from_record(record)
                            sensor_next = sensor_state.copy()
                            records_with_sensors += 1
                        except Exception:
                            sensor_state = None
                            sensor_next = None

                    self.replay_buffer.push(
                        img, np.array([residual], dtype=np.float32),
                        0.0, img, False,
                        sensor_state=sensor_state,
                        sensor_next_state=sensor_next)
                    total_records += 1
                except Exception as e:
                    logger.warning(f"Skipping record {i}: {e}")
                    continue

        logger.info(f"Loaded {total_records} total records into replay buffer")
        if records_with_sensors > 0:
            logger.info(f"  {records_with_sensors} records include sensor data")

    def _build_sensor_from_record(self, record):
        """Build sensor observation vector from Tub record."""
        obs = np.zeros(self.sensor_dim, dtype=np.float32)
        idx = 0

        if 'enc/speed' in record:
            obs[idx] = float(record['enc/speed'])
        idx += 1
        if 'enc/accel' in record:
            obs[idx] = float(record['enc/accel'])
        idx += 1

        for key in ['imu/acl_x', 'imu/acl_y', 'imu/acl_z',
                     'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z']:
            if key in record:
                obs[idx] = float(record[key])
            idx += 1

        if 'obs/distance' in record:
            obs[idx] = float(record['obs/distance'])
        elif 'lidar/dist' in record:
            obs[idx] = float(record['lidar/dist'])
        # idx + 1 → reaches 9

        return obs

    def train_offline(self, num_epochs=100, steps_per_epoch=1000):
        """Offline SAC training. Returns history dict."""
        history = {'critic_loss': [], 'actor_loss': [], 'alpha': []}

        for epoch in range(num_epochs):
            epoch_losses = {'critic_loss': 0, 'actor_loss': 0, 'alpha': 0}
            n_updates = 0

            for step in range(steps_per_epoch):
                if len(self.replay_buffer) < self.batch_size:
                    break
                losses = self.agent.update(self.replay_buffer, self.batch_size)
                if losses:
                    epoch_losses['critic_loss'] += losses['critic_loss']
                    epoch_losses['actor_loss'] += losses['actor_loss']
                    epoch_losses['alpha'] += losses['alpha']
                    n_updates += 1

            if n_updates == 0:
                logger.warning(
                    f"Epoch {epoch+1}: no updates "
                    f"(buffer {len(self.replay_buffer)} < batch {self.batch_size})")
                continue

            for k in epoch_losses:
                epoch_losses[k] /= n_updates
            for k in epoch_losses:
                history[k].append(epoch_losses[k])

            logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Critic: {epoch_losses['critic_loss']:.4f} | "
                f"Actor: {epoch_losses['actor_loss']:.4f} | "
                f"Alpha: {epoch_losses['alpha']:.4f}")

        return history

    def train_step(self, img_arr, action, reward, next_img_arr, done,
                   sensor_obs=None, sensor_obs_next=None):
        """Single online training step."""

        def preprocess(img):
            img = img.astype(np.float32) / 255.0
            return np.transpose(img, (2, 0, 1))

        self.replay_buffer.push(
            preprocess(img_arr),
            np.array([action], dtype=np.float32),
            reward,
            preprocess(next_img_arr),
            done,
            sensor_state=sensor_obs,
            sensor_next_state=sensor_obs_next)
        return self.agent.update(self.replay_buffer, self.batch_size)

    def save(self, path):
        self.agent.save(path)
