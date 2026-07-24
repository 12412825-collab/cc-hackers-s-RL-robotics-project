"""Sensor-only MLP SAC for the Library Robot residual heading correction.

Architecture (default 64×64):

    Actor:   5 → 64 → ReLU → 64 → ReLU → (mean 1, log_std 1) → tanh
    Critic:  (5+1) → 64 → ReLU → 64 → ReLU → Q(1)   (twin)

The action is a single normalised residual value in [-1, 1].

This module deliberately does NOT import torchvision, does NOT build a CNN,
and does NOT require a camera image.  It is designed to run on a Raspberry Pi
CPU at 5–10 Hz inference.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Normal

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

    class _DummyModule:
        pass

    class _DummyNN:
        Module = _DummyModule

    nn = _DummyNN()

from .types import ACTION_DIM, OBSERVATION_DIM

logger = logging.getLogger(__name__)


# ========================================================================
# Actor
# ========================================================================

class LibrarySACActor(nn.Module):
    """Gaussian policy: observation → (mean, log_std) → tanh-squashed action."""

    def __init__(
        self,
        obs_dim: int = OBSERVATION_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = 64,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.mean_head(x)
        log_std = torch.clamp(
            self.log_std_head(x), self.log_std_min, self.log_std_max
        )
        return mean, log_std

    def sample(self, obs: torch.Tensor):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(obs)
        return torch.tanh(mean)


# ========================================================================
# Critic (Twin-Q)
# ========================================================================

class LibrarySACCritic(nn.Module):
    """Twin Q-network: (observation, action) → (Q1, Q2)."""

    def __init__(
        self,
        obs_dim: int = OBSERVATION_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = 64,
    ):
        super().__init__()
        in_dim = obs_dim + action_dim
        self.fc1_1 = nn.Linear(in_dim, hidden_dim)
        self.fc1_2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1 = nn.Linear(hidden_dim, 1)

        self.fc2_1 = nn.Linear(in_dim, hidden_dim)
        self.fc2_2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2 = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        x = torch.cat([obs, action], dim=-1)
        q1 = F.relu(self.fc1_1(x))
        q1 = F.relu(self.fc1_2(q1))
        q1 = self.q1(q1)
        q2 = F.relu(self.fc2_1(x))
        q2 = F.relu(self.fc2_2(q2))
        q2 = self.q2(q2)
        return q1, q2


# ========================================================================
# Replay Buffer (sensor-only, no images)
# ========================================================================

class SensorReplayBuffer:
    """Flat replay buffer storing (obs, action, reward, next_obs, done)."""

    def __init__(self, capacity: int = 100_000, obs_dim: int = OBSERVATION_DIM):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, ACTION_DIM), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        idx = self.ptr % self.capacity
        self.obs[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_obs[idx] = next_obs
        self.dones[idx] = float(done)
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.obs[indices]),
            torch.FloatTensor(self.actions[indices]),
            torch.FloatTensor(self.rewards[indices]),
            torch.FloatTensor(self.next_obs[indices]),
            torch.FloatTensor(self.dones[indices]),
        )

    def __len__(self) -> int:
        return self.size


# ========================================================================
# SAC Agent (sensor-only, CPU)
# ========================================================================

class LibrarySACAgent:
    """Sensor-only SAC agent for the Library Robot residual correction.

    Key differences from the visual SACAgent in ``parts/residual_rl.py``:
      • No image encoder, no CNN, no torchvision dependency.
      • Input is a 5-dim normalised sensor vector.
      • Output is a 1-dim tanh-squashed residual action.
      • Designed for CPU inference on Raspberry Pi.
    """

    def __init__(
        self,
        obs_dim: int = OBSERVATION_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = 64,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        target_entropy: Optional[float] = None,
        device: Optional[str] = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for LibrarySACAgent. "
                "Install: pip install torch"
            )

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device or "cpu")

        # Networks
        self.actor = LibrarySACActor(obs_dim, action_dim, hidden_dim).to(
            self.device
        )
        self.critic = LibrarySACCritic(obs_dim, action_dim, hidden_dim).to(
            self.device
        )
        self.critic_target = LibrarySACCritic(
            obs_dim, action_dim, hidden_dim
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimisers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=lr_critic
        )

        # Auto-entropy tuning
        self.target_entropy = (
            target_entropy if target_entropy is not None else -float(action_dim)
        )
        self.log_alpha = torch.zeros(
            1, requires_grad=True, device=self.device
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
        self.alpha = self.log_alpha.exp().item()

        total_params = sum(
            p.numel()
            for net in (self.actor, self.critic)
            for p in net.parameters()
        )
        logger.info(
            "LibrarySACAgent: obs=%d action=%d hidden=%d params=%d device=%s",
            obs_dim,
            action_dim,
            hidden_dim,
            total_params,
            self.device,
        )

    # ----- action selection -----

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
        image: object = None,  # accepted but ignored for API compat
    ) -> float:
        """Select a residual action from a normalised observation vector.

        ``image`` is accepted to satisfy the interface contract but is
        always ignored — no visual encoder is constructed.
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic_action(obs_t)
            else:
                action, _ = self.actor.sample(obs_t)
        return float(action.cpu().squeeze())

    # ----- SAC update -----

    def update(self, buffer: SensorReplayBuffer, batch_size: int = 256):
        if len(buffer) < batch_size:
            return {}

        obs, actions, rewards, next_obs, dones = buffer.sample(batch_size)
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_obs = next_obs.to(self.device)
        dones = dones.to(self.device)

        # Critic update
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_obs)
            q1_next, q2_next = self.critic_target(next_obs, next_actions)
            q_next = (
                torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            )
            q_target = rewards + self.gamma * (1.0 - dones) * q_next

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        new_actions, log_probs = self.actor.sample(obs)
        q1_new, q2_new = self.critic(obs, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha update
        alpha_loss = -(
            self.log_alpha * (log_probs.detach() + self.target_entropy)
        ).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        # Soft target update
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha,
        }

    # ----- save / load -----

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "log_alpha": self.log_alpha.detach(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
            },
            path,
        )
        logger.info("Saved LibrarySACAgent to %s", path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        if "log_alpha" in checkpoint:
            self.log_alpha = checkpoint["log_alpha"].to(self.device).requires_grad_(True)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        if "actor_optimizer" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        if "critic_optimizer" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        logger.info("Loaded LibrarySACAgent from %s", path)
