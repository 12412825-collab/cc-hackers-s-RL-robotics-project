"""A2 residual policy adaptation with frozen estimator / base controller.

Phase-0 primary A2 uses a low-dimensional linear residual policy so that
online sample budgets are comparable to A1 (estimator RLS). Full Residual SAC
remains available as an optional diagnostic backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .baseline import BASELINE
from .env import Phase0CorridorEnv


@dataclass
class LinearResidualAdapter:
    """Two-parameter residual: a = tanh(theta0 + theta1 * heading_feature).

    heading_feature is observation index 2 (normalized fused heading error).
    Updated by SGD on instantaneous cost: heading^2 + 0.01 a^2.
    """

    lr: float = 0.05
    theta: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    theta0: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    history: list[dict[str, float]] = field(default_factory=list)
    steps_seen: int = 0

    def reset(self, seed: int = 0) -> None:
        rng = np.random.RandomState(seed)
        self.theta = rng.normal(0.0, 0.01, size=2).astype(np.float64)
        self.theta0 = self.theta.copy()
        self.history.clear()
        self.steps_seen = 0

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> float:
        h = float(obs[2])
        z = self.theta[0] + self.theta[1] * h
        a = float(np.tanh(z))
        if not deterministic:
            a = float(np.clip(a + np.random.normal(0.0, 0.05), -1.0, 1.0))
        return a

    def observe(
        self,
        obs: np.ndarray,
        action: float,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        info: Optional[dict[str, Any]] = None,
    ) -> dict[str, float]:
        """SGD on J=0.5*h^2 using estimated heading feature (not privileged).

        Assumes residual acts in the same direction as the base P term
        (positive residual reduces positive heading), matching Phase-0 plant.
        """
        h = float(obs[2])
        z = self.theta[0] + self.theta[1] * h
        a_det = float(np.tanh(z))
        d_tanh = 1.0 - a_det * a_det
        # dJ/dtheta ≈ h * (dh/da) * da/dtheta with dh/da < 0 under stabilizing residual
        plant_gain = 0.3
        grad = np.array([h * d_tanh, h * d_tanh * h], dtype=np.float64)
        self.theta = self.theta + self.lr * plant_gain * grad
        # Mild weight decay keeps residual small unless needed
        self.theta *= 0.999
        self.steps_seen += 1
        rec = {
            "steps_seen": float(self.steps_seen),
            "theta0": float(self.theta[0]),
            "theta1": float(self.theta[1]),
            "weight_l2_delta": float(np.linalg.norm(self.theta - self.theta0)),
            "reward": float(reward),
        }
        self.history.append(rec)
        _ = (next_obs, done, info, action)
        return rec

    def freeze_env_estimator(self, env: Phase0CorridorEnv) -> None:
        if env._estimator_locked:
            env.unlock_estimator()
        env.set_estimator_params(
            imu_bias_hat=0.0,
            fusion_weight=env.cfg.fusion_weight_init,
        )
        env.lock_estimator()

    def parameter_magnitude(self) -> float:
        return float(np.linalg.norm(self.theta - self.theta0))

    def get_params(self) -> dict[str, float]:
        return {"theta0": float(self.theta[0]), "theta1": float(self.theta[1])}

    def set_params(self, theta0: float, theta1: float) -> None:
        self.theta = np.array([theta0, theta1], dtype=np.float64)


# Back-compat alias used by experiment runner
ResidualAdapter = LinearResidualAdapter


@dataclass
class SACResidualAdapter:
    """Optional Residual SAC backend (diagnostic; not Phase-0 primary)."""

    lr: float = BASELINE.residual_lr
    batch_size: int = BASELINE.residual_batch_size
    warmup: int = BASELINE.residual_warmup
    updates_per_step: int = BASELINE.residual_updates_per_step
    device: str = "cpu"
    agent: Any = None
    buffer: Any = None
    steps_seen: int = 0
    weight_l2_at_init: float = 0.0
    history: list[dict[str, float]] = field(default_factory=list)

    def reset(self, seed: int = 0) -> None:
        from library_residual.policy import LibrarySACAgent, SensorReplayBuffer, TORCH_AVAILABLE
        import torch

        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for SAC residual adaptation")
        torch.manual_seed(seed)
        self.agent = LibrarySACAgent(device=self.device)
        for opt in (self.agent.actor_optimizer, self.agent.critic_optimizer):
            for g in opt.param_groups:
                g["lr"] = self.lr
        self.buffer = SensorReplayBuffer(capacity=50_000)
        self.steps_seen = 0
        self.weight_l2_at_init = self._weight_l2()
        self.history.clear()

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> float:
        assert self.agent is not None
        if self.steps_seen < self.warmup and not deterministic:
            return float(np.random.uniform(-1.0, 1.0))
        return float(self.agent.select_action(obs, deterministic=deterministic))

    def observe(
        self,
        obs: np.ndarray,
        action: float,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        info: Optional[dict[str, Any]] = None,
    ) -> dict[str, float]:
        assert self.agent is not None and self.buffer is not None
        self.buffer.push(obs, np.array([action], dtype=np.float32), reward, next_obs, done)
        self.steps_seen += 1
        loss_info: dict[str, float] = {}
        if self.steps_seen >= self.warmup and len(self.buffer) >= self.batch_size:
            for _ in range(self.updates_per_step):
                loss_info = self.agent.update(self.buffer, self.batch_size) or {}
        rec = {
            "steps_seen": float(self.steps_seen),
            "weight_l2_delta": float(abs(self._weight_l2() - self.weight_l2_at_init)),
            **{k: float(v) for k, v in loss_info.items()},
        }
        self.history.append(rec)
        _ = info
        return rec

    def freeze_env_estimator(self, env: Phase0CorridorEnv) -> None:
        if env._estimator_locked:
            env.unlock_estimator()
        env.set_estimator_params(
            imu_bias_hat=0.0,
            fusion_weight=env.cfg.fusion_weight_init,
        )
        env.lock_estimator()

    def parameter_magnitude(self) -> float:
        return float(abs(self._weight_l2() - self.weight_l2_at_init))

    def _weight_l2(self) -> float:
        assert self.agent is not None
        import torch

        total = 0.0
        with torch.no_grad():
            for p in self.agent.actor.parameters():
                total += float(torch.sum(p.detach() ** 2).item())
        return float(np.sqrt(total))
