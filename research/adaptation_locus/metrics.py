"""Phase-0 metrics including Performance Recovery Ratio."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np


EPS = 1e-6


@dataclass
class EpisodeMetrics:
    success: bool
    reason: str
    steps: int
    mean_heading_abs: float
    final_heading_abs: float
    mean_lateral_abs: float
    final_lateral_abs: float
    cumulative_control_effort: float
    mean_residual_abs: float
    rms_residual: float
    oscillation_count: int
    episode_return: float
    param_magnitude: float = 0.0

    @property
    def performance_score(self) -> float:
        """Higher is better; independent of RL reward shaping."""
        return -(self.mean_heading_abs + 0.1 * self.mean_lateral_abs)


@dataclass
class EvalAggregate:
    n: int
    success_rate: float
    mean_heading_abs: float
    final_heading_abs: float
    mean_lateral_abs: float
    cumulative_control_effort: float
    mean_residual_abs: float
    oscillation_count: float
    mean_return: float
    performance_score: float
    episodes: list[EpisodeMetrics] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def performance_recovery_ratio(
    adapted: float,
    nominal: float,
    shifted_unadapted: float,
) -> float:
    denom = nominal - shifted_unadapted
    if abs(denom) < EPS:
        return 0.0
    return float((adapted - shifted_unadapted) / (denom + EPS))


def aggregate_episodes(episodes: list[EpisodeMetrics]) -> EvalAggregate:
    if not episodes:
        return EvalAggregate(
            n=0,
            success_rate=0.0,
            mean_heading_abs=0.0,
            final_heading_abs=0.0,
            mean_lateral_abs=0.0,
            cumulative_control_effort=0.0,
            mean_residual_abs=0.0,
            oscillation_count=0.0,
            mean_return=0.0,
            performance_score=0.0,
            episodes=[],
        )
    return EvalAggregate(
        n=len(episodes),
        success_rate=float(np.mean([e.success for e in episodes])),
        mean_heading_abs=float(np.mean([e.mean_heading_abs for e in episodes])),
        final_heading_abs=float(np.mean([e.final_heading_abs for e in episodes])),
        mean_lateral_abs=float(np.mean([e.mean_lateral_abs for e in episodes])),
        cumulative_control_effort=float(
            np.mean([e.cumulative_control_effort for e in episodes])
        ),
        mean_residual_abs=float(np.mean([e.mean_residual_abs for e in episodes])),
        oscillation_count=float(np.mean([e.oscillation_count for e in episodes])),
        mean_return=float(np.mean([e.episode_return for e in episodes])),
        performance_score=float(np.mean([e.performance_score for e in episodes])),
        episodes=episodes,
    )


def samples_to_recovery(
    timeline: list[dict[str, Any]],
    *,
    nominal_score: float,
    shifted_score: float,
    threshold: float = 0.8,
) -> Optional[int]:
    for row in timeline:
        prr = performance_recovery_ratio(
            row["performance_score"], nominal_score, shifted_score
        )
        if prr >= threshold:
            return int(row["online_steps"])
    return None
