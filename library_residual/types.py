"""Core types and constants for the Library Robot Residual SAC system.

Schema: library-observation-v1
Action:  one normalized residual heading correction in [-1, 1]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

SCHEMA_VERSION = "library-observation-v1"
OBSERVATION_DIM = 5
ACTION_DIM = 1

FEATURE_NAMES: tuple[str, ...] = (
    "motion_direction",
    "segment_progress",
    "fused_heading_error",
    "left_right_encoder_error",
    "front_ultrasonic_distance",
)

FEATURE_UNITS: dict[str, str] = {
    "motion_direction": "unitless (+1 forward, -1 backward)",
    "segment_progress": "ratio [0, 1]",
    "fused_heading_error": "degrees (wrapped to [-180, 180])",
    "left_right_encoder_error": "centimetres (left - right)",
    "front_ultrasonic_distance": "centimetres (clipped to max)",
}

# Default clipping limits per feature.
FEATURE_CLIP: dict[str, tuple[float, float]] = {
    "motion_direction": (-1.0, 1.0),
    "segment_progress": (0.0, 1.0),
    "fused_heading_error": (-180.0, 180.0),
    "left_right_encoder_error": (-50.0, 50.0),
    "front_ultrasonic_distance": (0.0, 400.0),
}

# Default normalisation ranges (maps clipped range to [-1, 1]).
FEATURE_NORM: dict[str, tuple[float, float]] = {
    "motion_direction": (-1.0, 1.0),
    "segment_progress": (0.0, 1.0),
    "fused_heading_error": (-45.0, 45.0),
    "left_right_encoder_error": (-10.0, 10.0),
    "front_ultrasonic_distance": (0.0, 400.0),
}

# Value that represents an invalid/timeout ultrasonic reading.
ULTRASONIC_INVALID_CM: float = -1.0


@dataclass(frozen=True)
class ResidualResult:
    """Outcome of a single SAC residual inference call."""

    normalized_action: float = 0.0
    residual_pwm: int = 0
    valid: bool = False
    apply_to_motor: bool = False
    latency_ms: float = 0.0
    reason: str = "not_computed"
