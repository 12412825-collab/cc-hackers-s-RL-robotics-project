"""Five-dimensional observation for the Library Robot Residual SAC.

Schema version: library-observation-v1

Feature order (float32):
    0. motion_direction          +1.0 FORWARD, -1.0 BACKWARD
    1. segment_progress          completed / target distance, clipped [0, 1]
    2. fused_heading_error        target − fused heading (degrees, wrapped)
    3. left_right_encoder_error   left − right wheel distance (cm)
    4. front_ultrasonic_distance  forward HC-SR04 reading (cm, clipped)

This observation intentionally excludes camera data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .types import (
    FEATURE_CLIP,
    FEATURE_NAMES,
    FEATURE_NORM,
    OBSERVATION_DIM,
    SCHEMA_VERSION,
    ULTRASONIC_INVALID_CM,
)


def _wrap_degrees(angle: float) -> float:
    """Wrap an angle in degrees to the (-180, +180] range."""
    return float((angle + 180.0) % 360.0 - 180.0)


@dataclass
class LibraryObservationV1:
    """Immutable five-dimensional observation for library-observation-v1."""

    motion_direction: float
    segment_progress: float
    fused_heading_error: float
    left_right_encoder_error: float
    front_ultrasonic_distance: float

    # --- Class constants ---
    SCHEMA_VERSION: str = SCHEMA_VERSION
    DIM: int = OBSERVATION_DIM
    FEATURE_NAMES: tuple[str, ...] = FEATURE_NAMES

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_navigation_state(
        cls,
        *,
        is_forward: bool,
        completed_distance: float,
        target_distance: float,
        target_heading_deg: float,
        fused_heading_deg: float,
        left_distance_cm: float,
        right_distance_cm: float,
        front_ultrasonic_cm: float,
        ultrasonic_max_cm: float = 400.0,
    ) -> "LibraryObservationV1":
        """Build an observation from raw navigation telemetry."""
        direction = 1.0 if is_forward else -1.0

        if target_distance > 0:
            progress = min(1.0, max(0.0, completed_distance / target_distance))
        else:
            progress = 0.0

        heading_error = _wrap_degrees(target_heading_deg - fused_heading_deg)
        encoder_error = left_distance_cm - right_distance_cm

        us_cm = front_ultrasonic_cm
        if us_cm < 0 or not math.isfinite(us_cm):
            us_cm = ULTRASONIC_INVALID_CM
        else:
            us_cm = min(us_cm, ultrasonic_max_cm)

        return cls(
            motion_direction=direction,
            segment_progress=progress,
            fused_heading_error=heading_error,
            left_right_encoder_error=encoder_error,
            front_ultrasonic_distance=us_cm,
        )

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "LibraryObservationV1":
        """Construct from a flat float32 array of length 5."""
        arr = np.asarray(arr, dtype=np.float32).ravel()
        if arr.shape[0] != OBSERVATION_DIM:
            raise ValueError(
                f"Expected {OBSERVATION_DIM} features, got {arr.shape[0]}"
            )
        return cls(
            motion_direction=float(arr[0]),
            segment_progress=float(arr[1]),
            fused_heading_error=float(arr[2]),
            left_right_encoder_error=float(arr[3]),
            front_ultrasonic_distance=float(arr[4]),
        )

    @classmethod
    def from_dict(cls, data: dict) -> "LibraryObservationV1":
        """Construct from a dictionary with feature-name keys."""
        values = []
        for name in FEATURE_NAMES:
            if name not in data:
                raise KeyError(f"Missing feature: {name}")
            values.append(float(data[name]))
        return cls(*values)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty = valid)."""
        errors: list[str] = []
        values = self.to_tuple()
        if len(values) != OBSERVATION_DIM:
            errors.append(f"dimension mismatch: {len(values)} != {OBSERVATION_DIM}")
        for i, (name, val) in enumerate(zip(FEATURE_NAMES, values)):
            if not math.isfinite(val):
                errors.append(f"{name}[{i}] is not finite: {val}")
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0

    def has_valid_ultrasonic(self) -> bool:
        """Return False when the ultrasonic reading is the sentinel value."""
        return (
            self.front_ultrasonic_distance != ULTRASONIC_INVALID_CM
            and math.isfinite(self.front_ultrasonic_distance)
            and self.front_ultrasonic_distance >= 0
        )

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_tuple(self) -> tuple[float, ...]:
        return (
            self.motion_direction,
            self.segment_progress,
            self.fused_heading_error,
            self.left_right_encoder_error,
            self.front_ultrasonic_distance,
        )

    def to_numpy(self, dtype=np.float32) -> np.ndarray:
        """Return a 1-D float32 array of length 5."""
        return np.array(self.to_tuple(), dtype=dtype)

    def clip(self) -> "LibraryObservationV1":
        """Return a new observation with each feature clipped to its defined range."""
        values = list(self.to_tuple())
        for i, name in enumerate(FEATURE_NAMES):
            lo, hi = FEATURE_CLIP[name]
            values[i] = max(lo, min(hi, values[i]))
        return LibraryObservationV1(*values)

    def normalize(
        self,
        norm_ranges: Optional[dict[str, tuple[float, float]]] = None,
    ) -> np.ndarray:
        """Normalise each feature to [-1, 1] and return as float32 array.

        *norm_ranges* overrides the default per-feature normalisation ranges.
        Values outside the range are clamped to [-1, 1].
        """
        ranges = norm_ranges or FEATURE_NORM
        clipped = self.clip()
        values = clipped.to_tuple()
        normed = np.empty(OBSERVATION_DIM, dtype=np.float32)
        for i, name in enumerate(FEATURE_NAMES):
            lo, hi = ranges.get(name, FEATURE_CLIP[name])
            span = hi - lo
            if span == 0:
                normed[i] = 0.0
            else:
                normed[i] = np.clip(2.0 * (values[i] - lo) / span - 1.0, -1.0, 1.0)
        return normed

    def validate_schema(self, expected_version: str = SCHEMA_VERSION) -> None:
        """Raise if the schema version does not match."""
        if self.SCHEMA_VERSION != expected_version:
            raise ValueError(
                f"Schema mismatch: observation is {self.SCHEMA_VERSION}, "
                f"expected {expected_version}"
            )
