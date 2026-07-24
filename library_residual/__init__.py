"""Library Robot Residual SAC — public API.

Usage:
    from library_residual import (
        LibraryObservationV1,
        SafeResidualPolicy,
        ResidualResult,
        residual_action_to_pwm,
    )
"""

from .observation import LibraryObservationV1
from .safety import SafeResidualPolicy, residual_action_to_pwm
from .types import (
    ACTION_DIM,
    FEATURE_NAMES,
    FEATURE_UNITS,
    OBSERVATION_DIM,
    SCHEMA_VERSION,
    ResidualResult,
)

__all__ = [
    "LibraryObservationV1",
    "SafeResidualPolicy",
    "ResidualResult",
    "residual_action_to_pwm",
    "ACTION_DIM",
    "FEATURE_NAMES",
    "FEATURE_UNITS",
    "OBSERVATION_DIM",
    "SCHEMA_VERSION",
]
