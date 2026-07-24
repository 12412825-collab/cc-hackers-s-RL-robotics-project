"""Safe residual policy wrapper and action-to-PWM conversion.

This module provides the public inference API that Library_robot uses.
All failure modes return a zero residual — the base controller is never
overridden.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Optional

import numpy as np

from .observation import LibraryObservationV1
from .types import (
    ACTION_DIM,
    FEATURE_NAMES,
    OBSERVATION_DIM,
    SCHEMA_VERSION,
    ResidualResult,
    ULTRASONIC_INVALID_CM,
)

logger = logging.getLogger(__name__)

# Valid inference modes
VALID_MODES = {"disabled", "shadow", "active"}
# Actions during which residual inference is permitted
ALLOWED_ACTIONS = {"FORWARD", "BACKWARD"}

# ======================================================================
# Pure function: residual action → motor PWM
# ======================================================================


def residual_action_to_pwm(
    action: float,
    max_residual_pwm: int = 10,
) -> int:
    """Convert a normalised residual action in [-1, 1] to an integer PWM.

    Requirements:
      • Validates finite input.
      • Clips action to [-1, 1].
      • Returns an integer.
      • Clips output to [-max_residual_pwm, max_residual_pwm].

    >>> residual_action_to_pwm(0.5, max_residual_pwm=10)
    5
    >>> residual_action_to_pwm(1.5, max_residual_pwm=10)
    10
    >>> residual_action_to_pwm(float('nan'), max_residual_pwm=10)
    0
    """
    if not math.isfinite(action):
        return 0
    if max_residual_pwm < 5 or max_residual_pwm > 15:
        raise ValueError(
            f"max_residual_pwm must be between 5 and 15, got {max_residual_pwm}"
        )
    clipped_action = max(-1.0, min(1.0, action))
    raw_pwm = round(clipped_action * max_residual_pwm)
    return max(-max_residual_pwm, min(max_residual_pwm, raw_pwm))


# ======================================================================
# Zero result factory
# ======================================================================

_ZERO = ResidualResult(
    normalized_action=0.0,
    residual_pwm=0,
    valid=False,
    apply_to_motor=False,
    latency_ms=0.0,
    reason="disabled",
)


def _zero_result(reason: str, latency_ms: float = 0.0) -> ResidualResult:
    return ResidualResult(
        normalized_action=0.0,
        residual_pwm=0,
        valid=False,
        apply_to_motor=False,
        latency_ms=latency_ms,
        reason=reason,
    )


# ======================================================================
# SafeResidualPolicy
# ======================================================================


class SafeResidualPolicy:
    """Thread-safe wrapper that loads a TorchScript actor and returns
    bounded residual corrections with comprehensive safety checks.

    Modes:
      disabled — always returns residual_pwm=0
      shadow   — calculates and logs the recommendation; apply_to_motor=False
      active   — authorises the bounded residual when all safety checks pass
    """

    def __init__(
        self,
        mode: str = "disabled",
        max_residual_pwm: int = 10,
        deadline_ms: float = 50.0,
        observation_max_age_ms: float = 250.0,
    ):
        if mode not in VALID_MODES:
            logger.warning(
                "Invalid RL mode '%s', falling back to 'disabled'", mode
            )
            mode = "disabled"

        self.mode = mode
        self.max_residual_pwm = max_residual_pwm
        self.deadline_ms = deadline_ms
        self.observation_max_age_ms = observation_max_age_ms
        self._model = None
        self._manifest: dict | None = None
        self._loaded = False

    # ---- loading ----

    @classmethod
    def load(
        cls,
        model_directory: str,
        mode: str = "shadow",
        max_residual_pwm: int = 10,
        deadline_ms: float = 50.0,
        observation_max_age_ms: float = 250.0,
    ) -> "SafeResidualPolicy":
        """Load the exported actor bundle from *model_directory*."""
        policy = cls(
            mode=mode,
            max_residual_pwm=max_residual_pwm,
            deadline_ms=deadline_ms,
            observation_max_age_ms=observation_max_age_ms,
        )
        if mode == "disabled":
            logger.info("SafeResidualPolicy: mode=disabled, skipping load")
            return policy

        try:
            import torch  # local import for optional dependency

            actor_path = os.path.join(model_directory, "actor.ts")
            manifest_path = os.path.join(model_directory, "manifest.json")

            if not os.path.isfile(actor_path):
                logger.warning(
                    "Actor model not found at %s; RL disabled", actor_path
                )
                policy.mode = "disabled"
                return policy

            policy._model = torch.jit.load(actor_path, map_location="cpu")
            policy._model.eval()

            if os.path.isfile(manifest_path):
                with open(manifest_path, "r") as f:
                    policy._manifest = json.load(f)
                # Validate schema version
                manifest_schema = policy._manifest.get("schema_version", "")
                if manifest_schema != SCHEMA_VERSION:
                    logger.warning(
                        "Manifest schema '%s' != expected '%s'",
                        manifest_schema,
                        SCHEMA_VERSION,
                    )

            policy._loaded = True
            logger.info(
                "SafeResidualPolicy loaded from %s (mode=%s)",
                model_directory,
                mode,
            )
        except Exception as exc:
            logger.error("Failed to load RL model: %s", exc)
            policy.mode = "disabled"
            policy._loaded = False

        return policy

    # ---- prediction ----

    def predict(
        self,
        observation: LibraryObservationV1,
        current_action: str = "FORWARD",
        timestamp: Optional[float] = None,
    ) -> ResidualResult:
        """Run the actor on *observation* and return a bounded result.

        Returns zero residual for every failure case.
        """
        t0 = time.monotonic()

        # Mode check
        if self.mode == "disabled":
            return _zero_result("disabled")

        # Action check
        if current_action not in ALLOWED_ACTIONS:
            return _zero_result(f"action_not_allowed:{current_action}")

        # Model check
        if self._model is None or not self._loaded:
            return _zero_result("model_not_loaded")

        # Observation dimension check
        try:
            obs_array = observation.to_numpy()
        except Exception:
            return _zero_result("observation_conversion_failed")

        if obs_array.shape[0] != OBSERVATION_DIM:
            return _zero_result(
                f"observation_dim_mismatch:{obs_array.shape[0]}"
            )

        # Finite-value check
        if not np.all(np.isfinite(obs_array)):
            return _zero_result("observation_contains_nan_or_inf")

        # Ultrasonic validity
        if not observation.has_valid_ultrasonic():
            return _zero_result("ultrasonic_invalid")

        # Staleness check
        if timestamp is not None:
            age_ms = (time.monotonic() - timestamp) * 1000.0
            if age_ms > self.observation_max_age_ms:
                return _zero_result(f"observation_stale:{age_ms:.0f}ms")

        # Inference
        try:
            import torch

            obs_tensor = torch.FloatTensor(obs_array).unsqueeze(0)
            with torch.no_grad():
                raw_action = self._model(obs_tensor)

            elapsed_ms = (time.monotonic() - t0) * 1000.0

            # Deadline check
            if elapsed_ms > self.deadline_ms:
                return _zero_result(
                    f"inference_exceeded_deadline:{elapsed_ms:.1f}ms",
                    latency_ms=elapsed_ms,
                )

            action_value = float(raw_action.squeeze())

            # Output validation
            if not math.isfinite(action_value):
                return _zero_result("model_output_nan_or_inf", latency_ms=elapsed_ms)

            # Compute PWM
            pwm = residual_action_to_pwm(action_value, self.max_residual_pwm)
            apply = self.mode == "active"

            return ResidualResult(
                normalized_action=max(-1.0, min(1.0, action_value)),
                residual_pwm=pwm,
                valid=True,
                apply_to_motor=apply,
                latency_ms=elapsed_ms,
                reason="ok" if apply else "shadow_mode",
            )

        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            logger.error("RL inference error: %s", exc)
            return _zero_result(
                f"inference_exception:{type(exc).__name__}",
                latency_ms=elapsed_ms,
            )

    # ---- status ----

    @property
    def model_loaded(self) -> bool:
        return self._loaded

    @property
    def schema_version(self) -> str:
        return SCHEMA_VERSION

    def status_dict(self) -> dict:
        return {
            "rl_mode": self.mode,
            "rl_model_loaded": self._loaded,
            "rl_schema_version": SCHEMA_VERSION,
            "rl_max_residual_pwm": self.max_residual_pwm,
        }
