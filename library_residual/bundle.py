"""Model export helpers for the Library Robot SAC bundle.

Exports to a model directory:
    actor.ts            — TorchScript deterministic actor (inference only)
    checkpoint.pth      — full training checkpoint
    manifest.json       — schema version, feature metadata, training info
    normalization.json  — per-feature normalisation statistics
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

import numpy as np

from .types import (
    ACTION_DIM,
    FEATURE_CLIP,
    FEATURE_NAMES,
    FEATURE_NORM,
    FEATURE_UNITS,
    OBSERVATION_DIM,
    SCHEMA_VERSION,
)

logger = logging.getLogger(__name__)


def export_actor_torchscript(
    actor,
    output_path: str,
    obs_dim: int = OBSERVATION_DIM,
) -> str:
    """Export a deterministic actor wrapper to TorchScript.

    The exported module accepts (B, obs_dim) and returns tanh(mean).
    Critics and optimisers are NOT included.
    """
    import torch
    import torch.nn as nn

    class DeterministicActor(nn.Module):
        def __init__(self, actor_module):
            super().__init__()
            self.fc1 = actor_module.fc1
            self.fc2 = actor_module.fc2
            self.mean_head = actor_module.mean_head

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            x = torch.relu(self.fc1(obs))
            x = torch.relu(self.fc2(x))
            return torch.tanh(self.mean_head(x))

    wrapper = DeterministicActor(actor)
    wrapper.eval()

    example_input = torch.zeros(1, obs_dim)
    scripted = torch.jit.trace(wrapper, example_input)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    scripted.save(output_path)
    logger.info("Exported TorchScript actor to %s", output_path)
    return output_path


def write_manifest(
    output_path: str,
    *,
    model_version: str = "1.0.0",
    max_residual_pwm: int = 10,
    training_hz: float = 10.0,
    inference_hz: float = 5.0,
    torch_version: Optional[str] = None,
    extra: Optional[dict] = None,
) -> str:
    """Write manifest.json with all required metadata."""
    import torch as _torch

    manifest = {
        "model_version": model_version,
        "schema_version": SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "feature_units": FEATURE_UNITS,
        "feature_clip": {k: list(v) for k, v in FEATURE_CLIP.items()},
        "input_dimension": OBSERVATION_DIM,
        "action_dimension": ACTION_DIM,
        "action_type": "normalized_residual_heading_correction",
        "max_residual_pwm": max_residual_pwm,
        "training_control_frequency_hz": training_hz,
        "expected_inference_frequency_hz": inference_hz,
        "pytorch_version": torch_version or _torch.__version__,
        "creation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        manifest.update(extra)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote manifest to %s", output_path)
    return output_path


def write_normalization(
    output_path: str,
    norm_ranges: Optional[dict] = None,
) -> str:
    """Write normalization.json with per-feature scaling ranges."""
    ranges = norm_ranges or FEATURE_NORM
    data = {
        "schema_version": SCHEMA_VERSION,
        "normalization": {
            name: {"min": lo, "max": hi}
            for name, (lo, hi) in ranges.items()
        },
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Wrote normalization to %s", output_path)
    return output_path


def export_bundle(
    agent,
    output_dir: str,
    *,
    max_residual_pwm: int = 10,
    model_version: str = "1.0.0",
    training_hz: float = 10.0,
    inference_hz: float = 5.0,
) -> dict[str, str]:
    """Export a complete model bundle ready for deployment."""
    os.makedirs(output_dir, exist_ok=True)

    paths = {}

    # Actor TorchScript
    actor_path = os.path.join(output_dir, "actor.ts")
    export_actor_torchscript(agent.actor, actor_path, agent.obs_dim)
    paths["actor"] = actor_path

    # Full checkpoint
    checkpoint_path = os.path.join(output_dir, "checkpoint.pth")
    agent.save(checkpoint_path)
    paths["checkpoint"] = checkpoint_path

    # Manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    write_manifest(
        manifest_path,
        model_version=model_version,
        max_residual_pwm=max_residual_pwm,
        training_hz=training_hz,
        inference_hz=inference_hz,
    )
    paths["manifest"] = manifest_path

    # Normalization
    norm_path = os.path.join(output_dir, "normalization.json")
    write_normalization(norm_path)
    paths["normalization"] = norm_path

    logger.info("Exported complete bundle to %s", output_dir)
    return paths
