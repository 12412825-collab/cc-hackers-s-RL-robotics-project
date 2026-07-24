"""Tests for the library_residual SAC package in cc-hackers-s-RL-robotics-project."""

import json
import math
import os
import tempfile
import numpy as np
import pytest

from library_residual.types import (
    ACTION_DIM,
    FEATURE_NAMES,
    OBSERVATION_DIM,
    SCHEMA_VERSION,
    ULTRASONIC_INVALID_CM,
    ResidualResult,
)
from library_residual.observation import LibraryObservationV1
from library_residual.policy import (
    LibrarySACActor,
    LibrarySACCritic,
    LibrarySACAgent,
    SensorReplayBuffer,
)
from library_residual.safety import SafeResidualPolicy, residual_action_to_pwm
from library_residual.bundle import export_bundle
from library_residual.env import CorridorConfig, LibraryCorridorEnv


# ======================================================================
# Observation Tests
# ======================================================================

def test_observation_dimensions_and_defaults():
    obs = LibraryObservationV1.from_navigation_state(
        is_forward=True,
        completed_distance=50.0,
        target_distance=100.0,
        target_heading_deg=0.0,
        fused_heading_deg=5.0,
        left_distance_cm=50.5,
        right_distance_cm=50.0,
        front_ultrasonic_cm=150.0,
    )
    arr = obs.to_numpy()
    assert arr.shape == (5,)
    assert obs.motion_direction == 1.0
    assert obs.segment_progress == 0.5
    assert obs.fused_heading_error == -5.0
    assert abs(obs.left_right_encoder_error - 0.5) < 1e-4
    assert obs.front_ultrasonic_distance == 150.0
    assert obs.is_valid()


def test_observation_validation_and_clipping():
    # Negative ultrasonic sentinel
    obs = LibraryObservationV1.from_navigation_state(
        is_forward=False,
        completed_distance=0.0,
        target_distance=0.0,
        target_heading_deg=180.0,
        fused_heading_deg=-190.0,
        left_distance_cm=0.0,
        right_distance_cm=0.0,
        front_ultrasonic_cm=-1.0,
    )
    assert obs.motion_direction == -1.0
    assert obs.has_valid_ultrasonic() is False

    # Clipping test
    obs_extreme = LibraryObservationV1(
        motion_direction=5.0,
        segment_progress=2.0,
        fused_heading_error=300.0,
        left_right_encoder_error=100.0,
        front_ultrasonic_distance=1000.0,
    )
    clipped = obs_extreme.clip()
    assert clipped.motion_direction == 1.0
    assert clipped.segment_progress == 1.0
    assert clipped.fused_heading_error == 180.0
    assert clipped.left_right_encoder_error == 50.0
    assert clipped.front_ultrasonic_distance == 400.0


def test_observation_normalization_range():
    obs = LibraryObservationV1(
        motion_direction=1.0,
        segment_progress=0.5,
        fused_heading_error=0.0,
        left_right_encoder_error=0.0,
        front_ultrasonic_distance=200.0,
    )
    normed = obs.normalize()
    assert normed.shape == (5,)
    assert np.all(normed >= -1.0) and np.all(normed <= 1.0)


# ======================================================================
# Policy & Agent Tests
# ======================================================================

def test_actor_critic_shape():
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed")

    actor = LibrarySACActor(obs_dim=5, action_dim=1, hidden_dim=64)
    critic = LibrarySACCritic(obs_dim=5, action_dim=1, hidden_dim=64)

    obs = torch.randn(4, 5)
    mean, log_std = actor(obs)
    assert mean.shape == (4, 1)
    assert log_std.shape == (4, 1)

    act = torch.randn(4, 1)
    q1, q2 = critic(obs, act)
    assert q1.shape == (4, 1)
    assert q2.shape == (4, 1)


def test_agent_sensor_only_inference():
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed")

    agent = LibrarySACAgent(hidden_dim=32, device="cpu")
    obs = np.array([1.0, 0.5, -2.0, 0.1, 150.0], dtype=np.float32)

    # deterministic prediction
    act1 = agent.select_action(obs, deterministic=True, image=None)
    assert isinstance(act1, float)
    assert -1.0 <= act1 <= 1.0

    # stochastic sample
    act2 = agent.select_action(obs, deterministic=False, image=None)
    assert isinstance(act2, float)
    assert -1.0 <= act2 <= 1.0


# ======================================================================
# PWM Conversion & Safety Policy Tests
# ======================================================================

def test_residual_action_to_pwm_conversion():
    assert residual_action_to_pwm(0.0, max_residual_pwm=10) == 0
    assert residual_action_to_pwm(0.5, max_residual_pwm=10) == 5
    assert residual_action_to_pwm(-1.0, max_residual_pwm=10) == -10
    assert residual_action_to_pwm(1.5, max_residual_pwm=10) == 10
    assert residual_action_to_pwm(math.nan, max_residual_pwm=10) == 0


def test_safe_residual_policy_modes():
    policy = SafeResidualPolicy(mode="disabled")
    obs = LibraryObservationV1(1.0, 0.5, 0.0, 0.0, 100.0)

    res = policy.predict(obs, current_action="FORWARD")
    assert res.valid is False
    assert res.apply_to_motor is False
    assert res.residual_pwm == 0
    assert res.reason == "disabled"


def test_safe_residual_policy_bundle_export_and_inference():
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed")

    agent = LibrarySACAgent(hidden_dim=32, device="cpu")
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = export_bundle(agent, tmpdir, max_residual_pwm=10)
        assert os.path.isfile(paths["actor"])
        assert os.path.isfile(paths["manifest"])

        # Test shadow mode
        shadow_policy = SafeResidualPolicy.load(tmpdir, mode="shadow", max_residual_pwm=10)
        obs = LibraryObservationV1(1.0, 0.5, 0.0, 0.0, 100.0)
        res_shadow = shadow_policy.predict(obs, current_action="FORWARD")
        assert res_shadow.valid is True
        assert res_shadow.apply_to_motor is False
        assert res_shadow.reason == "shadow_mode"

        # Test active mode
        active_policy = SafeResidualPolicy.load(tmpdir, mode="active", max_residual_pwm=10)
        res_active = active_policy.predict(obs, current_action="FORWARD")
        assert res_active.valid is True
        assert res_active.apply_to_motor is True
        assert res_active.reason == "ok"
        assert -10 <= res_active.residual_pwm <= 10


def test_safe_residual_policy_rejects_non_linear_actions():
    policy = SafeResidualPolicy(mode="active")
    obs = LibraryObservationV1(1.0, 0.5, 0.0, 0.0, 100.0)
    for turn_action in ("TURN_LEFT", "TURN_RIGHT", "UTURN", "STOP"):
        res = policy.predict(obs, current_action=turn_action)
        assert res.valid is False
        assert res.apply_to_motor is False
        assert res.residual_pwm == 0


# ======================================================================
# Environment Tests
# ======================================================================

def test_corridor_environment_step():
    env = LibraryCorridorEnv(CorridorConfig(segment_length_cm=50.0), seed=42)
    obs = env.reset(direction=1.0)
    assert obs.shape == (5,)

    next_obs, reward, done, info = env.step(0.1)
    assert next_obs.shape == (5,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "reason" in info
