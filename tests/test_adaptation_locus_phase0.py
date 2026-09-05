"""Unit tests for Phase-0 mismatch injection and adaptation isolation."""

from __future__ import annotations

import numpy as np
import pytest

from research.adaptation_locus.env import Phase0CorridorEnv
from research.adaptation_locus.estimator import EstimatorAdapter
from research.adaptation_locus.mismatches import make_mismatch
from research.adaptation_locus.residual_adapt import ResidualAdapter


def test_imu_bias_affects_observations_not_true_dynamics_gains():
    """With privileged control, IMU bias must not change true trajectories."""
    m = make_mismatch("imu_bias", "large")
    env = Phase0CorridorEnv(mismatch=m, seed=0, use_privileged_heading=True)
    env.lock_estimator()
    m0 = make_mismatch("imu_bias", "0")
    env0 = Phase0CorridorEnv(mismatch=m0, seed=0, use_privileged_heading=True)
    env0.lock_estimator()

    env.reset(initial_heading_deg=0.0)
    env0.reset(initial_heading_deg=0.0)

    true_headings_biased = []
    true_headings_clean = []
    est_headings_biased = []
    est_headings_clean = []
    for _ in range(30):
        _, _, _, info_b = env.step(0.0)
        _, _, _, info_c = env0.step(0.0)
        true_headings_biased.append(info_b["heading_true_deg"])
        true_headings_clean.append(info_c["heading_true_deg"])
        est_headings_biased.append(info_b["heading_est_deg"])
        est_headings_clean.append(info_c["heading_est_deg"])
        assert info_b["left_gain"] == pytest.approx(1.0)
        assert info_b["right_gain"] == pytest.approx(1.0)
        assert info_b["imu_bias_true"] == pytest.approx(3.0)

    np.testing.assert_allclose(true_headings_biased, true_headings_clean, atol=1e-6)
    assert abs(est_headings_biased[-1] - est_headings_clean[-1]) > 0.5


def test_imu_bias_changes_closed_loop_estimates():
    """Without privilege, biased estimates should diverge from clean estimates."""
    env_b = Phase0CorridorEnv(mismatch=make_mismatch("imu_bias", "large"), seed=7)
    env_c = Phase0CorridorEnv(mismatch=make_mismatch("imu_bias", "0"), seed=7)
    env_b.lock_estimator()
    env_c.lock_estimator()
    env_b.reset(0.0)
    env_c.reset(0.0)
    for _ in range(20):
        _, _, _, ib = env_b.step(0.0)
        _, _, _, ic = env_c.step(0.0)
    assert abs(ib["heading_est_deg"] - ic["heading_est_deg"]) > 0.5


def test_motor_asymmetry_affects_dynamics_not_imu_bias():
    m = make_mismatch("motor_asymmetry", "large")
    env = Phase0CorridorEnv(mismatch=m, seed=1)
    env.lock_estimator()
    env.reset(initial_heading_deg=0.0)
    _, _, _, info = env.step(0.0)
    assert info["imu_bias_true"] == pytest.approx(0.0)
    assert info["left_gain"] == pytest.approx(1.045)
    assert info["right_gain"] == pytest.approx(0.955)
    # With zero commanded omega and asymmetric gains, true yaw rate is nonzero
    assert abs(info["heading_rate_true"]) > 1e-3


def test_estimator_adaptation_cannot_alter_controller_params():
    env = Phase0CorridorEnv(mismatch=make_mismatch("imu_bias", "medium"), seed=2)
    adapter = EstimatorAdapter()
    adapter.reset(env)
    before = env.get_controller_params()
    obs = env.reset()
    for _ in range(20):
        obs, reward, done, info = env.step(0.0)
        adapter.update(env, info)
        if done:
            obs = env.reset()
    after = env.get_controller_params()
    assert before == after
    assert abs(env.imu_bias_hat) > 0.0 or abs(env.fusion_weight - 0.7) > 0.0


def test_residual_adaptation_cannot_alter_sensor_calibration():
    env = Phase0CorridorEnv(mismatch=make_mismatch("motor_asymmetry", "medium"), seed=3)
    residual = ResidualAdapter(lr=0.05)
    residual.reset(seed=3)
    residual.freeze_env_estimator(env)
    assert env._estimator_locked
    with pytest.raises(RuntimeError):
        env.set_estimator_params(1.0, 0.2)
    obs = env.reset()
    for _ in range(30):
        action = residual.select_action(obs)
        next_obs, reward, done, info = env.step(action)
        residual.observe(obs, action, reward, next_obs, done, info=info)
        obs = next_obs if not done else env.reset()
        assert env.imu_bias_hat == pytest.approx(0.0)
        assert env.fusion_weight == pytest.approx(env.cfg.fusion_weight_init)
    assert residual.parameter_magnitude() >= 0.0


def test_a1_logs_adaptation_parameters():
    env = Phase0CorridorEnv(mismatch=make_mismatch("imu_bias", "small"), seed=4)
    adapter = EstimatorAdapter()
    adapter.reset(env)
    env.reset()
    for _ in range(10):
        _, _, _, info = env.step(0.0)
        adapter.update(env, info)
    assert len(adapter.history) == 10
    assert "imu_bias_hat" in adapter.history[-1]
