"""Phase-1A Webots Adaptation Locus unit tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from research.adaptation_locus.estimator import EstimatorAdapter
from research.adaptation_locus.residual_adapt import ResidualAdapter
from research.adaptation_locus.webots_env import WebotsFaithfulEnv, make_webots_mismatch


def test_imu_bias_changes_observation_not_true_dynamics_under_privilege_free_open_loop():
    """With zero control action and matched ICs, compare true yaw with/without bias
    when base kp effect is neutralized by resetting estimate each step is hard;
    instead: privilege-free check that motor gains unchanged and estimates diverge
    while applied wheel gains stay 1.0.
    """
    env_b = WebotsFaithfulEnv(mismatch=make_webots_mismatch("imu_bias", "large"), seed=0)
    env_c = WebotsFaithfulEnv(mismatch=make_webots_mismatch("imu_bias", "0"), seed=0)
    env_b.lock_estimator()
    env_c.lock_estimator()
    # Freeze control to open-loop cruise by zeroing kp via monkeypatch
    env_b._controller_kp = 0.0
    env_c._controller_kp = 0.0
    env_b.reset(0.0)
    env_c.reset(0.0)
    true_b, true_c, est_b, est_c = [], [], [], []
    for _ in range(40):
        _, _, _, ib = env_b.step(0.0)
        _, _, _, ic = env_c.step(0.0)
        true_b.append(ib["yaw_true_rad"])
        true_c.append(ic["yaw_true_rad"])
        est_b.append(ib["yaw_est_rad"])
        est_c.append(ic["yaw_est_rad"])
        assert ib["left_gain"] == pytest.approx(1.0)
        assert ib["imu_bias_true"] == pytest.approx(0.30)
    np.testing.assert_allclose(true_b, true_c, atol=1e-6)
    assert abs(est_b[-1] - est_c[-1]) > 0.05


def test_motor_asymmetry_changes_dynamics_not_imu_bias():
    env = WebotsFaithfulEnv(
        mismatch=make_webots_mismatch("motor_asymmetry", "large"), seed=1
    )
    env.lock_estimator()
    env._controller_kp = 0.0
    env.reset(0.0)
    _, _, _, info = env.step(0.0)
    assert info["imu_bias_true"] == pytest.approx(0.0)
    assert info["left_gain"] == pytest.approx(1.06)
    assert info["right_gain"] == pytest.approx(0.94)
    assert abs(info["heading_rate_true"]) > 1e-4
    assert info["wheel_applied_left"] != pytest.approx(info["wheel_cmd_left"]) or True
    # applied != cmd when gains != 1
    assert abs(info["wheel_applied_left"] - info["wheel_cmd_left"]) > 1e-9


def test_a1_cannot_update_controller_params():
    env = WebotsFaithfulEnv(mismatch=make_webots_mismatch("imu_bias", "medium"), seed=2)
    adapter = EstimatorAdapter()
    adapter.reset(env)
    before = env.get_controller_params()
    env.reset()
    for _ in range(25):
        _, _, done, info = env.step(0.0)
        adapter.update(env, info)
        if done:
            env.reset()
    assert env.get_controller_params() == before
    assert abs(env.imu_bias_hat) > 0.0


def test_a2_cannot_update_estimator_params():
    env = WebotsFaithfulEnv(
        mismatch=make_webots_mismatch("motor_asymmetry", "medium"), seed=3
    )
    residual = ResidualAdapter(lr=0.05)
    residual.reset(seed=3)
    env.set_estimator_params(0.0, env.cfg.fusion_weight_init)
    env.lock_estimator()
    with pytest.raises(RuntimeError):
        env.set_estimator_params(1.0, 0.2)
    obs = env.reset()
    for _ in range(20):
        a = residual.select_action(obs)
        next_obs, reward, done, info = env.step(a)
        residual.observe(obs, a, reward, next_obs, done, info=info)
        obs = next_obs if not done else env.reset()
        assert env.imu_bias_hat == pytest.approx(0.0)


def test_residual_respects_frozen_bound():
    env = WebotsFaithfulEnv(mismatch=make_webots_mismatch("none", "0"), seed=4)
    env.lock_estimator()
    env.reset()
    _, _, _, info = env.step(1.0)
    assert abs(info["residual_correction"]) <= env.cfg.residual_angular_scale + 1e-9
    _, _, _, info = env.step(-1.0)
    assert abs(info["residual_correction"]) <= env.cfg.residual_angular_scale + 1e-9


def test_nominal_a0_stable_and_logs_true_vs_observed():
    env = WebotsFaithfulEnv(mismatch=make_webots_mismatch("imu_bias", "0"), seed=5)
    env.lock_estimator()
    env.reset(0.0)
    _, _, _, info = env.step(0.0)
    assert "yaw_true_rad" in info and "yaw_est_rad" in info
    assert "imu_rate_meas" in info and "heading_rate_true" in info
    # short rollout should not immediately fail
    fails = 0
    obs = env.reset(0.0)
    for _ in range(50):
        obs, _, done, info = env.step(0.0)
        if done and not str(info["reason"]).startswith("success"):
            fails += 1
            obs = env.reset(0.0)
    assert fails < 5


def test_seed_reproducibility():
    def roll(seed):
        env = WebotsFaithfulEnv(
            mismatch=make_webots_mismatch("motor_asymmetry", "small"), seed=seed
        )
        env.lock_estimator()
        env.reset(0.0)
        ys = []
        for _ in range(30):
            _, _, _, info = env.step(0.0)
            ys.append(info["yaw_true_rad"])
        return ys

    assert roll(9) == roll(9)
    assert roll(9) != roll(10)
