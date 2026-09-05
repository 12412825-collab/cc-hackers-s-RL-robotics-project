"""Phase-1A experiment cells on the Webots-faithful plant."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import numpy as np

from .estimator import EstimatorAdapter
from .metrics import EpisodeMetrics, aggregate_episodes
from .residual_adapt import ResidualAdapter
from .webots_baseline import WEBOTS_BASELINE, WebotsFaithfulBaseline
from .webots_env import WebotsFaithfulEnv, make_webots_mismatch

AdaptationName = Literal["A0", "A1", "A2"]


def _oscillation_count(series: list[float]) -> int:
    if len(series) < 2:
        return 0
    signs = [1 if x >= 0 else -1 for x in series]
    return sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])


def run_eval_episodes(
    env: WebotsFaithfulEnv,
    action_fn: Callable[[np.ndarray], float],
    n_episodes: int,
) -> list[EpisodeMetrics]:
    episodes: list[EpisodeMetrics] = []
    for i in range(n_episodes):
        init = math.radians([-2.0, -1.0, 1.0, 2.0][i % 4])
        obs = env.reset(initial_yaw_rad=init)
        headings: list[float] = []
        laterals: list[float] = []
        residuals: list[float] = []
        residual_signed: list[float] = []
        effort = 0.0
        total_return = 0.0
        info: dict[str, Any] = {"reason": "ongoing"}
        steps = 0
        for _ in range(env.cfg.max_steps):
            action = float(action_fn(obs))
            obs, reward, done, info = env.step(action)
            total_return += reward
            headings.append(abs(float(info["heading_true_deg"])))
            laterals.append(abs(float(info["lateral_cm"])))
            residuals.append(abs(action))
            residual_signed.append(action)
            effort += float(info.get("control_effort", 0.0))
            steps += 1
            if done:
                break
        episodes.append(
            EpisodeMetrics(
                success=str(info.get("reason", "")).startswith("success"),
                reason=str(info.get("reason", "")),
                steps=steps,
                mean_heading_abs=float(np.mean(headings)) if headings else 0.0,
                final_heading_abs=float(headings[-1]) if headings else 0.0,
                mean_lateral_abs=float(np.mean(laterals)) if laterals else 0.0,
                final_lateral_abs=float(laterals[-1]) if laterals else 0.0,
                cumulative_control_effort=float(effort),
                mean_residual_abs=float(np.mean(residuals)) if residuals else 0.0,
                rms_residual=float(np.sqrt(np.mean(np.square(residuals))))
                if residuals
                else 0.0,
                oscillation_count=_oscillation_count(residual_signed),
                episode_return=float(total_return),
            )
        )
    return episodes


def _a0_action(_obs: np.ndarray) -> float:
    return 0.0


def run_webots_cell(
    *,
    family: str,
    severity: str,
    adaptation: AdaptationName,
    seed: int,
    baseline: WebotsFaithfulBaseline = WEBOTS_BASELINE,
    online_steps: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    online_steps = online_steps or baseline.online_steps
    mismatch = make_webots_mismatch(family, severity)
    np.random.seed(seed)

    env = WebotsFaithfulEnv(mismatch=mismatch, baseline=baseline, seed=seed)
    estimator = EstimatorAdapter(lr=baseline.estimator_lr)
    residual = ResidualAdapter(lr=baseline.residual_lr)

    if adaptation == "A1":
        estimator.reset(env)  # type: ignore[arg-type]
        env.unlock_estimator()
    elif adaptation == "A2":
        residual.reset(seed=seed)
        if env._estimator_locked:
            env.unlock_estimator()
        env.set_estimator_params(0.0, baseline.fusion_weight_init)
        env.lock_estimator()
    else:
        if env._estimator_locked:
            env.unlock_estimator()
        env.set_estimator_params(0.0, baseline.fusion_weight_init)
        env.lock_estimator()

    timeline: list[dict[str, Any]] = []
    adapt_param_log: list[dict[str, float]] = []
    t0 = time.time()
    obs = env.reset()

    for t in range(1, online_steps + 1):
        action = (
            residual.select_action(obs, deterministic=False)
            if adaptation == "A2"
            else 0.0
        )
        next_obs, reward, done, info = env.step(action)

        if adaptation == "A1":
            rec = estimator.update(env, info)  # type: ignore[arg-type]
            adapt_param_log.append({"online_steps": float(t), **rec})
        elif adaptation == "A2":
            rec = residual.observe(obs, action, reward, next_obs, done, info=info)
            adapt_param_log.append(
                {
                    "online_steps": float(t),
                    "weight_l2_delta": rec.get("weight_l2_delta", 0.0),
                    "theta0": rec.get("theta0", 0.0),
                    "theta1": rec.get("theta1", 0.0),
                    "imu_bias_hat": float(env.imu_bias_hat),
                    "fusion_weight": float(env.fusion_weight),
                }
            )
            assert env._estimator_locked
        else:
            adapt_param_log.append(
                {
                    "online_steps": float(t),
                    "imu_bias_hat": float(env.imu_bias_hat),
                    "fusion_weight": float(env.fusion_weight),
                }
            )

        obs = next_obs if not done else env.reset()

        if t % baseline.eval_interval == 0 or t == online_steps:
            eval_env = WebotsFaithfulEnv(
                mismatch=mismatch, baseline=baseline, seed=seed + 10_000
            )
            if adaptation == "A1":
                eval_env.unlock_estimator()
                eval_env.set_estimator_params(env.imu_bias_hat, env.fusion_weight)
                eval_env.lock_estimator()
                action_fn = _a0_action
            elif adaptation == "A2":
                eval_env.set_estimator_params(0.0, baseline.fusion_weight_init)
                eval_env.lock_estimator()

                def action_fn(o, _r=residual):
                    return float(_r.select_action(o, deterministic=True))

            else:
                eval_env.set_estimator_params(0.0, baseline.fusion_weight_init)
                eval_env.lock_estimator()
                action_fn = _a0_action

            agg = aggregate_episodes(
                run_eval_episodes(eval_env, action_fn, baseline.eval_episodes)
            )
            timeline.append(
                {
                    "online_steps": t,
                    "performance_score": agg.performance_score,
                    "success_rate": agg.success_rate,
                    "mean_heading_abs": agg.mean_heading_abs,
                    "final_heading_abs": agg.final_heading_abs,
                    "mean_lateral_abs": agg.mean_lateral_abs,
                    "cumulative_control_effort": agg.cumulative_control_effort,
                    "mean_residual_abs": agg.mean_residual_abs,
                    "oscillation_count": agg.oscillation_count,
                    "mean_return": agg.mean_return,
                    "param_magnitude": (
                        estimator.parameter_magnitude()
                        if adaptation == "A1"
                        else residual.parameter_magnitude()
                        if adaptation == "A2"
                        else 0.0
                    ),
                }
            )

    adapted_params: dict[str, float] = {}
    if adaptation == "A1":
        adapted_params = env.get_estimator_params()
    elif adaptation == "A2":
        adapted_params = residual.get_params()

    result = {
        "family": family,
        "severity": severity,
        "adaptation": adaptation,
        "seed": seed,
        "plant": "webots_faithful",
        "mismatch": asdict(mismatch),
        "imu_bias_rad_s": float(env.imu_bias_true_rad_s),
        "online_steps": online_steps,
        "wall_clock_s": time.time() - t0,
        "final": timeline[-1] if timeline else {},
        "timeline": timeline,
        "adapted_params": adapted_params,
        "adapt_param_log_tail": adapt_param_log[-50:],
        "adapt_param_log_path": None,
        "baseline": baseline.to_dict(),
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{family}_{severity}_{adaptation}_seed{seed}"
        param_path = output_dir / f"{stem}_params.jsonl"
        with param_path.open("w", encoding="utf-8") as f:
            for row in adapt_param_log:
                f.write(json.dumps(row) + "\n")
        result["adapt_param_log_path"] = str(param_path)
        with (output_dir / f"{stem}.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    return result


def evaluate_webots_transfer(
    *,
    adapted_result: dict[str, Any],
    eval_severity: str,
    baseline: WebotsFaithfulBaseline = WEBOTS_BASELINE,
) -> dict[str, Any]:
    family = adapted_result["family"]
    adaptation = adapted_result["adaptation"]
    seed = adapted_result["seed"]
    mismatch = make_webots_mismatch(family, eval_severity)
    env = WebotsFaithfulEnv(mismatch=mismatch, baseline=baseline, seed=seed + 20_000)
    params = adapted_result.get("adapted_params") or {}

    if adaptation == "A1":
        env.unlock_estimator()
        env.set_estimator_params(
            float(params.get("imu_bias_hat", 0.0)),
            float(params.get("fusion_weight", baseline.fusion_weight_init)),
        )
        env.lock_estimator()
        action_fn = _a0_action
    elif adaptation == "A2":
        residual = ResidualAdapter(lr=baseline.residual_lr)
        residual.reset(seed=seed)
        residual.set_params(
            float(params.get("theta0", 0.0)),
            float(params.get("theta1", 0.0)),
        )
        env.set_estimator_params(0.0, baseline.fusion_weight_init)
        env.lock_estimator()

        def action_fn(o, _r=residual):
            return float(_r.select_action(o, deterministic=True))

    else:
        env.lock_estimator()
        action_fn = _a0_action

    agg = aggregate_episodes(
        run_eval_episodes(env, action_fn, baseline.eval_episodes)
    )
    return {
        "train_severity": adapted_result["severity"],
        "eval_severity": eval_severity,
        "family": family,
        "adaptation": adaptation,
        "seed": seed,
        "metrics": {k: v for k, v in agg.to_dict().items() if k != "episodes"},
    }
