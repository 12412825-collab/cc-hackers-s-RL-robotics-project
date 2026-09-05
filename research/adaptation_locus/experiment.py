"""Single-cell and matrix experiment runners for Phase-0."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import numpy as np

from .baseline import BASELINE, FrozenBaseline
from .env import Phase0CorridorEnv
from .estimator import EstimatorAdapter
from .metrics import (
    EpisodeMetrics,
    aggregate_episodes,
)
from .mismatches import MismatchFamily, SeverityLabel, make_mismatch
from .residual_adapt import ResidualAdapter

AdaptationName = Literal["A0", "A1", "A2"]


def _oscillation_count(series: list[float]) -> int:
    if len(series) < 2:
        return 0
    signs = [1 if x >= 0 else -1 for x in series]
    return sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])


def run_eval_episodes(
    env: Phase0CorridorEnv,
    action_fn: Callable[[np.ndarray], float],
    n_episodes: int,
    seed: int,
) -> list[EpisodeMetrics]:
    episodes: list[EpisodeMetrics] = []
    for i in range(n_episodes):
        init_h = [-2.0, -1.0, 1.0, 2.0][i % 4]
        obs = env.reset(initial_heading_deg=init_h)
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


def run_cell(
    *,
    family: MismatchFamily,
    severity: SeverityLabel,
    adaptation: AdaptationName,
    seed: int,
    baseline: FrozenBaseline = BASELINE,
    online_steps: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Run one Phase-0 cell and return metrics + timeline."""
    online_steps = online_steps or baseline.online_steps
    mismatch = make_mismatch(family, severity)
    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    env = Phase0CorridorEnv(mismatch=mismatch, baseline=baseline, seed=seed)
    estimator = EstimatorAdapter(lr=baseline.estimator_lr)
    residual = ResidualAdapter(lr=0.05)

    if adaptation == "A1":
        estimator.reset(env)
        env.unlock_estimator()
    elif adaptation == "A2":
        residual.reset(seed=seed)
        residual.freeze_env_estimator(env)
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
        if adaptation == "A2":
            action = residual.select_action(obs, deterministic=False)
        else:
            action = 0.0

        next_obs, reward, done, info = env.step(action)

        if adaptation == "A1":
            rec = estimator.update(env, info)
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

        obs = next_obs
        if done:
            obs = env.reset()

        if t % baseline.eval_interval == 0 or t == online_steps:
            eval_env = Phase0CorridorEnv(
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

            eps = run_eval_episodes(
                eval_env, action_fn, baseline.eval_episodes, seed=seed
            )
            agg = aggregate_episodes(eps)
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

    wall_clock_s = time.time() - t0
    final = timeline[-1] if timeline else {}
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
        "mismatch": asdict(mismatch),
        "online_steps": online_steps,
        "wall_clock_s": wall_clock_s,
        "final": final,
        "timeline": timeline,
        "adapted_params": adapted_params,
        "adapt_param_log_tail": adapt_param_log[-50:],
        "adapt_param_log_path": None,
        "rng_probe": float(rng.rand()),
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


def evaluate_transfer(
    *,
    adapted_result: dict[str, Any],
    eval_severity: SeverityLabel,
    baseline: FrozenBaseline = BASELINE,
) -> dict[str, Any]:
    """Evaluate frozen adapted parameters at a nearby severity."""
    family = adapted_result["family"]
    adaptation = adapted_result["adaptation"]
    seed = adapted_result["seed"]
    mismatch = make_mismatch(family, eval_severity)
    env = Phase0CorridorEnv(mismatch=mismatch, baseline=baseline, seed=seed + 20_000)
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
        residual = ResidualAdapter(lr=0.05)
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

    eps = run_eval_episodes(env, action_fn, baseline.eval_episodes, seed=seed)
    agg = aggregate_episodes(eps)
    return {
        "train_severity": adapted_result["severity"],
        "eval_severity": eval_severity,
        "family": family,
        "adaptation": adaptation,
        "seed": seed,
        "metrics": {
            k: v
            for k, v in agg.to_dict().items()
            if k != "episodes"
        },
    }
