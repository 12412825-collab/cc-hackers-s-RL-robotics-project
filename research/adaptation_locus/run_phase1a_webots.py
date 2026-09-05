"""CLI for Phase-1A Webots Adaptation Locus replication."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.adaptation_locus.metrics import (
    performance_recovery_ratio,
    samples_to_recovery,
)
from research.adaptation_locus.plotting import generate_phase0_plots
from research.adaptation_locus.webots_baseline import WEBOTS_BASELINE
from research.adaptation_locus.webots_experiment import (
    evaluate_webots_transfer,
    run_webots_cell,
)

STAGE_SEEDS = {
    "smoke": [0],
    "debug": [0, 1, 2],
    "decision": [0, 1, 2, 3, 4],
}
STAGE_SEVERITIES = {
    "smoke": ["medium"],
    "debug": ["0", "medium", "large"],
    "decision": ["0", "small", "medium", "large"],
}
FAMILIES = ["imu_bias", "motor_asymmetry"]
ADAPTATIONS = ["A0", "A1", "A2"]


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None or not Path(path).is_file():
        return {}
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError:
            alt = path.with_suffix(".json")
            if alt.is_file():
                return json.loads(alt.read_text(encoding="utf-8"))
            return {}
        return yaml.safe_load(text) or {}
    return json.loads(text)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    nominal_scores = [
        r["final"]["performance_score"]
        for r in rows
        if r["severity"] == "0" and r["adaptation"] == "A0" and r.get("final")
    ]
    if not nominal_scores:
        nom = run_webots_cell(
            family="imu_bias",
            severity="0",
            adaptation="A0",
            seed=0,
            online_steps=WEBOTS_BASELINE.eval_interval,
        )
        nominal_scores = [nom["final"]["performance_score"]]
    nominal = float(np.mean(nominal_scores))

    enriched = []
    for r in rows:
        if not r.get("final"):
            continue
        a0 = next(
            (
                x
                for x in rows
                if x["family"] == r["family"]
                and x["severity"] == r["severity"]
                and x["adaptation"] == "A0"
                and x["seed"] == r["seed"]
                and x.get("final")
            ),
            None,
        )
        shifted = a0["final"]["performance_score"] if a0 else nominal
        prr = performance_recovery_ratio(
            r["final"]["performance_score"], nominal, shifted
        )
        str_steps = samples_to_recovery(
            r.get("timeline", []),
            nominal_score=nominal,
            shifted_score=shifted,
            threshold=WEBOTS_BASELINE.recovery_threshold,
        )
        enriched.append(
            {
                **{k: r[k] for k in ("family", "severity", "adaptation", "seed")},
                "performance_score": r["final"]["performance_score"],
                "success_rate": r["final"]["success_rate"],
                "mean_heading_abs": r["final"]["mean_heading_abs"],
                "final_heading_abs": r["final"]["final_heading_abs"],
                "cumulative_control_effort": r["final"]["cumulative_control_effort"],
                "mean_residual_abs": r["final"]["mean_residual_abs"],
                "oscillation_count": r["final"]["oscillation_count"],
                "param_magnitude": r["final"].get("param_magnitude", 0.0),
                "wall_clock_s": r.get("wall_clock_s", 0.0),
                "prr": prr,
                "samples_to_recovery": str_steps,
            }
        )

    interaction = {}
    for family in FAMILIES:
        interaction[family] = {}
        for adapt in ADAPTATIONS:
            vals = [
                e["prr"]
                for e in enriched
                if e["family"] == family
                and e["adaptation"] == adapt
                and e["severity"] == "medium"
            ]
            interaction[family][adapt] = {
                "n": len(vals),
                "mean_prr": float(np.mean(vals)) if vals else None,
                "std_prr": float(np.std(vals)) if vals else None,
                "mean_perf": float(
                    np.mean(
                        [
                            e["performance_score"]
                            for e in enriched
                            if e["family"] == family
                            and e["adaptation"] == adapt
                            and e["severity"] == "medium"
                        ]
                    )
                )
                if vals
                else None,
            }
    return {
        "nominal_performance_score": nominal,
        "rows": enriched,
        "interaction_medium": interaction,
        "plant": "webots_faithful",
        "amendment": "W-1",
    }


def decide(summary: dict[str, Any]) -> dict[str, Any]:
    inter = summary["interaction_medium"]
    imu = inter.get("imu_bias", {})
    mot = inter.get("motor_asymmetry", {})

    def mean(d, k):
        return (d.get(k) or {}).get("mean_prr")

    imu_a1, imu_a2, imu_a0 = mean(imu, "A1"), mean(imu, "A2"), mean(imu, "A0")
    mot_a1, mot_a2, mot_a0 = mean(mot, "A1"), mean(mot, "A2"), mean(mot, "A0")

    h1 = (
        imu_a1 is not None
        and imu_a2 is not None
        and imu_a1 > imu_a2
        and (imu_a0 is None or imu_a1 > imu_a0)
    )
    h2 = (
        mot_a2 is not None
        and mot_a1 is not None
        and mot_a2 > mot_a1
        and (mot_a0 is None or mot_a2 > mot_a0)
    )

    rows = summary["rows"]
    misloc = []
    for family, right, wrong in (
        ("imu_bias", "A1", "A2"),
        ("motor_asymmetry", "A2", "A1"),
    ):
        right_rows = [
            r
            for r in rows
            if r["family"] == family and r["adaptation"] == right and r["severity"] == "medium"
        ]
        wrong_rows = [
            r
            for r in rows
            if r["family"] == family and r["adaptation"] == wrong and r["severity"] == "medium"
        ]
        if not right_rows or not wrong_rows:
            continue
        misloc.append(
            {
                "family": family,
                "right_prr": float(np.mean([r["prr"] for r in right_rows])),
                "wrong_prr": float(np.mean([r["prr"] for r in wrong_rows])),
                "wrong_still_positive": float(np.mean([r["prr"] for r in wrong_rows])) > 0.2,
                "wrong_higher_effort": float(
                    np.mean([r["cumulative_control_effort"] for r in wrong_rows])
                )
                > float(np.mean([r["cumulative_control_effort"] for r in right_rows])),
            }
        )

    if h1 and h2:
        verdict = "STRONG REPLICATION"
    elif h1 or h2 or any(
        m["wrong_still_positive"] and m["wrong_higher_effort"] for m in misloc
    ):
        verdict = "PARTIAL REPLICATION"
    else:
        verdict = "FAIL TO REPLICATE"

    return {
        "verdict": verdict,
        "h1_supported": bool(h1),
        "h2_supported": bool(h2),
        "mislocalization_signals": misloc,
        "interaction_means": {
            "imu_bias": {"A0": imu_a0, "A1": imu_a1, "A2": imu_a2},
            "motor_asymmetry": {"A0": mot_a0, "A1": mot_a1, "A2": mot_a2},
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase-1A Webots Adaptation Locus")
    parser.add_argument("--stage", choices=["smoke", "debug", "decision"], default="smoke")
    parser.add_argument(
        "--config",
        type=str,
        default="research/adaptation_locus/configs/phase1a_webots.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/adaptation_locus_phase1a_webots",
    )
    parser.add_argument("--online-steps", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(Path(args.config) if args.config else None)
    online_steps = args.online_steps or cfg.get("online_steps") or WEBOTS_BASELINE.online_steps
    if args.stage == "smoke":
        online_steps = min(online_steps, cfg.get("smoke_online_steps", 400))

    out = Path(args.output)
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    seeds = STAGE_SEEDS[args.stage]
    severities = STAGE_SEVERITIES[args.stage]
    rows: list[dict[str, Any]] = []

    for family in FAMILIES:
        for severity in severities:
            for adaptation in ADAPTATIONS:
                for seed in seeds:
                    print(
                        f"[phase1a] {family} | {severity} | {adaptation} | seed={seed}",
                        flush=True,
                    )
                    rows.append(
                        run_webots_cell(
                            family=family,
                            severity=severity,
                            adaptation=adaptation,  # type: ignore[arg-type]
                            seed=seed,
                            online_steps=online_steps,
                            output_dir=raw_dir,
                        )
                    )

    transfers = []
    if args.stage in ("debug", "decision") and "medium" in severities:
        for family in FAMILIES:
            for adaptation in ("A1", "A2"):
                for seed in seeds:
                    base = next(
                        r
                        for r in rows
                        if r["family"] == family
                        and r["adaptation"] == adaptation
                        and r["severity"] == "medium"
                        and r["seed"] == seed
                    )
                    for eval_sev in ("small", "large"):
                        transfers.append(
                            evaluate_webots_transfer(
                                adapted_result=base, eval_severity=eval_sev
                            )
                        )

    summary = summarize(rows)
    summary["transfers"] = transfers
    decision = decide(summary)

    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"stage": args.stage, "summary": summary, "decision": decision},
            f,
            indent=2,
        )

    plot_paths = generate_phase0_plots(rows, out / "plots")
    print("Wrote", out / "summary.json")
    for p in plot_paths:
        print("Plot", p)
    print("VERDICT:", decision["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
