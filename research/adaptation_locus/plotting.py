"""Plotting for Phase-0 decision visuals."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _try_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def generate_phase0_plots(rows: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    """Generate the four required conceptual plots from flat result rows."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        plt = _try_pyplot()
    except ImportError:
        # Still write CSV tables so analysis is possible without matplotlib
        csv_path = out_dir / "phase0_final_metrics.csv"
        with csv_path.open("w", encoding="utf-8") as f:
            f.write(
                "family,severity,adaptation,seed,performance_score,mean_heading_abs,"
                "cumulative_control_effort,mean_residual_abs\n"
            )
            for r in rows:
                fin = r.get("final") or {}
                f.write(
                    f"{r['family']},{r['severity']},{r['adaptation']},{r['seed']},"
                    f"{fin.get('performance_score', '')},"
                    f"{fin.get('mean_heading_abs', '')},"
                    f"{fin.get('cumulative_control_effort', '')},"
                    f"{fin.get('mean_residual_abs', '')}\n"
                )
        return [csv_path]

    paths: list[Path] = []

    severities = ["0", "small", "medium", "large"]
    sev_x = np.arange(len(severities))
    families = ["imu_bias", "motor_asymmetry"]
    adaptations = ["A0", "A1", "A2"]
    colors = {"A0": "#444444", "A1": "#1f77b4", "A2": "#d62728"}

    def _collect(family, adaptation, severity, key):
        vals = [
            r["final"].get(key, np.nan)
            for r in rows
            if r["family"] == family
            and r["adaptation"] == adaptation
            and r["severity"] == severity
            and r.get("final")
        ]
        return vals

    # 1) severity vs final performance
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, family in zip(axes, families):
        for adapt in adaptations:
            means, stds = [], []
            for sev in severities:
                vals = _collect(family, adapt, sev, "performance_score")
                means.append(np.mean(vals) if vals else np.nan)
                stds.append(np.std(vals) if vals else 0.0)
            ax.errorbar(
                sev_x, means, yerr=stds, marker="o", label=adapt, color=colors[adapt]
            )
        ax.set_xticks(sev_x, severities)
        ax.set_title(family)
        ax.set_xlabel("mismatch severity")
        ax.set_ylabel("performance score (higher better)")
        ax.legend()
    fig.suptitle("Mismatch severity vs final performance")
    fig.tight_layout()
    p1 = out_dir / "severity_vs_performance.png"
    fig.savefig(p1, dpi=140)
    plt.close(fig)
    paths.append(p1)

    # 2) samples vs recovery (PRR timeline for medium)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, family in zip(axes, families):
        for adapt in adaptations:
            # average timeline across seeds at medium
            timelines = [
                r["timeline"]
                for r in rows
                if r["family"] == family
                and r["adaptation"] == adapt
                and r["severity"] == "medium"
            ]
            if not timelines:
                continue
            # align by online_steps
            steps = [row["online_steps"] for row in timelines[0]]
            mat = []
            for tl in timelines:
                mat.append([row["performance_score"] for row in tl])
            mean = np.mean(mat, axis=0)
            ax.plot(steps, mean, label=adapt, color=colors[adapt])
        ax.set_title(f"{family} (medium)")
        ax.set_xlabel("online adaptation samples")
        ax.set_ylabel("performance score")
        ax.legend()
    fig.suptitle("Online samples vs performance recovery")
    fig.tight_layout()
    p2 = out_dir / "samples_vs_recovery.png"
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    paths.append(p2)

    # 3) severity vs residual/control magnitude
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, family in zip(axes, families):
        for adapt in adaptations:
            means, stds = [], []
            for sev in severities:
                # prefer residual for A2, control effort otherwise
                key = "mean_residual_abs" if adapt == "A2" else "cumulative_control_effort"
                vals = _collect(family, adapt, sev, key)
                means.append(np.mean(vals) if vals else np.nan)
                stds.append(np.std(vals) if vals else 0.0)
            ax.errorbar(
                sev_x, means, yerr=stds, marker="o", label=adapt, color=colors[adapt]
            )
        ax.set_xticks(sev_x, severities)
        ax.set_title(family)
        ax.set_xlabel("mismatch severity")
        ax.set_ylabel("control / residual magnitude")
        ax.legend()
    fig.suptitle("Mismatch severity vs residual/control magnitude")
    fig.tight_layout()
    p3 = out_dir / "severity_vs_control_magnitude.png"
    fig.savefig(p3, dpi=140)
    plt.close(fig)
    paths.append(p3)

    # 4) interaction plot at medium severity
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(families))
    width = 0.25
    for i, adapt in enumerate(adaptations):
        means, stds = [], []
        for family in families:
            vals = _collect(family, adapt, "medium", "performance_score")
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if vals else 0.0)
        ax.bar(
            x + (i - 1) * width,
            means,
            width,
            yerr=stds,
            label=adapt,
            color=colors[adapt],
            capsize=3,
        )
    ax.set_xticks(x, families)
    ax.set_ylabel("performance score @ medium")
    ax.set_title("Mismatch Type × Adaptation Locus")
    ax.legend()
    fig.tight_layout()
    p4 = out_dir / "mismatch_x_locus_interaction.png"
    fig.savefig(p4, dpi=140)
    plt.close(fig)
    paths.append(p4)

    return paths
