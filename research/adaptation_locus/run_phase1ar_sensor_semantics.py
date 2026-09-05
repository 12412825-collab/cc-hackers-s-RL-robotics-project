"""Launch Phase-1A-R Step 3.5 sensor semantics validation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORLD = REPO / "simulation" / "worlds" / "four_wheel_track_phase1ar_semantics.wbt"
WEBOTS = Path(os.environ.get("WEBOTS_HOME", r"C:\Program Files\Webots")) / (
    "msys64/mingw64/bin/webots.exe"
)


def main() -> int:
    out = REPO / "results" / "adaptation_locus_phase1ar" / "sensor_semantics"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(WEBOTS),
        "--mode=fast",
        "--batch",
        "--stdout",
        "--stderr",
        "--minimize",
        "--no-rendering",
        str(WORLD),
    ]
    print("CMD:", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("WEBOTS_HOME", str(WEBOTS.parents[3]))
    return subprocess.run(cmd, env=env, cwd=str(REPO)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
