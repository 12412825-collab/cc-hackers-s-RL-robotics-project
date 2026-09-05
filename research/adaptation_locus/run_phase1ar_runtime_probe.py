"""Launch Phase-1A-R Step 1 live Webots physics provenance probe."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_WORLD = REPO / "simulation" / "worlds" / "four_wheel_track_phase1ar_probe.wbt"
HISTORICAL_WORLD = REPO / "simulation" / "worlds" / "four_wheel_track.wbt"
WEBOTS_CANDIDATES = [
    Path(os.environ.get("WEBOTS_HOME", r"C:\Program Files\Webots"))
    / "msys64"
    / "mingw64"
    / "bin"
    / "webots.exe",
    Path(r"C:\Program Files\Webots\msys64\mingw64\bin\webots.exe"),
]


def find_webots() -> Path:
    for path in WEBOTS_CANDIDATES:
        if path.is_file():
            return path
    raise FileNotFoundError("webots.exe not found under WEBOTS_HOME candidates")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--world",
        type=Path,
        default=DEFAULT_WORLD,
        help="World to load (default: historical assets + probe controller)",
    )
    parser.add_argument(
        "--historical-check",
        action="store_true",
        help="Also print path of unmodified historical world for audit",
    )
    parser.add_argument("--mode", default="fast", choices=["fast", "realtime", "pause"])
    parser.add_argument("--no-rendering", action="store_true", default=True)
    args = parser.parse_args()

    webots = find_webots()
    world = args.world.resolve()
    if not world.is_file():
        raise FileNotFoundError(world)

    cmd = [
        str(webots),
        f"--mode={args.mode}",
        "--batch",
        "--stdout",
        "--stderr",
        "--minimize",
    ]
    if args.no_rendering:
        cmd.append("--no-rendering")
    cmd.append(str(world))

    print("WEBOTS_EXE:", webots)
    print("WORLD:", world)
    if args.historical_check:
        print("HISTORICAL_WORLD:", HISTORICAL_WORLD, "exists=", HISTORICAL_WORLD.is_file())
    print("CMD:", " ".join(cmd))
    sys.stdout.flush()

    env = os.environ.copy()
    env.setdefault("WEBOTS_HOME", str(webots.parents[3]))  # .../Webots
    # Controllers use runtime.ini python; ensure Webots home is visible.
    proc = subprocess.run(cmd, env=env, cwd=str(REPO))
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
