#!/usr/bin/env python3
"""Export a trained Library Robot SAC checkpoint to a deployment bundle.

Usage:
    python tools/export_library_sac.py
    python tools/export_library_sac.py --input models/library_sac/checkpoint.pth --output models/library_sac
"""

import argparse
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Export Library SAC to deployment bundle")
    parser.add_argument("--input", type=str, default="models/library_sac/checkpoint.pth")
    parser.add_argument("--output", type=str, default="models/library_sac")
    parser.add_argument("--max-residual-pwm", type=int, default=10)
    args = parser.parse_args()

    from library_residual.policy import LibrarySACAgent
    from library_residual.bundle import export_bundle

    if not os.path.isfile(args.input):
        logger.error("Checkpoint not found: %s", args.input)
        return

    agent = LibrarySACAgent()
    agent.load(args.input)

    paths = export_bundle(
        agent,
        args.output,
        max_residual_pwm=args.max_residual_pwm,
    )

    logger.info("Export complete:")
    for key, path in paths.items():
        logger.info("  %s: %s", key, path)


if __name__ == "__main__":
    main()
