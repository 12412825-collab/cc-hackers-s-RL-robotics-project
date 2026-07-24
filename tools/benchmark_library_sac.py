#!/usr/bin/env python3
"""Benchmark Library Robot SAC inference latency on CPU.

Usage:
    python tools/benchmark_library_sac.py
    python tools/benchmark_library_sac.py --model models/library_sac/ --iterations 5000
"""

import argparse
import logging
import os
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Library SAC latency")
    parser.add_argument("--model", type=str, default="models/library_sac")
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()

    import torch
    from library_residual.types import OBSERVATION_DIM

    actor_path = os.path.join(args.model, "actor.ts")
    if not os.path.isfile(actor_path):
        logger.error("TorchScript actor not found: %s", actor_path)
        return

    model = torch.jit.load(actor_path, map_location="cpu")
    model.eval()

    # Warm-up
    for _ in range(50):
        x = torch.randn(1, OBSERVATION_DIM)
        _ = model(x)

    latencies = []
    invalid_count = 0

    for i in range(args.iterations):
        obs = np.random.uniform(-1.0, 1.0, size=OBSERVATION_DIM).astype(np.float32)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

        t0 = time.perf_counter()
        with torch.no_grad():
            output = model(obs_tensor)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000.0)  # ms

        action = float(output.squeeze())
        if not (-1.0 <= action <= 1.0) or not np.isfinite(action):
            invalid_count += 1

    latencies = np.array(latencies)

    logger.info("=" * 50)
    logger.info("Library SAC Inference Benchmark")
    logger.info("=" * 50)
    logger.info("Model:        %s", actor_path)
    logger.info("Iterations:   %d", args.iterations)
    logger.info("Avg latency:  %.3f ms", np.mean(latencies))
    logger.info("Median:       %.3f ms", np.median(latencies))
    logger.info("P95:          %.3f ms", np.percentile(latencies, 95))
    logger.info("P99:          %.3f ms", np.percentile(latencies, 99))
    logger.info("Max:          %.3f ms", np.max(latencies))
    logger.info("Min:          %.3f ms", np.min(latencies))
    logger.info("Invalid out:  %d / %d (%.1f%%)",
                invalid_count, args.iterations,
                100 * invalid_count / args.iterations if args.iterations else 0)
    logger.info("Throughput:   %.0f inferences/second",
                1000.0 / np.mean(latencies) if np.mean(latencies) > 0 else 0)

    # Check Pi 5 feasibility (need < 50ms per inference for 10Hz)
    p95 = np.percentile(latencies, 95)
    if p95 < 50.0:
        logger.info("✓ Feasible for 10Hz inference on this hardware")
    elif p95 < 100.0:
        logger.info("⚠ Feasible for 5Hz inference; 10Hz may be tight")
    else:
        logger.info("✗ May be too slow for real-time inference")


if __name__ == "__main__":
    main()
