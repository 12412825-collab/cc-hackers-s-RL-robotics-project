#!/usr/bin/env python3
"""
Train Library Robot Residual SAC
================================

Trains the sensor-only MLP SAC in the LibraryCorridorEnv simulator.

Usage:
    python train_library_sac.py
    python train_library_sac.py --total-steps 50000 --seed 42
    python train_library_sac.py --resume models/library_sac/checkpoint.pth
    python train_library_sac.py --total-steps 200000 --eval-interval 5000

Options:
    --total-steps N          Total environment steps [default: 50000]
    --seed N                 Random seed [default: 42]
    --output DIR             Output directory [default: models/library_sac]
    --device DEV             Torch device [default: cpu]
    --resume PATH            Resume from checkpoint
    --eval-interval N        Evaluate every N steps [default: 2500]
    --checkpoint-interval N  Save checkpoint every N steps [default: 10000]
    --warmup-steps N         Random exploration steps [default: 1000]
    --batch-size N           Batch size for SAC updates [default: 256]
    --hidden-dim N           MLP hidden layer dimension [default: 64]
    --buffer-size N          Replay buffer capacity [default: 100000]
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Library Robot Residual SAC")
    parser.add_argument("--total-steps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="models/library_sac")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval-interval", type=int, default=2_500)
    parser.add_argument("--checkpoint-interval", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    from library_residual.env import (
        CorridorConfig,
        LibraryCorridorEnv,
        make_eval_scenarios,
        run_eval_episode,
    )
    from library_residual.policy import LibrarySACAgent, SensorReplayBuffer
    from library_residual.bundle import export_bundle

    logger.info("=" * 60)
    logger.info("Library Robot Residual SAC Training")
    logger.info("=" * 60)
    logger.info("Total steps:    %d", args.total_steps)
    logger.info("Seed:           %d", args.seed)
    logger.info("Hidden dim:     %d", args.hidden_dim)
    logger.info("Batch size:     %d", args.batch_size)
    logger.info("Buffer size:    %d", args.buffer_size)
    logger.info("Warmup steps:   %d", args.warmup_steps)
    logger.info("Device:         %s", args.device)
    logger.info("Output:         %s", args.output)

    # Create environment
    cfg = CorridorConfig()
    env = LibraryCorridorEnv(cfg, seed=args.seed)
    eval_env = LibraryCorridorEnv(cfg, seed=args.seed + 1000)

    # Create agent
    agent = LibrarySACAgent(
        hidden_dim=args.hidden_dim,
        device=args.device,
    )

    # Resume from checkpoint
    if args.resume and os.path.isfile(args.resume):
        logger.info("Resuming from %s", args.resume)
        agent.load(args.resume)

    # Create replay buffer
    buffer = SensorReplayBuffer(capacity=args.buffer_size)

    # Training loop
    obs = env.reset()
    episode_reward = 0.0
    episode_rewards = []
    training_log = []
    best_eval_return = -float("inf")
    t_start = time.time()

    for step in range(1, args.total_steps + 1):
        # Select action
        if step <= args.warmup_steps:
            action = np.random.uniform(-1.0, 1.0)
        else:
            action = agent.select_action(obs, deterministic=False)

        # Step environment
        next_obs, reward, done, info = env.step(action)
        buffer.push(obs, np.array([action], dtype=np.float32), reward, next_obs, done)
        episode_reward += reward

        if done:
            episode_rewards.append(episode_reward)
            obs = env.reset()
            episode_reward = 0.0
        else:
            obs = next_obs

        # Update agent
        if step > args.warmup_steps and len(buffer) >= args.batch_size:
            metrics = agent.update(buffer, batch_size=args.batch_size)
        else:
            metrics = {}

        # Evaluation
        if step % args.eval_interval == 0:
            scenarios = make_eval_scenarios(eval_env)
            eval_results = [
                run_eval_episode(eval_env, agent, sc) for sc in scenarios
            ]
            mean_return = np.mean([r["return"] for r in eval_results])
            success_rate = np.mean([r["success"] for r in eval_results])
            mean_heading = np.mean([r["mean_abs_heading"] for r in eval_results])
            mean_residual = np.mean([r["mean_residual"] for r in eval_results])
            mean_oscillation = np.mean(
                [r["oscillation_count"] for r in eval_results]
            )

            elapsed = time.time() - t_start
            recent_rewards = episode_rewards[-20:] if episode_rewards else [0]
            log_entry = {
                "step": step,
                "elapsed_s": round(elapsed, 1),
                "train_mean_return": round(float(np.mean(recent_rewards)), 2),
                "eval_mean_return": round(float(mean_return), 2),
                "eval_success_rate": round(float(success_rate), 3),
                "eval_mean_heading": round(float(mean_heading), 2),
                "eval_mean_residual": round(float(mean_residual), 4),
                "eval_mean_oscillation": round(float(mean_oscillation), 1),
                "buffer_size": len(buffer),
                "alpha": round(agent.alpha, 4),
            }
            training_log.append(log_entry)

            logger.info(
                "Step %d/%d (%.0fs) | Train %.1f | Eval %.1f | "
                "Success %.0f%% | Heading %.1f° | Residual %.3f | α=%.4f",
                step,
                args.total_steps,
                elapsed,
                log_entry["train_mean_return"],
                mean_return,
                success_rate * 100,
                mean_heading,
                mean_residual,
                agent.alpha,
            )

            # Save best model
            if mean_return > best_eval_return:
                best_eval_return = mean_return
                best_path = os.path.join(args.output, "checkpoint_best.pth")
                os.makedirs(args.output, exist_ok=True)
                agent.save(best_path)
                logger.info(
                    "New best eval return: %.2f → saved to %s",
                    mean_return,
                    best_path,
                )

        # Periodic checkpoint
        if step % args.checkpoint_interval == 0:
            ckpt_path = os.path.join(args.output, f"checkpoint_{step}.pth")
            os.makedirs(args.output, exist_ok=True)
            agent.save(ckpt_path)

    # Final export
    logger.info("Training complete. Exporting bundle...")
    paths = export_bundle(agent, args.output)

    # Save training log
    log_path = os.path.join(args.output, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    logger.info("Bundle exported to: %s", args.output)
    logger.info("Training log: %s", log_path)
    logger.info(
        "Total episodes: %d | Best eval return: %.2f",
        len(episode_rewards),
        best_eval_return,
    )


if __name__ == "__main__":
    main()
