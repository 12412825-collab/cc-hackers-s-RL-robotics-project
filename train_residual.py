#!/usr/bin/env python3
"""
Residual RL Training Script (SAC - Soft Actor-Critic)
======================================================

Trains a Residual RL model on DonkeyCar Tub data.

Backbone Options (set in myconfig.py: RESIDUAL_BACKBONE):
  - mobilenet_v3_small  - default, ImageNet pretrained, ~2.5M params
  - mobilenet_v2        - ImageNet pretrained, ~3.5M params
  - efficientnet_b0     - ImageNet pretrained, ~5.3M params
  - donkey_cnn          - original 5-conv from-scratch CNN (baseline)

Training Modes:
  1. Transfer Learning (default): pretrained backbone frozen,
     only projection head + SAC heads trained. Fast convergence.
  2. Full Fine-tuning: set RESIDUAL_FREEZE_BACKBONE=False

Image Pipeline:
  Camera (BGR, 120x160) -> BGR->RGB -> Resize(224x224) -> ImageNet Norm -> Backbone

Usage:
    # Train with default config (mobilenet_v3_small)
    python train_residual.py --tubs data/ --base models/mypilot.h5

    # Train with specific backbone (overrides myconfig.py)
    python train_residual.py --tubs data/ --base models/mypilot.h5 --backbone efficientnet_b0

    # Full fine-tuning (unfreeze backbone)
    python train_residual.py --tubs data/ --base models/mypilot.h5 --unfreeze

    # Resume from checkpoint + fine-tune
    python train_residual.py --tubs data/ --base models/mypilot.h5 --resume models/residual_sac.pth --unfreeze

    # Custom parameters
    python train_residual.py --tubs data/ --base models/mypilot.h5 --epochs 200 --steps 2000

Options:
    -h --help               Show this screen.
    --tubs=<paths>          Tub directory paths (space-separated)
    --base=<model>          Path to base KerasLinear model (.h5) [required]
    --output=<model>        Output path for trained model [default: models/residual_sac.pth]
    --resume=<model>        Resume training from checkpoint
    --backbone=<type>       Override backbone type [default: ]
    --unfreeze              Unfreeze backbone for full fine-tuning
    --epochs=<n>            Number of training epochs [default: 100]
    --steps=<n>             Gradient steps per epoch [default: 1000]
"""

import os
import sys
import glob
import logging
from docopt import docopt

import donkeycar as dk

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def find_tub_dirs(base_path):
    """Find all Tub directories under base_path."""
    tubs = []
    if os.path.isfile(os.path.join(base_path, 'manifest.json')):
        tubs.append(base_path)
    else:
        for root, dirs, files in os.walk(base_path):
            if 'manifest.json' in files or any(
                f.endswith('.json') for f in files if not f.startswith('.')
            ):
                json_files = glob.glob(os.path.join(root, 'record_*.json'))
                if json_files:
                    tubs.append(root)
    return tubs


def main():
    args = docopt(__doc__)
    cfg = dk.load_config()

    # Parse arguments
    tub_paths = args['--tubs']
    base_model = args['--base']
    output_path = args['--output']
    resume_path = args['--resume']
    backbone_override = args['--backbone']
    unfreeze = args['--unfreeze']
    num_epochs = int(args['--epochs'])
    steps_per_epoch = int(args['--steps'])

    # Apply backbone override
    if backbone_override:
        cfg.RESIDUAL_BACKBONE = backbone_override
        logger.info(f"Backbone override: {backbone_override}")

    if unfreeze:
        cfg.RESIDUAL_FREEZE_BACKBONE = False
        logger.info("Unfreeze mode: backbone will be fully trainable")

    # Get backbone info
    backbone = getattr(cfg, 'RESIDUAL_BACKBONE', 'mobilenet_v3_small')
    freeze = getattr(cfg, 'RESIDUAL_FREEZE_BACKBONE', True)
    feature_dim = getattr(cfg, 'RESIDUAL_FEATURE_DIM', 50)

    logger.info("=" * 60)
    logger.info("Residual RL Training (SAC)")
    logger.info("=" * 60)
    logger.info(f"Backbone:       {backbone}")
    logger.info(f"Feature dim:    {feature_dim}")
    logger.info(f"Freeze:         {freeze}")
    logger.info(f"Tub paths:      {tub_paths}")
    logger.info(f"Base model:     {base_model}")
    logger.info(f"Output:         {output_path}")
    logger.info(f"Epochs:         {num_epochs}")
    logger.info(f"Steps/epoch:    {steps_per_epoch}")

    # Find all tub directories
    all_tubs = []
    for path in tub_paths.split():
        found = find_tub_dirs(path)
        all_tubs.extend(found)
        logger.info(f"  {path}: found {len(found)} tub(s)")

    if not all_tubs:
        logger.error(
            "No Tub directories found! "
            "Make sure you have recorded data in data/"
        )
        sys.exit(1)

    logger.info(f"Total tubs: {len(all_tubs)}")

    # Import Residual RL module
    try:
        from parts.residual_rl import ResidualTrainer
    except ImportError:
        logger.error(
            "Cannot import parts.residual_rl. "
            "Make sure parts/residual_rl.py exists and PyTorch is installed."
        )
        sys.exit(1)

    # Create trainer
    trainer = ResidualTrainer(cfg)

    # Resume from checkpoint if specified
    if resume_path and os.path.exists(resume_path):
        logger.info(f"Resuming from {resume_path}")
        trainer.agent.load(resume_path)
        if unfreeze and backbone != 'donkey_cnn':
            logger.info("Unfreezing backbone for fine-tuning...")
            trainer.agent.encoder.unfreeze(num_layers_to_unfreeze=0)

    # Load tub data
    logger.info("Loading Tub data...")
    trainer.load_tub_data(all_tubs, base_model_path=base_model)
    logger.info(f"Replay buffer size: {len(trainer.replay_buffer)}")

    if len(trainer.replay_buffer) == 0:
        logger.error("No valid records loaded. Check your Tub data.")
        sys.exit(1)

    # Train
    logger.info(f"Starting training ({num_epochs} epochs)...")
    history = trainer.train_offline(
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch
    )

    # Save model
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else '.',
        exist_ok=True
    )
    trainer.save(output_path)
    logger.info(f"Model saved to {output_path}")
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
