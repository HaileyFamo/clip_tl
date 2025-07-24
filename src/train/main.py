# src/main.py

"""Main script for training CLIP TunedLens."""

import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path

import wandb
from src.train.train_clip_tl import Train
from src.train.utils import (
    load_config,
    resolve_path_from_config,
    setup_logging,
)


def main(project_root: Path):
    """Main training function."""

    parser = argparse.ArgumentParser(
        description="Train CLIP TunedLens from a YAML config.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs.yaml",
        help="Path to the YAML configuration file, relative to project root.",
    )
    args, unknown = parser.parse_known_args()

    # --- Resolve Paths ---
    # Config path is relative to the project root, or use absolute path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found at: {config_path}")

    # Load config from YAML
    config = load_config(str(config_path))

    # --- Setup Output Directory ---
    output_cfg = config.get("output", {})
    exp_name = output_cfg.get("experiment_name")
    if not exp_name:
        exp_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Output base directory is relative to the project root.
    base_dir = resolve_path_from_config(output_cfg.get("base_dir", "outputs"),
                                        project_root)
    output_dir = base_dir / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Setup Logging ---
    log_cfg = config.get("logging", {})
    log_path = output_dir / "training.log"
    setup_logging(log_cfg.get("level", "INFO"), log_path)

    logger = logging.getLogger(__name__)
    logger.info("Starting training with config: %s", args.config)
    logger.info("Output directory: %s", output_dir)
    logger.info("Log file: %s", log_path)

    # --- Setup WandB ---
    def update_config(original_config, sweep_params):
        for key, value in sweep_params.items():
            if '.' in key:
                keys = key.split('.')
                d = original_config
                for k in keys[:-1]:
                    d = d.setdefault(k, {})
                d[keys[-1]] = value
            else:
                original_config[key] = value

    use_wandb = log_cfg.get("use_wandb", False)
    logger.info(f"use_wandb: {use_wandb}")
    if use_wandb:
        wandb.init(
            project="clip-tl",
            name=config.get("experiment_name"),
            config=config,
            dir=str(config.get("output_dir"))
        )
        sweep_config = wandb.config
        update_config(config, sweep_config)

    # Save the config for this run for reproducibility
    shutil.copyfile(config_path, output_dir / "config.yaml")
    logger.info("Saved config snapshot to %s", output_dir / "config.yaml")

    # --- Create trainer ---
    trainer = Train(
        config=config,
        output_dir=output_dir,
        project_root=project_root
    )

    try:
        logger.info("Starting training...")
        trainer.run()
        logger.info("Training completed successfully.")
        logger.info(f"All training artifacts saved to: {output_dir}.")

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise

    logger.info("All done.")
