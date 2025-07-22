#!/usr/bin/env python3
"""Main script for training CLIP TunedLens."""

import argparse
import logging
import sys
import shutil
import yaml
from pathlib import Path
from typing import Union
from datetime import datetime

# --- Start of Path Setup ---
# Find the project root directory, which is the parent of the 'src' directory
# This makes the script runnable from anywhere.
_project_root = Path(__file__).resolve().parent.parent

# Add the project root to the Python path to allow for absolute imports
# from within the project.
sys.path.insert(0, str(_project_root))

# Add src to path to allow for relative imports, though absolute is preferred.
_src_path = _project_root / 'src'
sys.path.insert(0, str(_src_path))
# --- End of Path Setup ---


from train_clip_tl import Train  # noqa: E402
from ingredients import CLIPModel, ImageData, Optimizer  # noqa: E402


def setup_logging(log_level: str = "INFO",
                  log_path: Union[str, Path] = "./training.log") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_path))
        ]
    )


def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train CLIP TunedLens from a YAML config.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs.yaml",
        help="Path to the YAML configuration file, relative to project root.",
    )
    args = parser.parse_args()

    # --- Resolve Paths ---
    # Config path is relative to the project root.
    config_path = _project_root / args.config
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found at: {config_path}")

    # Load config from YAML
    config = load_config(config_path)

    # --- Setup Output Directory ---
    output_cfg = config.get("output", {})
    exp_name = output_cfg.get("experiment_name")
    if not exp_name:
        exp_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Output base directory is relative to the project root.
    base_dir = _project_root / output_cfg.get("base_dir", "outputs")
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

    # Save the config for this run for reproducibility
    shutil.copyfile(config_path, output_dir / "config.yaml")
    logger.info("Saved config snapshot to %s", output_dir / "config.yaml")

    # --- Setup Device ---
    device = config["training"].get("device", "auto")
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Create configuration objects from config ---
    model = CLIPModel(
        model_name=config["model"]["name"],
        device=device
    )
    # Data path is relative to the project root
    data_path = config["data"]["path"]
    data = ImageData(
        data_path=data_path,
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        validation_split=config["data"].get("validation_split", 0.0)
    )

    optimizer = Optimizer(
        optimizer=config["optimizer"]["name"],
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"]["weight_decay"],
        momentum=config["optimizer"].get("momentum", 0.9)
    )

    # --- Create trainer ---
    trainer = Train(
        config=config,
        model=model,
        data=data,
        optimizer=optimizer,
        project_root=_project_root,
    )

    try:
        # Start training
        logger.info("Starting training...")
        trainer.run()
        logger.info("Training completed successfully!")

        # Save final model
        final_model_path = output_dir / "final_model"
        # Note: This will be implemented when save method is available
        logger.info(f"Final model would be saved to: {final_model_path}")
        logger.info(f"All training artifacts saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise

    logger.info("All done!")


if __name__ == "__main__":
    main()
