import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib.figure import Figure


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
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def resolve_path_from_config(path_str: str, project_root: Path) -> Path:
    """Resolve a path from a string.

    if path_str is an absolute path, return it as is.
    if path_str is a relative path, resolve it relative to the project root.
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def name_from_hf(model_name: str) -> str:
    """Return the huggingface model name from open-clip model name.

    return model_name.replace("open-clip:", "hf-hub:")
    """
    return model_name.replace("open-clip:", "hf-hub:")


def plot_training_history(
    history: List[Dict[str, Any]],
    output_dir: Union[str, Path],
    exp_name: str
) -> Tuple[Figure, Figure]:

    """Plots training history from a list of records.

    This function generates two plots:
    1. Step-wise training loss.
    2. Averaged training and validation losses with learning rate.

    It saves both plots to the output directory and returns the figure objects.

    Args:
        history: A list of dictionaries, where each dict is a log record.
        output_dir: The directory to save the plot images.
        exp_name: The name of the experiment for the plot titles.

    Returns:
        A tuple containing the two matplotlib figure objects.
    """

    output_dir = Path(output_dir)
    df = pd.DataFrame(history)

    # --- FIGURE 1: Step-wise Training Loss ---
    fig_step, ax_step = plt.subplots(figsize=(15, 6))
    fig_step.suptitle(f'Step-wise Training Loss: {exp_name}', fontsize=16)

    step_df = df.dropna(subset=['train_loss'])
    if not step_df.empty:
        ax_step.plot(
            step_df['step'],
            step_df['train_loss'].interpolate(),
            label='Step Train Loss',
            alpha=0.7,
            color='blue'
        )
        ax_step.set_xlabel('Global Step')
        ax_step.set_ylabel('Loss')
        ax_step.set_yscale('log')
        ax_step.legend()
        ax_step.grid(True, which='both', linestyle='--', linewidth=0.5)

    fig_step.tight_layout(rect=(0, 0.03, 1, 0.95))
    step_loss_path = output_dir / 'training_step_loss.png'
    fig_step.savefig(step_loss_path)
    logging.info(f"Step loss plot saved to {step_loss_path}")

    # --- FIGURE 2: Average Losses ---
    fig_avg, ax_avg1 = plt.subplots(figsize=(15, 6))
    fig_avg.suptitle(f'Average Losses: {exp_name}', fontsize=16)

    avg_df = df.dropna(subset=['avg_train_loss'])
    if not avg_df.empty:
        # Plotting losses on the y-axis
        ax_avg1.set_xlabel('Global Step')
        ax_avg1.set_ylabel('Average Loss')
        ax_avg1.plot(
            avg_df['step'],
            avg_df['avg_train_loss'],
            label='Avg Train Loss',
            marker='o',
            linestyle='-',
            color='blue'
        )
        if ('avg_val_loss' in avg_df.columns and
                not avg_df['avg_val_loss'].isnull().all()):
            ax_avg1.plot(
                avg_df['step'],
                avg_df['avg_val_loss'],
                label='Avg Val Loss',
                marker='x',
                linestyle='--',
                color='orange'
            )
        ax_avg1.set_yscale('log')
        ax_avg1.grid(True, which='major', linestyle='--', linewidth=0.5)
        ax_avg1.legend(loc='upper right')

    fig_avg.tight_layout(rect=(0, 0.03, 1, 0.95))
    avg_loss_path = output_dir / 'training_avg_loss.png'
    fig_avg.savefig(avg_loss_path)
    logging.info(f"Average loss plot saved to {avg_loss_path}")

    return fig_step, fig_avg
