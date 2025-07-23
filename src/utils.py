import ast
import logging
from pathlib import Path
import random
from typing import Union, List, Dict, Any, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import sys
import yaml
from huggingface_hub import hf_hub_download
from vit_prisma.sae import SparseAutoencoder
from src.constants import TC_NAMES


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


def load_transcoder(repo_id, file_name="weights.pt", config_name="config.json") -> SparseAutoencoder:
    transcoder_path = hf_hub_download(repo_id, file_name, cache_dir=LOCAL_DIR)
    hf_hub_download(repo_id, config_name, cache_dir=LOCAL_DIR)

    logging.info(f"Loading SAE from {transcoder_path}")
    transcoder = SparseAutoencoder.load_from_pretrained(transcoder_path) 
    # This now automatically gets config.json and converts into the
    # VisionSAERunnerConfig object
    return transcoder


def load_all_tc(tc_names=TC_NAMES, file_name="weights.pt", config_name="config.json") -> List:
    tc_list = []
    for tc_name in tc_names:
        tc = load_sae(tc_name, file_name, config_name)
        tc_list.append(tc)
    return tc_list


def load_labels_txt(file_path: str = 'imagenet-1000.txt') -> List:
    """Load imagenet 1000 labels and return a list of labels."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # parse dict
    labels_dict = ast.literal_eval(content)
    # convert to list, sorted by index
    labels_list = [labels_dict[i] for i in range(len(labels_dict))]

    return labels_list


def load_words_from_txt(file_path: str) -> List:
    """Return a list of words from a txt file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f]

    return words


def load_labels_csv(file_path: str = 'concreteness_rating.csv') -> List:
    """Load concreteness rating labels and return a list of labels."""
    df = pd.read_csv(file_path)
    labels = df['Word'].tolist()
    labels = [str(label) for label in labels if pd.notna(label) and str(label).strip()]
    return labels


def load_labels(file_path: str = 'concreteness_rating.csv', num_labels: int = -1) -> List:
    """Load labels from a file and return a list of labels."""
    if '20k' in file_path:
        labels = load_words_from_txt(file_path)
    elif file_path.endswith('.txt'):
        labels = load_labels_txt(file_path)
    elif file_path.endswith('.csv'):
        labels = load_labels_csv(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    if num_labels > 0:
        labels = random.sample(labels, num_labels)
    # add extra labels
    # labels.extend(EXTRA_LABELS)
    labels = list(set(labels))
    return labels
    