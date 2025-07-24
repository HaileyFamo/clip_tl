import ast
import logging
import random
import sys
from pathlib import Path

import pandas as pd
import yaml
from huggingface_hub import hf_hub_download
from vit_prisma.sae import SparseAutoencoder

from src.analysis.constants import LOCAL_DIR, TC_NAMES
from src.analysis.extra_labels import EXTRA_LABELS


def setup_logging(log_file="experiment.log"):
    logger = logging.getLogger("notebook_logger")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def load_sae(
    repo_id, file_name="weights.pt", config_name="config.json"
) -> SparseAutoencoder:
    sae_path = hf_hub_download(
        repo_id, file_name, cache_dir=LOCAL_DIR
    )  # Download weights
    hf_hub_download(
        repo_id, config_name, cache_dir=LOCAL_DIR
    )  # Download config

    print(f"Loading SAE from {sae_path}...")
    sae = SparseAutoencoder.load_from_pretrained(
        sae_path
    )  # This now automatically gets config.json and converts into the VisionSAERunnerConfig object
    return sae


def load_all_tc(
    tc_names=TC_NAMES, file_name="weights.pt", config_name="config.json"
) -> list[SparseAutoencoder]:
    tc_list = []
    for tc_name in tc_names:
        tc = load_sae(tc_name, file_name, config_name)
        tc_list.append(tc)
    return tc_list


def load_labels_txt(file_path="imagenet-1000.txt") -> list[str]:
    """
    load imagenet 1000 labels and return a list of labels
    """
    with open(file_path, encoding="utf-8") as f:
        content = f.read()
    labels_dict = ast.literal_eval(content)
    labels_list = [labels_dict[i] for i in range(len(labels_dict))]

    return labels_list


def load_words_from_txt(file_path):
    """return a list of words from a txt file"""
    with open(file_path, encoding="utf-8") as f:
        words = [line.strip() for line in f]
    return words


def load_labels_csv(file_path="concreteness_rating.csv") -> list[str]:
    """load concreteness rating labels."""
    df = pd.read_csv(file_path)
    labels = df["Word"].tolist()
    labels = [
        str(label) for label in labels if pd.notna(label) and str(label).strip()
    ]
    return labels


def load_labels(
    file_path="concreteness_rating.csv", num_labels=-1
) -> list[str]:
    """load labels from multiple file types."""

    if "20k" in file_path:
        labels = load_words_from_txt(file_path)
    elif file_path.endswith(".txt"):
        labels = load_labels_txt(file_path)
    elif file_path.endswith(".csv"):
        labels = load_labels_csv(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    if num_labels > 0:
        labels = random.sample(labels, num_labels)
    # add extra labels
    labels.extend(EXTRA_LABELS)
    labels = list(set(labels))
    return labels


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
