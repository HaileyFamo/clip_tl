from huggingface_hub import hf_hub_download, list_repo_files
from vit_prisma.sae import SparseAutoencoder

from src.old_config import *


import logging
import sys
from datetime import datetime

def setup_logging(log_file='experiment.log'):
    logger = logging.getLogger('notebook_logger')
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def load_sae(repo_id, file_name="weights.pt", config_name="config.json") -> SparseAutoencoder:
    sae_path = hf_hub_download(repo_id, file_name, cache_dir=LOCAL_DIR) # Download weights
    hf_hub_download(repo_id, config_name, cache_dir=LOCAL_DIR) # Download config

    print(f"Loading SAE from {sae_path}...")
    sae = SparseAutoencoder.load_from_pretrained(sae_path) # This now automatically gets config.json and converts into the VisionSAERunnerConfig object
    return sae


def load_all_tc(tc_names=TC_NAMES, file_name="weights.pt", config_name="config.json") -> list:
    tc_list = []
    for tc_name in tc_names:
        tc = load_sae(tc_name, file_name, config_name)
        tc_list.append(tc)
    return tc_list