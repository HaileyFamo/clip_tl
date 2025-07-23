# src/analyze_lens.py

import argparse
import json
import logging
import open_clip
import torch
from datetime import datetime
from pathlib import Path
from PIL import Image
from vit_prisma.transforms import get_clip_val_transforms
from src.clip_tl import CLIPTunedLens, get_clip_hidden_states
from src.ingredients import CLIPModel
from src.utils import (load_all_tc, load_config, resolve_path_from_config,
                       setup_logging)

logger = logging.getLogger(__name__)


class AnalysisRunner:
    """Runner for CLIP model analysis."""

    def __init__(self, config: dict, project_root: Path, output_dir: Path):
        """Initialize AnalysisRunner.

        Args:
            config: Configuration dictionary loaded from YAML file.
            project_root: Project root directory.
            output_dir: Output directory for storing results.
        """
        self.config = config
        self.project_root = project_root
        self.output_dir = output_dir

        # --- extract config ---
        self.model_cfg = self.config.get('model', {})
        self.lens_cfg = self.config.get('lens', {})
        self.analysis_cfg = self.config.get('analysis', {})

        # --- setup components ---
        self.device = self._setup_device()
        self._setup_model_and_tokenizer()
        self._setup_lens()
        self._setup_transcoders()
        self._prepare_input_data()

    def _setup_device(self) -> str:
        """Set device based on configuration."""
        device = self.config.get('device', 'auto')
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f'Using device: {device}')
        return device

    def _setup_model_and_tokenizer(self) -> None:
        """Load CLIP model, preprocessor and tokenizer."""
        logger.info(f"Loading model: {self.model_cfg['name']}")
        self.model_manager = CLIPModel(
            model_name=self.model_cfg['name'], device=self.device)
        self.model, self.preprocess = self.model_manager.load()
        assert self.model is not None, 'model not loaded.'
        assert self.preprocess is not None, 'preprocess not loaded.'
        self.tokenizer = self.model_manager.get_tokenizer()

    def _setup_lens(self) -> None:
        """Load Tuned Lens from checkpoint."""
        lens_checkpoint_path = resolve_path_from_config(
            self.lens_cfg['checkpoint'], self.project_root)
        if not lens_checkpoint_path.exists():
            raise FileNotFoundError(
                f'Tuned Lens checkpoint not found: {lens_checkpoint_path}')

        logger.info(f'Loading Tuned Lens from: {lens_checkpoint_path}')
        self.lens = CLIPTunedLens.from_checkpoint(
            lens_checkpoint_path, self.model_manager)
        
    def _get_text_embedding(self):


    def _setup_transcoders(self) -> None:
        """Load all Transcoders."""
        logger.info('Loading all Transcoders...')
        self.transcoders = load_all_tc()
        logger.info(f'Successfully loaded {len(self.transcoders)} Transcoders.')

    def _prepare_input_data(self) -> None:
        """Prepare input data for analysis."""

        image_path = resolve_path_from_config(
            self.analysis_cfg['image_path'], self.project_root)
        logger.info(f'Loading image: {image_path}')
        image = Image.open(image_path).convert('RGB')
        # assert self.preprocess is not None, 'preprocess not loaded'
        # self.image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        transforms = get_clip_val_transforms()
        self.img_tensor = transforms(image).unsqueeze(0).to(self.device)

        target_label = self.analysis_cfg['target_label']
        logger.info(f"Target abstract concept: '{target_label}'")
        text_input = self.tokenizer([target_label]).to(self.device)
        with torch.no_grad():
            assert self.model is not None, 'model not loaded'
            text_embedding = self.model.encode_text(text_input)
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
            self.text_embedding = text_embedding

    def run(self) -> None:
        """Execute core analysis process."""
        target_layer_idx = self.analysis_cfg['target_layer_idx']
        top_k = self.analysis_cfg['top_k']

        logger.info(f'Analyzing features of layer {target_layer_idx}...')

        assert self.model is not None, 'model not loaded'
        with get_clip_hidden_states(self.model.visual, self.img_tensor) as (
            _, hidden_states):

            # --- get target layer transcoder and lens translator ---
            tc_target_layer = self.transcoders[target_layer_idx]
            translator = self.lens.layer_translators[target_layer_idx]

            # --- get decoder matrix and project with Tuned Lens ---
            decoder_vectors = tc_target_layer.W_dec.T
            projected_vectors = decoder_vectors + translator(
                decoder_vectors.T).T
            logger.info(f'Used Tuned Lens to project decoder vectors of layer {target_layer_idx}.')

            # --- map projected visual features to multimodal shared space ---
            assert self.model.visual.proj is not None, 'visual projection layer not found'
            final_vectors = self.model.visual.proj.T @ projected_vectors
            final_vectors /= final_vectors.norm(dim=-1, keepdim=True)
            logger.info('Mapped projected vectors to multimodal shared space.')

            # --- calculate similarity with target text ---
            similarities = self.text_embedding @ final_vectors

            # --- find most relevant features ---
            top_sims, top_indices = torch.topk(
                similarities.squeeze(), k=top_k)

            # --- show and save results ---
            results = {
                'target_layer': target_layer_idx,
                'target_label': self.analysis_cfg['target_label'],
                'top_features': []
            }
            header = (f'After using Tuned Lens, the most relevant Top {top_k} '
                      f'features (layer {target_layer_idx}) for '
                      f'"{self.analysis_cfg["target_label"]}":')
            print('\n' + '=' * 60)
            print(header)
            print('-' * 60)

            for i in range(top_k):
                idx = top_indices[i].item()
                sim = top_sims[i].item()
                results['top_features'].append({'index': idx, 'similarity': sim})
                print(f'  - feature index: {idx:<6} | cosine similarity: {sim:.4f}')
            print('=' * 60 + '\n')

            # --- save results to JSON file ---
            results_path = self.output_dir / 'analysis_results.json'
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            logger.info(f'Analysis results saved to: {results_path}')


def main(project_root: Path) -> None:
    """Main analysis function, responsible for parsing parameters, loading
    configuration, and starting the analysis process.

    Args:
        project_root: Project root directory.
    """
    parser = argparse.ArgumentParser(
        description='Analyze CLIP model using Tuned Lens and Transcoder.')
    parser.add_argument(
        '--config',
        type=str,
        default='clip_tl/analyze_config.yaml',
        help='Path to analysis configuration file (relative to project root).',
    )
    args = parser.parse_args()

    # --- parse path and load config ---
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    if not config_path.is_file():
        raise FileNotFoundError(f'Config file not found: {config_path}')
    config = load_config(str(config_path))

    # --- setup output directory ---
    output_cfg = config.get('output', {})
    exp_name = output_cfg.get('experiment_name')
    if not exp_name:
        exp_name = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base_dir = resolve_path_from_config(
        output_cfg.get('base_dir', 'analysis_results'), project_root)
    output_dir = base_dir / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- setup logging ---
    log_cfg = config.get('logging', {})
    log_path = output_dir / 'analysis.log'
    setup_logging(log_cfg.get('level', 'INFO'), log_path)

    logger.info(f'Analysis config: {args.config}')
    logger.info(f'Output directory: {output_dir}')
    logger.info(f'Log file: {log_path}')

    # --- run analysis ---
    try:
        runner = AnalysisRunner(
            config=config, project_root=project_root, output_dir=output_dir)
        runner.run()
        logger.info('Analysis completed successfully.')
    except Exception as e:
        logger.error(f'Analysis failed: {e}', exc_info=True)
        raise


if __name__ == '__main__':
    # assume this script is run in the parent directory of the project root,
    # or PYTHONPATH is correctly set
    # (e.g., the parent directory of 'clip_tl' is project_root)
    _project_root = Path(__file__).resolve().parents[1]
    main(_project_root)
