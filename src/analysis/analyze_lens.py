# src/analyze_lens.py

import logging
from datetime import datetime
from pathlib import Path
from typing import Union

import torch
from PIL import Image
from vit_prisma.models.model_loader import load_hooked_model
from vit_prisma.sae import SparseAutoencoder
from vit_prisma.transforms import get_clip_val_transforms
from vit_prisma.utils.enums import ModelType

from src.analysis.circuit_analysis import (
    get_paths_via_filter,
    greedy_get_top_paths,
)
from src.analysis.components import (
    ComponentType,
    FeatureFilter,
    make_transcoder_feature_vector,
)
from src.analysis.deembedding import print_deembeddings_for_all_paths
from src.analysis.utils import (
    hf_to_open_clip,
    load_all_tc,
    load_config,
    load_vocab,
    resolve_path_from_config,
    setup_logging,
)
from src.tuned_lens.clip_tl import CLIPTunedLens
from src.tuned_lens.ingredients import CLIPModel

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
        self._setup_device()
        self._setup_model_and_tokenizer()
        self._setup_lens()
        # self._replace_visual_module()
        self._setup_transcoders()

    def _setup_device(self):
        """Set device based on configuration."""
        device = self.config.get('device', 'auto')
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f'Using device: {device}')
        self.device = device

    def _setup_model_and_tokenizer(self) -> None:
        """Load CLIP model, preprocessor and tokenizer.

        Replace the visual part with HookViT from prisma.
        """
        logger.info(f'Loading model: {self.model_cfg["name"]}')
        self.model_manager = CLIPModel(
            model_name=self.model_cfg['name'], device=self.device
        )
        self.model, self.preprocess = self.model_manager.load()
        self.tokenizer = self.model_manager.get_tokenizer()
        assert self.model is not None, 'model not loaded.'
        assert self.preprocess is not None, 'preprocess not loaded.'
        assert self.tokenizer is not None, 'tokenizer not loaded.'

    def _setup_lens(self) -> None:
        """Load Tuned Lens from checkpoint."""
        lens_path = resolve_path_from_config(
            self.lens_cfg['lens_path'], self.project_root
        )
        if not lens_path.exists():
            raise FileNotFoundError(f'Tuned Lens path not found: {lens_path}')

        logger.info(f'Loading Tuned Lens from: {lens_path}')
        if lens_path.is_dir():
            self.lens = CLIPTunedLens.from_lens_path(
                lens_path, self.model_manager
            )
        elif lens_path.is_file():
            self.lens = CLIPTunedLens.from_checkpoint(
                lens_path, self.model_manager
            )
        else:
            raise ValueError(f'Invalid lens path: {lens_path}')

    # @deprecated('Only use hookedvit once for cache and delete it.')
    def _replace_visual_module(self):
        """Replace the visual module with HookViT from prisma."""
        hooked_model = load_hooked_model(
            hf_to_open_clip(self.model_cfg['name']),
            model_type=ModelType.VISION,
        )
        # copy the state dict of the original model to the hooked model
        # hooked_model.load_state_dict(self.model.state_dict())
        hooked_model.to(self.device)
        hooked_model.eval()
        self.model.visual = hooked_model

    def _get_text_embedding(self, text_input: Union[str, list[str]]):
        """Get text embedding for a small text input str or list."""
        if isinstance(text_input, str):
            text_tokens = self.tokenizer([text_input])
        elif isinstance(text_input, list):
            text_tokens = self.tokenizer(text_input)
        else:
            raise ValueError(f'Invalid text input type: {type(text_input)}')
        text_tokens = text_tokens.to(self.device)
        with torch.no_grad():
            text_embedding = self.model.encode_text(text_tokens)
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
        return text_embedding

    def _get_vocab_embeddings(self, labels: list[str], chunk_size: int = 512):
        """Get vocabulary embeddings (for all labels) after applying template.

        This is the same as the `_get_text_embedding` function, but for all
        labels.
        """
        prompts = [f'a photo of a {label}' for label in labels]
        vocab_tokens = self.tokenizer(prompts)
        vocab_tokens = vocab_tokens.to(self.device)
        with torch.no_grad():
            _vocab_embeddings = []
            for i in range(0, len(vocab_tokens), chunk_size):
                chunk = vocab_tokens[i : i + chunk_size].to(
                    self.device, non_blocking=True
                )
                feats = self.model.encode_text(chunk)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                _vocab_embeddings.append(feats)
        vocab_embeddings = torch.cat(_vocab_embeddings)
        self.vocab_embeddings = vocab_embeddings

    def _setup_transcoders(self) -> None:
        """Load all Transcoders."""
        logger.info('Loading all Transcoders...')
        self.transcoders = load_all_tc()
        self.transcoders = [tc.to(self.device) for tc in self.transcoders]
        logger.info(f'Successfully loaded {len(self.transcoders)} Transcoders.')

    def _prepare_single_input_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get image tensor and label embedding for a single image."""

        image_path = resolve_path_from_config(
            self.analysis_cfg['image_path'], self.project_root
        )
        logger.info(f'Loading image: {image_path}')
        image = Image.open(image_path).convert('RGB')
        transforms = get_clip_val_transforms()
        img_tensor = transforms(image).unsqueeze(0).to(self.device)

        target_label = self.analysis_cfg['target_label']
        logger.info(f"Target abstract concept: '{target_label}'")
        label_tokens = self.tokenizer([target_label]).to(self.device)
        with torch.no_grad():
            assert self.model is not None, 'model not loaded'
            label_embedding = self.model.encode_text(label_tokens)
            label_embedding /= label_embedding.norm(dim=-1, keepdim=True)

        return img_tensor, label_embedding

    def _get_decoders(
        self, layer_idx: int = -1
    ) -> Union[torch.Tensor, torch.nn.Module]:
        """Get the decoder matrix of a transcoder for a given layer."""
        tc_target_layer = self.transcoders[layer_idx]
        decoder_vectors = tc_target_layer.W_dec.T
        return decoder_vectors

    def _get_label_target_feature(
        self, decoder_vectors: Union[torch.Tensor, torch.nn.Module]
    ) -> int:
        """Get the most relevant decoder feature index for a given label."""

        projected_decoder_vectors = self.lens.unembed.projection(
            decoder_vectors.T
        )  # this use the original projection matrix
        similarities = self.vocab_embeddings @ projected_decoder_vectors.T
        _, top_indices = torch.topk(similarities.squeeze(), k=1)
        return int(top_indices[0].item())

    def _run_with_cache(self, model_input: torch.Tensor):
        """Run the batch through the model to get activations"""
        hooked_model = load_hooked_model(
            hf_to_open_clip(self.model_cfg['name']),
            model_type=ModelType.VISION,
        )
        hooked_model.to(self.device)
        hooked_model.eval()

        logits, cache = hooked_model.run_with_cache(model_input)
        del hooked_model
        return logits, cache

    def _get_feature_activations(
        self, transcoder: SparseAutoencoder, cache
    ) -> torch.Tensor:
        """Compute the activation given a cache.

        If cache is not provided, run the model with cache.
        we only need the activations of the hook point, so here use .encode()
        to get the activations instead of a forward pass.
        """
        hook_point_activation = cache[transcoder.cfg.hook_point].to(self.device)
        _, feature_acts, *_ = transcoder.encode(hook_point_activation)

        return feature_acts

    def run_single_input(self) -> None:
        """Execute core analysis process."""

        target_layer_idx = self.analysis_cfg['target_layer_idx']
        top_k = self.analysis_cfg['top_k']

        logger.info(f'Analyzing features of layer {target_layer_idx}...')

        assert self.model is not None, 'model not loaded'

        img_tensor, label_embedding = self._prepare_single_input_data()

        decoder_vectors = self._get_decoders(target_layer_idx)

        # --- get vocab embeddings ---
        label_dict = load_vocab(
            self.analysis_cfg['vocab_path'], return_dict=True
        )
        self._get_vocab_embeddings(list(label_dict.keys()))

        # --- get label target feature and build feature vector ---
        label_target_feature_idx = self._get_label_target_feature(
            decoder_vectors
        )
        feature_vector = make_transcoder_feature_vector(
            self.transcoders[target_layer_idx], label_target_feature_idx
        )

        # --- get cache and logits ---
        logits, cache = self._run_with_cache(img_tensor)

        # --- build the paths ---
        all_paths = greedy_get_top_paths(
            self.model,
            self.transcoders,
            cache,
            feature_vector,
            num_iters=self.analysis_cfg['num_iters'],
            num_branches=self.analysis_cfg['num_branches'],
        )

        # filter paths and only keep paths that end in layer 0
        filtered_paths = get_paths_via_filter(
            all_paths,
            suffix_path=[
                FeatureFilter(component_type=ComponentType.EMBED, layer=0)
            ],
        )

        # print the paths
        print_deembeddings_for_all_paths(
            self.vocab_embeddings, label_dict, filtered_paths, self.lens
        )


def main(project_root: Path, config_path: Path) -> None:
    """Main analysis function, responsible for parsing parameters, loading
    configuration, and starting the analysis process.

    Args:
        project_root: Project root directory.
        config_path: Path to the analysis configuration file.
    """

    # --- parse path and load config ---
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    if not config_path.is_file():
        raise FileNotFoundError(f'Config file not found: {config_path}')
    config = load_config(config_path)

    # --- setup output directory ---
    output_cfg = config.get('output', {})
    exp_name = output_cfg.get('experiment_name')
    if not exp_name:
        exp_name = f'analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    base_dir = resolve_path_from_config(
        output_cfg.get('base_dir', 'analysis_results'), project_root
    )
    output_dir = base_dir / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- setup logging ---
    log_cfg = config.get('logging', {})
    log_path = output_dir / 'analysis.log'
    setup_logging(log_path, log_cfg.get('level', 'INFO'))

    logger.info(f'Analysis config: {config_path}')
    logger.info(f'Output directory: {output_dir}')
    logger.info(f'Log file: {log_path}')

    # --- run analysis ---
    try:
        runner = AnalysisRunner(
            config=config, project_root=project_root, output_dir=output_dir
        )
        runner.run_single_input()
        logger.info('Analysis completed successfully.')
    except Exception as e:
        logger.error(f'Analysis failed: {e}', exc_info=True)
        raise
