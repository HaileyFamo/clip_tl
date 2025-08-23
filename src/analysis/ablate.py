import random
from typing import Any
from vit_prisma.models.base_vit import HookedTranscoderViT
from vit_prisma.src.sae import SparseAutoencoder
import numpy as np
import torch
from typing import Optional


def get_img_ts(img_path):
    """Transform the image to img tensor for CLIP."""
    from PIL import Image
    from vit_prisma.transforms import get_clip_val_transforms

    img = Image.open(img_path).convert()  # Ensure it's 3 channels
    transforms = get_clip_val_transforms()
    img_tensor = transforms(img)

    return img_tensor


def get_label_embeddings(text_model, labels, chunk_size=1024):
    """Get label embeddings."""
    import open_clip
    prompts = [f'a photo of a {l}' for l in labels]
    label_tokens = open_clip.tokenize(prompts)  # [N, 77] on CPU

    label_embedding = []
    for i in range(0, len(label_tokens), chunk_size):
        chunk = label_tokens[i : i + chunk_size].to(text_model.device, non_blocking=True)
        _label_embedding = text_model.encode_text(chunk)
        _label_embedding = _label_embedding / _label_embedding.norm(dim=-1, keepdim=True)
        label_embedding.append(_label_embedding.cpu())
    label_embedding = torch.cat(label_embedding).to(text_model.device)
    return label_embedding


def zero_ablate_feature_hook(
    feature_activations, hook, feature_ids, position=None
):
    if position is None:
        feature_activations[:, :, feature_ids] = 0
    else:
        feature_activations[:, position, feature_ids] = 0

    return feature_activations


def gaussian_ablate_feature_hook(
    feature_activations,
    hook,
    feature_ids,
    position=None,
    sigma=3.0,
    # n_repeats=10,
    mode='add',
    match_scale=True,
    eps=1e-6,
    seed=42,
) -> torch.Tensor:
    """Add gaussian noise to the feature activations.

    Args:
        - feature_activations: The feature activations after the activation
        function in transcoder.
        - hook: The hook to ablate, should be after the activation function in
        transcoder.
        - feature_ids: The feature ids to ablate. If None, all features are
        ablated.
        - position: The token position to ablate. If None, all positions are
        ablated.
        - sigma: The standard deviation of the gaussian noise. In the ROME
        paper, they use 3.0, and point out that the noise level should be large
        enough to make an effect.
        - mode: The mode to add the gaussian noise. Can be 'add' or 'replace'.
        - match_scale: Whether to match the scale of the feature activations.
            Reccomended to be True since the feature activations can vary.
        - eps: The epsilon to avoid division by zero.
        - seed: The seed to generate the gaussian noise.

    Returns:
        The feature activations after the gaussian noise is added.
    """

    if position is None:
        target_slice = (slice(None), slice(None), feature_ids)
    else:
        target_slice = (slice(None), position, feature_ids)

    target = feature_activations[target_slice]

    # print(f'target shape: {target.shape}')
    # print(f'target sum: {target.sum().item()}')
    # print(f'target max: {target.max().item()}')
    # print(f'target min: {target.min().item()}')
    # print(f'feature_activations shape: {feature_activations.shape}')
    # print(f'feature_activations sum: {feature_activations.sum().item()}')
    # print(f'feature_activations max: {feature_activations.max().item()}')
    # print(f'feature_ids range: {min(feature_ids)} to {max(feature_ids)}')

    if match_scale:
        dims = (0, 1) if target.dim() == 3 else (0,)
        # dims = tuple(range(target.dim()))
        std = (
            target.detach()
            # feature_activations.detach()
            .float()
            .std(dim=dims, keepdim=True, unbiased=False)
            .clamp_min(eps)
            .to(target.device)
        )
        noise_std = sigma * std
    else:
        noise_std = sigma

    # print(f'noise_std: {noise_std}')

    # shape_full = (n_repeats, *target.shape)

    # set seed for reproducibility
    current_rng_state = torch.get_rng_state()
    torch.manual_seed(seed)
    noise = torch.randn_like(target).mul_(noise_std)  # type: ignore
    torch.set_rng_state(current_rng_state)

    # noise = (
    #     torch.randn(shape_full, device=target.device, dtype=target.dtype)
    #     .mul_(noise_std)
    #     .sum(dim=0)
    # )

    # see how much element in target are nonzero
    print(f'{target.nonzero().numel()} / {target.numel()} elements are nonzero')

    if mode == 'add':
        target.add_(noise)

    elif mode == 'replace':
        target.copy_(noise)

    feature_activations[target_slice] = target

    return feature_activations


class AblationExperimentRunner:
    """
    Encapsulates the logic for running a suite of ablation experiments
    on a HookedTranscoderViT model, ensuring comparability between different
    ablation methods by using a feature injection technique. Will run 4 types of
    forward passes:
    - Original CLIP baseline
    - Transcoder baseline
    - Zero Ablation
    - Gaussian Noise Ablation (with n_gn_samples samples)

    The feature activations from the transcoder baseline are used to inject
    zero and Gaussian noise into the model for apple-to-apple comparison.
    """

    def __init__(
        self,
        model: HookedTranscoderViT,
        transcoders: list[SparseAutoencoder],
        img_tensor: torch.Tensor,
        labels: list[str],
        label_embedding: torch.Tensor,
        device='cuda',
    ):
        self.model = model
        self.transcoders = {tc.cfg.hook_point_layer: tc for tc in transcoders}
        self.img_tensor = img_tensor.to(device)
        self.labels = labels
        self.label_embedding = label_embedding.to(device)
        self.device = device
        self.gn_generator = torch.Generator(device=device)
        self.results: dict[str, Any] = {}

    @staticmethod
    def set_seeds(seed: int):
        """Sets random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _format_results(self, probs, idxs) -> dict[str, float]:
        """Formats model output into a dictionary."""
        return {
            self.labels[idx.item()]: prob.item()
            for prob, idx in zip(probs, idxs)
        }

    def _run_forward_pass(
        self, transcoders_to_use, fwd_hooks
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generic helper to run the model and get top predictions."""
        with torch.no_grad():
            img_input = self.img_tensor.unsqueeze(0).to(self.device)
            vis_out = self.model.run_with_hooks_with_transcoders(
                img_input,
                transcoders=transcoders_to_use,
                fwd_hooks=fwd_hooks,
                bwd_hooks=[],
                reset_hooks_end=True,
            )
            image_features = vis_out.to(self.device)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_probs = (
                100.0 * image_features @ self.label_embedding.T
            ).softmax(dim=-1)
            top_probs, top_idx = text_probs.squeeze().topk(20)
        return top_probs, top_idx

    def run_original_clip_baseline(self, seed: int):
        """Runs the baseline CLIP model without any transcoders."""
        print(f'--- [Seed {seed}] Running Original CLIP Baseline ---')
        self.set_seeds(seed)
        self.model.eval()
        # This resets all transcoders, restoring the original MLP layers
        self.model.reset_transcoders()
        top_probs, top_idx = self._run_forward_pass(
            transcoders_to_use=[], fwd_hooks=[]
        )
        self.results[f'original_clip_seed{seed}'] = self._format_results(
            top_probs, top_idx
        )
        print('Baseline Done.')

    def run_ablation_suite(
        self,
        layer_idx: int,
        ablation_feature_ids: Optional[list[int]],
        seed: int,
        n_gn_samples: int = 10,
        sigma: float = 1.0,
    ):
        """
        Runs the full suite of experiments for a single layer and seed.
        1. Captures baseline activations with the transcoder.
        2. Runs zero and Gaussian noise ablations by injecting modified versions
           of the captured activations.

        Args:
            layer_idx: The layer index to run the experiments on.
            n_gn_samples: The number of Gaussian noise samples to run.
            sigma: The standard deviation of the Gaussian noise.
        """
        self.set_seeds(seed)
        self.model.eval()

        if layer_idx not in self.transcoders:
            raise ValueError(f'No transcoder found for layer {layer_idx}')
        transcoder = self.transcoders[layer_idx]
        hook_point = f'blocks.{layer_idx}.mlp.hook_hidden_post'

        # --- Step 1: Capture baseline activations & get baseline transcoder results ---
        print(
            f'\n--- [Seed {seed}] Running Transcoder Baseline & Capturing Activations for L{layer_idx} ---'
        )
        captured_activations = None

        def capture_hook(activations, hook):
            nonlocal captured_activations
            captured_activations = activations.clone().detach()

        top_probs, top_idx = self._run_forward_pass(
            transcoders_to_use=[transcoder],
            fwd_hooks=[(hook_point, capture_hook)],
        )
        if captured_activations is None:
            raise RuntimeError('Failed to capture activations.')
        self.results[f'transcoder_L{layer_idx}_seed{seed}'] = (
            self._format_results(top_probs, top_idx)
        )
        print('Activations captured.')

        # --- Step 2: Run ablation experiments using injection ---

        # 2a. Zero Ablation
        print(f'--- [Seed {seed}] Running Zero Ablation for L{layer_idx} ---')

        def inject_hook_zero_ablation(activations, hook):
            modified_activations = captured_activations.clone()
            return zero_ablate_feature_hook(
                modified_activations, hook, feature_ids=ablation_feature_ids
            )

        top_probs, top_idx = self._run_forward_pass(
            transcoders_to_use=[transcoder],
            fwd_hooks=[(hook_point, inject_hook_zero_ablation)],
        )
        self.results[f'zero_ablation_L{layer_idx}_seed{seed}'] = (
            self._format_results(top_probs, top_idx)
        )
        print('Zero Ablation Done.')

        # 2b. Gaussian Noise Ablation
        print(
            f'--- [Seed {seed}] Running Gaussian Noise Ablation for L{layer_idx} ({n_gn_samples} samples) ---'
        )
        gn_results_list = []
        for i in range(n_gn_samples):
            gn_seed = seed * 100 + i

            def inject_hook_gn_ablation(activations, hook):
                modified_activations = captured_activations.clone()
                return gaussian_ablate_feature_hook(
                    modified_activations,
                    hook,
                    feature_ids=ablation_feature_ids,
                    sigma=sigma,
                    match_scale=True,
                    seed=gn_seed,
                )

            top_probs, top_idx = self._run_forward_pass(
                transcoders_to_use=[transcoder],
                fwd_hooks=[(hook_point, inject_hook_gn_ablation)],
            )
            gn_results_list.append(self._format_results(top_probs, top_idx))

        self.results[f'gn_ablation_L{layer_idx}_seed{seed}'] = gn_results_list
        print('GN Ablation Done.')


