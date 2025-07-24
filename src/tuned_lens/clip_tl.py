"""CLIP Tuned Lens implementation"""

import abc
import inspect
import json
import logging
from collections.abc import Generator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import open_clip
import torch
from open_clip.model import VisionTransformer

from src.tuned_lens.ingredients import CLIPModel, Unembed

logger = logging.getLogger(__name__)


class Lens(abc.ABC, torch.nn.Module):
    """Abstract base class for all Lens."""

    unembed: Unembed

    def __init__(self, unembed: Unembed):
        """Create a Lens.

        Args:
            unembed: The unembed operation to use.
        """
        super().__init__()

        self.unembed = unembed

    @abc.abstractmethod
    def transform_hidden(self, h: torch.Tensor, idx: int) -> torch.Tensor:
        """Convert a hidden state to the final hidden just before the
        unembedding.

        Args:
            h: The hidden state to convert.
            idx: The layer of the transformer these hidden states come from.
        """
        ...

    @abc.abstractmethod
    def forward(self, h: torch.Tensor, idx: int) -> torch.Tensor:
        """Decode hidden states into logits."""
        ...


@dataclass
class TunedLensConfig:
    """A configuration for a TunedLens.

    Attributes:
        base_model_name_or_path: The name of the base model.
        d_model: The hidden size of the base model.
        num_hidden_layers: The number of layers in the base model.
        bias: Whether to use a bias in the linear translators.
        dtype: The dtype of the linear translators.
    """

    base_model_name_or_path: str
    d_model: int
    num_hidden_layers: int
    bias: bool = True
    dtype: torch.dtype = torch.float32

    def to_dict(self):
        """Convert this config to a dictionary."""
        config_dict = asdict(self)
        # Convert torch.dtype to string for JSON serialization
        if "dtype" in config_dict and isinstance(
            config_dict["dtype"], torch.dtype
        ):
            config_dict["dtype"] = str(config_dict["dtype"])
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict):
        """Create a config from a dictionary."""
        config_dict = deepcopy(config_dict)
        # Drop unrecognized config keys
        unrecognized = set(config_dict) - set(inspect.getfullargspec(cls).args)
        for key in unrecognized:
            logger.warning(f"Ignoring config key '{key}'")
            del config_dict[key]

        # Convert string back to torch.dtype
        if "dtype" in config_dict and isinstance(config_dict["dtype"], str):
            try:
                # e.g., "torch.float32" -> torch.float32
                config_dict["dtype"] = getattr(
                    torch, config_dict["dtype"].split(".")[-1]
                )
            except AttributeError:
                logger.warning(
                    f"Could not convert '{config_dict['dtype']}' to a "
                    f"torch.dtype. Ignoring."
                )
                del config_dict["dtype"]
        return cls(**config_dict)


class CLIPTunedLens(Lens):
    """Translate intermediate hidden states to the final layer.

    This class only trains the linear translators. However the final layernorm
    and the unembed are also included from the original CLIP model.

    Attributes:
        config: The configuration for the lens.
        unembed: The unembed operation to use.
        layer_translators: A list of layer translators.
    """

    config: TunedLensConfig
    unembed: Unembed
    layer_translators: torch.nn.ModuleList

    def __init__(self, config: TunedLensConfig, unembed: Unembed):
        super().__init__(unembed)
        self.config = config

        # # The unembedding might be int8 if we're using bitsandbytes
        # w = unembed.unembedding.weight
        # dtype = w.dtype if torch.is_floating_point(w) else torch.float16

        # Set up the linear translator configs
        translator = torch.nn.Linear(
            config.d_model, config.d_model, bias=config.bias, dtype=config.dtype
        )
        translator.weight.data.zero_()
        translator.bias.data.zero_()

        # Don't include the final layer since it does not need a translator
        self.layer_translators = torch.nn.ModuleList(
            [deepcopy(translator) for _ in range(self.config.num_hidden_layers)]
        )

    def __getitem__(self, item: int) -> torch.nn.Module:
        """Get the probe module at the given index layer."""
        return self.layer_translators[item]

    def __iter__(self) -> Generator[torch.nn.Module, None, None]:
        """Get iterator over the translators within the lens."""
        yield from self.layer_translators

    def __len__(self) -> int:
        """Return the number of layer translators in the lens."""
        return len(self.layer_translators)

    @classmethod
    def from_clip_model(
        cls, clip_model: CLIPModel, bias: bool = True
    ) -> "CLIPTunedLens":
        """Create a lens from a CLIPModel instance.

        Args:
            clip_model: CLIPModel instance (from ingredients.py)
        """
        # get loaded model and config
        actual_model = clip_model.get_model()
        config_dict = clip_model.get_config()

        # create TunedLensConfig
        config = TunedLensConfig(
            base_model_name_or_path=config_dict["base_model_name_or_path"],
            d_model=config_dict["d_model"],
            num_hidden_layers=config_dict["num_hidden_layers"],
            dtype=config_dict.get("dtype", torch.float32),
            bias=bias,
        )
        # create unembed (using actual model)
        unembed = Unembed(actual_model)

        return cls(config, unembed)

    @classmethod
    def from_model(cls, model, bias: bool = True) -> "CLIPTunedLens":
        """Create a lens from a raw model (deprecated, use from_clip_model
        instead)."""
        # for backward compatibility
        return cls.from_clip_model(model, bias)

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: Union[str, Path], clip_model: CLIPModel
    ) -> "CLIPTunedLens":
        """Create a lens from a checkpoint."""

        logger.info(f"Loading lens from checkpoint: {checkpoint_path}")
        lens = cls.from_clip_model(clip_model)
        checkpoint = torch.load(checkpoint_path, map_location=clip_model.device)
        assert (
            "lens_state_dict" in checkpoint
        ), "Checkpoint does not contain lens state dict"
        lens.load_state_dict(checkpoint["lens_state_dict"])
        lens.to(clip_model.device)
        lens.eval()

        return lens

    @classmethod
    def from_pretrained_model(
        cls, path: Union[str, Path], model: open_clip.model.CLIP
    ) -> "CLIPTunedLens":
        """Load the lens from a model directory"""
        path = Path(path)
        config_path = path / "config.json"
        params_path = path / "params.pt"

        if not config_path.exists() or not params_path.exists():
            raise FileNotFoundError(
                f"Lens directory '{path}' must contain 'config.json' and "
                f"'params.pt'."
            )

        with open(config_path) as f:
            config_dict = json.load(f)
        config = TunedLensConfig.from_dict(config_dict)

        unembed = Unembed(model)
        # Create a new lens instance with the loaded config and unembed layer
        lens = cls(config, unembed)
        # Load the translator weights
        state_dict = torch.load(params_path, map_location="cuda")
        lens.layer_translators.load_state_dict(state_dict)
        # lens.to(clip_model.device)
        lens.eval()

        logger.info(f"Loaded lens from {path}")
        return lens

    @classmethod
    def from_lens_path(
        cls, path: Union[str, Path], clip_model: CLIPModel
    ) -> "CLIPTunedLens":
        """Load the lens from a lens directory.

        This method assumes that the directory contains 'config.json' and
        'params.pt' files, saved by the 'save' method.

        Args:
            path: Path to the lens directory.
            clip_model: The CLIPModel instance to use with the lens.

        Returns:
            A CLIPTunedLens instance loaded from the directory.
        """
        path = Path(path)
        config_path = path / "config.json"
        params_path = path / "params.pt"

        if not config_path.exists() or not params_path.exists():
            raise FileNotFoundError(
                f"Lens directory '{path}' must contain 'config.json' and "
                f"'params.pt'."
            )

        with open(config_path) as f:
            config_dict = json.load(f)
        config = TunedLensConfig.from_dict(config_dict)

        unembed = Unembed(clip_model.get_model())
        # Create a new lens instance with the loaded config and unembed layer
        lens = cls(config, unembed)
        # Load the translator weights
        state_dict = torch.load(params_path, map_location=clip_model.device)
        lens.layer_translators.load_state_dict(state_dict)

        lens.to(clip_model.device)
        lens.eval()

        logger.info(f"Loaded lens from {path}")
        return lens

    def transform_hidden(self, h: torch.Tensor, idx: int) -> torch.Tensor:
        """Transform hidden state from layer `idx`."""
        # Note that we add the translator output residually, in contrast to the
        # formula in the paper. By parametrizing it this way we ensure that
        # weight decay regularizes the transform toward the identity, not the
        # zero transformation.
        return h + self[idx](h)

    def forward(self, h: torch.Tensor, idx: int) -> torch.Tensor:
        """Transform and then decode the hidden states into logits."""
        h = self.transform_hidden(h, idx)
        return self.unembed.forward(h)

    def save(self, dir: Union[str, Path]) -> None:
        """Save the lens to a directory"""
        dir = Path(dir)
        dir.mkdir(exist_ok=True, parents=True)
        state_dict = self.layer_translators.state_dict()

        # save the parameters
        torch.save(state_dict, dir / "params.pt")
        # save the config
        with open(dir / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)


class HiddenStatesHook:
    """Hook to extract hidden states from the CLIP model."""

    def __init__(self):
        self.hidden_states = []

    def hook(self, module, input, output):
        # the output of each transformer block is the hidden state we want
        self.hidden_states.append(output.detach())

    def clear(self):
        self.hidden_states = []


@contextmanager
def get_clip_hidden_states(
    model: open_clip.CLIP,
    images: torch.Tensor,
) -> Generator[Tuple[torch.Tensor, List[torch.Tensor]], None, None]:
    """Use hooks to extract the CLIP hidden states.

    Args:
        model: The CLIP model.
        images: The images to encode, need to be tensor after preprocess.

    Returns:
        The final output and the hidden states.
    """

    hook_handler = HiddenStatesHook()
    handles = []  # used to control the lifetime of the hooks

    assert isinstance(model.visual, VisionTransformer)
    for block in model.visual.transformer.resblocks:
        # use handle for later removal
        handle = block.register_forward_hook(hook_handler.hook)
        handles.append(handle)

    try:
        with torch.no_grad():
            final_output = model.encode_image(images)
        hidden_states = hook_handler.hidden_states

        yield final_output, hidden_states

    except Exception as e:
        logger.error(f"Error getting hidden states: {e}")
        raise e

    finally:
        for handle in handles:
            handle.remove()
        hook_handler.clear()
