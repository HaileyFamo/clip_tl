"""CLIP Tuned Lens implementation"""
import abc
from contextlib import contextmanager
import json
from pathlib import Path
import torch
from typing import List, Dict, Generator, Union, Tuple
from dataclasses import dataclass, asdict
from copy import deepcopy
import inspect
import logging
import open_clip
from src.ingredients import CLIPModel, Unembed
# from model_surgery import *

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

    Args:
        base_model_name_or_path: The name of the base model.
        d_model: The hidden size of the base model.
        num_hidden_layers: The number of layers in the base model.
        bias: Whether to use a bias in the linear translators.
        dtype: The dtype of the linear translators.

    Returns:
        A TunedLensConfig instance.
    """

    base_model_name_or_path: str
    d_model: int
    num_hidden_layers: int
    bias: bool = True
    dtype: torch.dtype = torch.float32
    # # The revision of the base model this lens was tuned for.
    # base_model_revision: Optional[str] = None
    # # The hash of the base's unembed model this lens was tuned for.
    # unembed_hash: Optional[str] = None
    # # The name of the lens type.
    # lens_type: str = "linear_tuned_lens"

    def to_dict(self):
        """Convert this config to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict):
        """Create a config from a dictionary."""
        config_dict = deepcopy(config_dict)
        # Drop unrecognized config keys
        unrecognized = set(config_dict) - set(inspect.getfullargspec(cls).args)
        for key in unrecognized:
            logger.warning(f"Ignoring config key '{key}'")
            del config_dict[key]

        return cls(**config_dict)


class CLIPTunedLens(Lens):
    """CLIP Tuned Lens.

    Args:
        config: The configuration for the lens.
        unembed: The unembed operation to use.
        layer_translators: A list of layer translators.

    Returns:
        A CLIP Tuned Lens instance.
    """

    config: TunedLensConfig
    unembed: Unembed  # TODO
    layer_translators: torch.nn.ModuleList

    def __init__(self, config: TunedLensConfig, unembed: Unembed):
        super().__init__(unembed)
        self.config = config

        # # The unembedding might be int8 if we're using bitsandbytes
        # w = unembed.unembedding.weight
        # dtype = w.dtype if torch.is_floating_point(w) else torch.float16

        # Set up the linear translator configs
        translator = torch.nn.Linear(
            config.d_model, config.d_model, bias=config.bias,
            dtype=config.dtype)
        translator.weight.data.zero_()
        translator.bias.data.zero_()

        # Don't include the final layer since it does not need a translator
        self.layer_translators = torch.nn.ModuleList(
            [deepcopy(translator) for _ in
             range(self.config.num_hidden_layers)]
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
    def from_clip_model(cls, clip_model: CLIPModel, bias: bool = True) \
            -> "CLIPTunedLens":
        """Create a lens from a CLIPModel instance.

        Args:
            clip_model: CLIPModel instance (from ingredients.py)
        """
        # get loaded model and config
        actual_model = clip_model.get_model()
        config_dict = clip_model.get_config()

        # create TunedLensConfig
        config = TunedLensConfig(
            base_model_name_or_path=config_dict['base_model_name_or_path'],
            d_model=config_dict['d_model'],
            num_hidden_layers=config_dict['num_hidden_layers'],
            dtype=config_dict.get('dtype', torch.float32),
            bias=bias
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

    # TODO: implement this
    @classmethod
    def from_pretrained_model(cls, path: Union[str, Path]) -> None:
        """Load the lens from a model directory"""
        pass

    # TODO: implement this
    @classmethod
    def from_lens_path(cls, path: Union[str, Path]) -> None:
        """Load the lens from a model and unembed directory"""
        pass

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

    def save(self, path: Union[str, Path]) -> None:
        """Save the lens to a directory"""
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        state_dict = self.layer_translators.state_dict()

        # save the parameters
        torch.save(state_dict, path / "params.pt")
        # save the config
        with open(path / "config.json", "w") as f:
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

    for block in model.visual.transformer.resblocks:  # type: ignore
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
