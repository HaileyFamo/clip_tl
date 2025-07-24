import copy
import logging
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union

import open_clip
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose

from src.tuned_lens import model_surgery

logger = logging.getLogger(__name__)


OptmizerChoice = Literal["Adam", "AdamW", "SGD"]


class CLIPModel:
    """CLIP model configuration and manager.

    Args:
        device: The device to use for the model.
        model_name: The name of the model to load.
    """

    def __init__(
        self,
        model_name: str = (
            "open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K"
        ),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize CLIPModel with configuration parameters.

        Args:
            model_name: The name of the model to load.
            device: The device to use for the model.
        """

        # Configuration parameters (set at init)
        self.model_name = model_name
        self.device = device

        # Will be set after load()
        self.model: Optional[open_clip.model.CLIP] = None
        self.preprocess: Optional[Compose] = None
        self.config: Optional[dict] = None
        self.tokenizer: Optional[Any] = None

    def load(self) -> tuple[open_clip.model.CLIP, Compose]:
        """Load open-clip model and extract configuration.

        Returns:
            The model and preprocess function, also stores them internally.
        """

        if self.model is None:
            # load model
            self.model, _, self.preprocess = (
                open_clip.create_model_and_transforms(self.model_name)
            )

            self.model = self.model.to(self.device)
            self.model.eval()  # type: ignore
            self.model.requires_grad_(False)  # type: ignore

            # extract and store config
            self.config = self._extract_config()

        return self.model, self.preprocess

    def _extract_config(self) -> dict:
        """Extract and store config from the loaded model."""

        if self.model is None:
            raise ValueError("Model must be loaded first")

        # use model_surgery to extract config
        visual = self.model.visual

        # get hidden_size
        d_model = None
        if hasattr(visual, "ln_pre") and isinstance(
            visual.ln_pre, torch.nn.LayerNorm
        ):
            d_model = int(visual.ln_pre.normalized_shape[0])
        elif hasattr(visual, "conv1") and isinstance(
            visual.conv1, torch.nn.Conv2d
        ):
            d_model = int(visual.conv1.out_channels)

        if d_model is None:
            raise ValueError("Cannot determine d_model from model structure")

        # get number of layers
        num_hidden_layers = None
        if hasattr(visual, "transformer") and hasattr(
            visual.transformer, "resblocks"
        ):
            num_hidden_layers = len(visual.transformer.resblocks)

        if num_hidden_layers is None:
            raise ValueError(
                "Cannot determine number of layers from model structure"
            )

        return {
            "base_model_name_or_path": self.model_name,
            "d_model": d_model,
            "num_hidden_layers": num_hidden_layers,
            "dtype": torch.float32,
        }

    def get_config(self) -> dict:
        """Get model config."""
        if self.config is None:
            self.load()  # auto load
        assert self.config is not None, "Config not loaded."
        return self.config

    def get_model(self) -> model_surgery.Model:
        """Get the loaded model."""
        if self.model is None:
            self.load()  # auto load
        assert self.model is not None, "Model not loaded."
        return self.model

    def get_tokenizer(self) -> open_clip.tokenizer.SimpleTokenizer:
        """Get the tokenizer."""
        if self.tokenizer is None:
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
        assert self.tokenizer is not None, "Tokenzier not loaded."
        return self.tokenizer


class ImageData:
    """Image data configuration, default is ImageNet."""

    def __init__(
        self,
        data_path: str = "./data/images",
        batch_size: int = 64,
        image_size: int = 224,
        num_workers: int = 8,
        validation_split: float = 0.0,
    ):
        """Initialize ImageData configuration.

        Args:
            data_path: Path to image folder.
            batch_size: Batch size for dataloader.
            image_size: Size of input images.
            num_workers: Number of workers for dataloader.
            validation_split: Fraction of data to use for validation.
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.validation_split = validation_split

    def load(
        self, preprocess: Callable
    ) -> tuple[DataLoader, Optional[DataLoader]]:
        """Load image data and split into training and validation sets."""
        dataset = ImageFolder(self.data_path, transform=preprocess)

        if self.validation_split <= 0 or self.validation_split >= 1:
            train_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
            logger.info(
                "No validation set, using %d images for training", len(dataset)
            )
            return train_loader, None

        # Split dataset into training and validation
        num_total = len(dataset)
        num_val = int(self.validation_split * num_total)
        num_train = num_total - num_val

        # Ensure splits are not empty
        if num_train == 0 or num_val == 0:
            raise ValueError(
                f"Validation split of {self.validation_split} resulted in an "
                f"empty training or validation set."
            )

        train_dataset, val_dataset = random_split(
            dataset,
            [num_train, num_val],
            generator=torch.Generator().manual_seed(42),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # No need to shuffle validation data
            num_workers=self.num_workers,
        )
        return train_loader, val_loader


class Optimizer:
    """Optimizer configuration."""

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        weight_decay: float = 1e-3,
        momentum: float = 0.9,
        optimizer: OptmizerChoice = "Adam",
    ):
        """Initialize optimizer configuration.

        Args:
            lr: The learning rate.
            weight_decay: The weight decay coefficient.
            momentum: The momentum coefficient for SGD, or beta1 for Adam.
            optimizer: The optimizer type to use.
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.optimizer = optimizer

    def create_optim(
        self, params: list[torch.nn.Parameter]
    ) -> torch.optim.Optimizer:
        """Create the optimizer.

        Args:
            params: The parameters to optimize.

        Returns:
            A torch.optim.Optimizer instance.
        """

        if self.optimizer == "Adam":
            return torch.optim.Adam(
                params,
                lr=self.lr,
                betas=(self.beta1, self.beta2),
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "AdamW":
            return torch.optim.AdamW(
                params,
                lr=self.lr,
                betas=(self.beta1, self.beta2),
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "SGD":
            return torch.optim.SGD(
                params,
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer '{self.optimizer}'")


# do not use dataclass here, because we need to inherit from torch.nn.Module
# its __init__ method expects explicit arguments registering the parameters
# and usually we override __init__ method a lot for different models
class Unembed(torch.nn.Module):
    """Module that maps visual transformer hidden states to logits in the text
    space.

    This class is included in the tuned lens and frozen during training.

    Args:
        final_norm: The final normalization layer.
        projection: The projection matrix of CLIP wrapped in a nn.Linear.
        unembedding: The unembedding matrix.
    """

    final_norm: torch.nn.Module
    projection: torch.nn.Linear
    unembedding: torch.nn.Linear

    def __init__(self, model: model_surgery.Model):
        super().__init__()
        self.model = model
        final_norm = model_surgery.get_final_norm(model)
        projection = model_surgery.get_projection_matrix(model)
        unembedding = model_surgery.get_unembed_matrix(model)

        self.final_norm = copy.deepcopy(final_norm)  # type: ignore
        self.unembedding = copy.deepcopy(unembedding)  # type: ignore
        self.projection = copy.deepcopy(projection)  # type: ignore

        # In general we don't want to finetune the unembed operation.
        self.requires_grad_(False)

    # TODO: implement this
    @classmethod
    def from_pretrained(cls, path: Union[str, Path]) -> None:
        """Load the unembed from a directory"""
        pass

    def project_feature(self, feature_vector: torch.Tensor) -> torch.Tensor:
        """Project a feature's vector into the text space.

        Returns:
            The projected feature vector. Will not modify the input tensor.
        """
        feature_vector = self.final_norm(feature_vector.clone())
        feature_vector = self.projection(feature_vector)
        return feature_vector

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Convert hidden states into image embeddings (not vocabulary
        logits)."""

        if h.dim() == 3:  # [batch, seq_len, d_model]
            # take the [CLS] token
            h = h[:, 0, :]
        else:
            raise ValueError(f"Hidden state dimension != 3, got {h.dim()}")

        # final norm and projection from visual to text space
        h = self.final_norm(h)
        proj_h = self.projection(h)
        # Return the projected embeddings (512-dim), not the vocabulary logits
        return proj_h
