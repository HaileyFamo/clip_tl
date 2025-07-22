"""Tools for taking components from the CLIP model"""

import torch
import open_clip
import logging


logger = logging.getLogger(__name__)

Model = open_clip.model.CLIP


def get_final_norm(model: Model):
    """Get the final norm layer from the CLIP model"""

    if not isinstance(model, Model):
        raise ValueError("Model is not a open_clip.CLIP model")
    return model.visual.ln_post


def get_unembed_matrix(model: Model) -> torch.nn.Linear:
    """Get the unembed matrix from the model"""

    if not isinstance(model, Model):
        raise ValueError("Model is not a open_clip.CLIP model")
    # the unembed matrix is the transpose of the token embedding matrix
    if hasattr(model, "token_embedding") and model.token_embedding is not None:
        unembed_matrix = model.token_embedding.weight.T  # shape: (512, 49408)
        # pytorch linear layer expects weight shape (out_features, in_features)
        unembed_layer = torch.nn.Linear(unembed_matrix.T.shape[0],
                                        unembed_matrix.T.shape[1], bias=False)
        unembed_layer.weight.data = unembed_matrix.T.clone().detach()
        return unembed_layer
    else:
        raise ValueError("Token embedding matrix not found in CLIP model")


def get_projection_matrix(model: Model) -> torch.nn.Linear:
    """Get the projection matrix from model, and wrap it in a linear layer"""

    if not isinstance(model, Model):
        raise ValueError("Model is not a open_clip.CLIP model")
    logger.info("Getting projection matrix from CLIP model, shape: %s",
                model.visual.proj.T.shape)  # type: ignore

    if (hasattr(model, "visual") and
            hasattr(model.visual, "proj") and
            model.visual.proj is not None):
        # add a linear layer to wrap the projection matrix
        d_model = model.visual.proj.T.shape[0]
        d_out = model.visual.proj.T.shape[1]
        proj_layer = torch.nn.Linear(d_model, d_out, bias=False)
        # PyTorch Linear layer expects weight shape (out_features, in_features)
        # so we need to transpose the projection matrix
        proj_layer.weight.data = model.visual.proj.T.clone().detach()
        return proj_layer
    else:
        raise ValueError("Projection matrix not found in CLIP model")
