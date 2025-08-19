"""Tools for taking components from the CLIP model"""

import logging

import open_clip
import torch

logger = logging.getLogger(__name__)

Model = open_clip.model.CLIP


def get_final_norm(model: Model):
    """Get the final norm layer from the CLIP model"""

    if not isinstance(model, Model):
        raise ValueError('Model is not a open_clip.CLIP model')

    assert hasattr(model, 'visual'), 'Model does not have a visual module'

    if hasattr(model.visual, 'ln_post'):
        return model.visual.ln_post
    elif hasattr(model.visual, 'ln_final'):
        # match the hookedvit from prisma
        return model.visual.ln_final
    else:
        raise AttributeError('Final norm layer not found in CLIP model')


def get_unembed_matrix(model: Model) -> torch.nn.Linear:
    """Get the unembed matrix from the model"""

    if not isinstance(model, Model):
        raise ValueError('Model is not a open_clip.CLIP model')
    # the unembed matrix is the transpose of the token embedding matrix
    if hasattr(model, 'token_embedding') and model.token_embedding is not None:
        unembed_matrix = model.token_embedding.weight.T  # shape: (512, 49408)
        # pytorch linear layer expects weight shape (out_features, in_features)
        unembed_layer = torch.nn.Linear(
            unembed_matrix.T.shape[0], unembed_matrix.T.shape[1], bias=False
        )
        unembed_layer.weight.data = unembed_matrix.T.clone().detach()
        return unembed_layer
    else:
        raise ValueError('Token embedding matrix not found in CLIP model')


def get_projection_matrix(model: Model) -> torch.nn.Linear:
    """Get the projection matrix from model, and wrap it in a linear layer"""

    assert isinstance(model, Model), 'Model is not a open_clip.CLIP model'
    assert hasattr(model, 'visual'), 'Model does not have a visual module'
    visual_module = model.visual

    if hasattr(visual_module, 'proj') and visual_module.proj is not None:
        # add a linear layer to wrap the projection matrix
        d_model = visual_module.proj.T.shape[0]
        d_out = visual_module.proj.T.shape[1]
        proj_layer = torch.nn.Linear(d_model, d_out, bias=False)
        # PyTorch Linear layer expects weight shape (out_features, in_features)
        # so we need to transpose the projection matrix
        proj_layer.weight.data = visual_module.proj.T.clone().detach()
        return proj_layer

    # match the hookedvit from prisma
    elif hasattr(visual_module, 'W_H') and visual_module.W_H is not None:
        # in prisma, they convert the projection matrix to a linear layer:
        # taken from: src/vit_prisma/models/weight_conversion.py
        # convert_open_clip_weights():
        # new_vision_model_state_dict["head.W_H"] = old_state_dict["visual.proj"]
        # new_vision_model_state_dict["head.b_H"] = torch.zeros((cfg.n_classes,))
        # since b_H is all zeros, we can just use the W_H matrix as the
        # projection matrix, which matches the original CLIP model.

        weight_tensor = visual_module.head.W_H

        d_model = weight_tensor.shape[0]
        d_out = weight_tensor.shape[1]
        proj_layer = torch.nn.Linear(d_model, d_out, bias=False)
        proj_layer.weight.data = weight_tensor.T.clone().detach()
        return proj_layer
    else:
        raise ValueError('Projection matrix not found in CLIP model')
