# build a feature vector first
from dataclasses import dataclass
from typing import List, Optional
import torch
from src.components import Component, ContribType, FeatureType, ComponentType


@dataclass
class FeatureVector:
    component_path: List[Component]
    vector: torch.Tensor
    layer: int
    sublayer: str
    token: Optional[int] = None
    contrib: Optional[float] = None
    contrib_type: Optional[ContribType] = None
    error: float = 0.0

    def __post_init__(self):
        if self.token is None and len(self.component_path)>0: self.token = self.component_path[-1].token
        if self.layer is None and len(self.component_path)>0: self.layer = self.component_path[-1].layer

    # note: str(FeatureVector) should return a string that uniquely identifies a feature direction (e.g. for use in a causal graph)
    # (this is distinct from a unique feature *vector*, by the way)
    def __str__(self, show_full=True, show_contrib=True, show_last_token=True):
        retstr = ''
        token_str = '' if self.token is None or not show_last_token else f'@{self.token}'
        if len(self.component_path) > 0:
            if show_full:
                retstr = ''.join(x.__str__(show_token=False) for x in self.component_path[:-1])
            retstr = ''.join([retstr, self.component_path[-1].__str__(show_token=False), token_str])
        else:
            retstr = f'*{self.sublayer}{self.layer}{token_str}'
        if show_contrib and self.contrib is not None:
            retstr = ''.join([retstr, f': {self.contrib:.2}'])
        return retstr

    def __repr__(self):
        contrib_type_str = '' if self.contrib_type is None else f' contrib_type={self.contrib_type.value}'
        return f'<FeatureVector object {str(self)}, sublayer={self.sublayer}{contrib_type_str}>'
    

@torch.no_grad()
def make_transcoder_feature_vector(sae, feature_idx, use_encoder=True, token=0) -> FeatureVector:
    """Build a feature vector for a given transcoder feature index.
    Args:
        sae: the transcoder model
        feature_idx: the index of the feature
        use_encoder: output encoder or decoder feature
        token: the token index. Since we are building for image input, we pay attentiont to the CLS token, so we set token to 0.
    Returns:
        a FeatureVector object
    """
    
    hook_point = sae.cfg.hook_point if (use_encoder or not sae.cfg.is_transcoder) else sae.cfg.out_hook_point
    layer = sae.cfg.hook_point_layer if (use_encoder or not sae.cfg.is_transcoder) else sae.cfg.out_hook_point_layer
    feature_type = FeatureType.SAE if not sae.cfg.is_transcoder else FeatureType.TRANSCODER
    vector = sae.W_enc[:,feature_idx] if use_encoder else sae.W_dec[feature_idx]
    vector = torch.clone(vector.detach())
    vector.requires_grad = False
    vector.requires_grad_(False)
    if 'resid_mid' in hook_point or ('normalized' in hook_point and 'ln2' in hook_point):
        # currently, we treat ln2normalized as resid_mid
        # this is kinda ugly, but because we account for layernorm constants in later
        #  functions, this does work now
        sublayer = 'resid_mid'
        component_type = ComponentType.MLP
    elif 'resid_pre' in hook_point:
        sublayer = 'resid_pre'
        component_type = ComponentType.ATTN
    elif 'mlp_out' in hook_point:
        sublayer = 'mlp_out'
        component_type = ComponentType.MLP
    elif 'resid_post' in hook_point:
        sublayer = 'resid_post'
        component_type = ComponentType.ATTN

    my_feature = FeatureVector(
        component_path=[Component(
            layer=layer,
            component_type=component_type,
            token=token,
            feature_type=feature_type,
            feature_idx=feature_idx
        )],
        
        layer = layer,
        sublayer = sublayer,
        vector = vector
    )

    return my_feature
