# borrow from transcoder_circuits/circuit_analysis.py:
# https://github.com/Prisma-Multimodal/ViT-Prisma.git

import enum
from dataclasses import dataclass, field, fields
from typing import List, Optional

import torch
from vit_prisma.sae import SparseAutoencoder


class ComponentType(enum.Enum):
    MLP = 'mlp'
    ATTN = 'attn'
    EMBED = 'embed'
    # error terms
    TC_ERROR = 'tc_error'  # error due to inaccurate transcoders
    PRUNE_ERROR = (
        'prune_error'  # error due to only looking at top paths in graph
    )
    BIAS_ERROR = 'bias_error'  # account for bias terms in transcoders


class FeatureType(enum.Enum):
    NONE = 'none'
    SAE = 'sae'
    TRANSCODER = 'tc'


class ContribType(enum.Enum):
    RAW = 'raw'
    ZERO_ABLATION = 'zero_ablation'


# an individual component (e.g. an attn head or a transcoder feature)
@dataclass
class Component:
    layer: int
    component_type: ComponentType

    token: Optional[int] = None

    attn_head: Optional[int] = None

    feature_type: Optional[FeatureType] = None
    feature_idx: Optional[int] = None

    def __str__(self, show_token=True):
        retstr = ''
        feature_type_str = ''

        base_str = f'{self.component_type.value}{self.layer}'
        attn_str = (
            ''
            if self.component_type != ComponentType.ATTN
            else f'[{self.attn_head}]'
        )

        feature_str = ''
        if self.feature_type is not None and self.feature_idx is not None:
            feature_str = f'{self.feature_type.value}[{self.feature_idx}]'

        token_str = ''
        if self.token is not None and show_token:
            token_str = f'@{self.token}'

        retstr = ''.join([base_str, attn_str, feature_str, token_str])
        return retstr

    def __repr__(self):
        return f'<Component object {self!s}>'


class FilterType(enum.Enum):
    EQ = enum.auto()  # equals
    NE = enum.auto()  # not equal to
    GT = enum.auto()  # greater than
    GE = enum.auto()  # greater than or equal to
    LT = enum.auto()  # less than
    LE = enum.auto()  # less than or equal to


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
        if self.token is None and len(self.component_path) > 0:
            self.token = self.component_path[-1].token
        if self.layer is None and len(self.component_path) > 0:
            self.layer = self.component_path[-1].layer

    # note: str(FeatureVector) should return a string that uniquely identifies a feature direction (e.g. for use in a causal graph)
    # (this is distinct from a unique feature *vector*, by the way)
    def __str__(self, show_full=True, show_contrib=True, show_last_token=True):
        retstr = ''
        token_str = (
            ''
            if self.token is None or not show_last_token
            else f'@{self.token}'
        )
        if len(self.component_path) > 0:
            if show_full:
                retstr = ''.join(
                    x.__str__(show_token=False)
                    for x in self.component_path[:-1]
                )
            retstr = ''.join(
                [
                    retstr,
                    self.component_path[-1].__str__(show_token=False),
                    token_str,
                ]
            )
        else:
            retstr = f'*{self.sublayer}{self.layer}{token_str}'
        if show_contrib and self.contrib is not None:
            retstr = ''.join([retstr, f': {self.contrib:.2}'])
        return retstr

    def __repr__(self):
        contrib_type_str = (
            ''
            if self.contrib_type is None
            else f' contrib_type={self.contrib_type.value}'
        )
        return f'<FeatureVector object {self!s}, sublayer={self.sublayer}{contrib_type_str}>'


@dataclass
class FeatureFilter:
    # feature-level filters
    layer: Optional[int] = field(
        default=None, metadata={'filter_level': 'feature'}
    )
    layer_filter_type: FilterType = FilterType.EQ
    sublayer: Optional[int] = field(
        default=None, metadata={'filter_level': 'feature'}
    )
    sublayer_filter_type: FilterType = FilterType.EQ
    token: Optional[int] = field(
        default=None, metadata={'filter_level': 'feature'}
    )
    token_filter_type: FilterType = FilterType.EQ

    # filters on last component in component_path
    component_type: Optional[ComponentType] = field(
        default=None, metadata={'filter_level': 'component'}
    )
    component_type_filter_type: FilterType = FilterType.EQ
    attn_head: Optional[int] = field(
        default=None, metadata={'filter_level': 'component'}
    )
    attn_head_filter_type: FilterType = FilterType.EQ
    feature_type: Optional[FeatureType] = field(
        default=None, metadata={'filter_level': 'component'}
    )
    feature_type_filter_type: FilterType = FilterType.EQ
    feature_idx: Optional[int] = field(
        default=None, metadata={'filter_level': 'component'}
    )
    feature_idx_filter_type: FilterType = FilterType.EQ

    def match(self, feature):
        component = None

        for field in fields(self):
            name = field.name
            val = self.__dict__[name]
            if val is None:
                continue

            try:
                filter_level = field.metadata['filter_level']
            except KeyError:
                continue  # not a filter
            if filter_level == 'feature':
                if val is not None:
                    filter_type = self.__dict__[f'{name}_filter_type']
                    if (
                        filter_type == FilterType.EQ
                        and val != feature.__dict__[name]
                    ):
                        return False
                    if (
                        filter_type == FilterType.NE
                        and val == feature.__dict__[name]
                    ):
                        return False
                    if (
                        filter_type == FilterType.GT
                        and feature.__dict__[name] <= val
                    ):
                        return False
                    if (
                        filter_type == FilterType.GE
                        and feature.__dict__[name] < val
                    ):
                        return False
                    if (
                        filter_type == FilterType.LT
                        and feature.__dict__[name] >= val
                    ):
                        return False
                    if (
                        filter_type == FilterType.LE
                        and feature.__dict__[name] > val
                    ):
                        return False
            elif filter_level == 'component':
                if component is None:
                    if len(feature.component_path) <= 0:
                        return False
                    component = feature.component_path[-1]
                if val is not None:
                    filter_type = self.__dict__[f'{name}_filter_type']
                    if (
                        filter_type == FilterType.EQ
                        and val != component.__dict__[name]
                    ):
                        return False
                    if (
                        filter_type == FilterType.NE
                        and val == component.__dict__[name]
                    ):
                        return False
        return True


@torch.no_grad()
def make_transcoder_feature_vector(
    transcoder: SparseAutoencoder,
    feature_idx: int,
    use_encoder: bool = True,
    token: int = 0,
) -> FeatureVector:
    """Build a feature vector for a given transcoder feature index.

    Args:
        transcoder: the transcoder model of a layer
        feature_idx: the index of the feature
        use_encoder: output encoder or decoder feature
        token: the token index. Since we are building for image input, we pay
        attention to the CLS token, so we set token to 0.
    """

    hook_point = (
        transcoder.cfg.hook_point
        if (use_encoder or not transcoder.cfg.is_transcoder)
        else transcoder.cfg.out_hook_point
    )
    layer = (
        transcoder.cfg.hook_point_layer
        if (use_encoder or not transcoder.cfg.is_transcoder)
        else transcoder.cfg.out_hook_point_layer
    )
    feature_type = (
        FeatureType.SAE
        if not transcoder.cfg.is_transcoder
        else FeatureType.TRANSCODER
    )
    vector = (
        transcoder.W_enc[:, feature_idx]
        if use_encoder
        else transcoder.W_dec[feature_idx]
    )
    vector = torch.clone(vector.detach())
    vector.requires_grad = False
    vector.requires_grad_(False)
    if 'resid_mid' in hook_point or (
        'normalized' in hook_point and 'ln2' in hook_point
    ):
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
        component_path=[
            Component(
                layer=layer,
                component_type=component_type,
                token=token,
                feature_type=feature_type,
                feature_idx=feature_idx,
            )
        ],
        layer=layer,
        sublayer=sublayer,
        vector=vector,
    )

    return my_feature
