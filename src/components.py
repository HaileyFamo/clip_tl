# borrow from transcoder_circuits/circuit_analysis.py: 
# https://github.com/Prisma-Multimodal/ViT-Prisma.git

import enum
from dataclasses import dataclass, field
from typing import Optional, List
import copy

# define some classes

class ComponentType(enum.Enum):
    MLP = 'mlp'
    ATTN = 'attn'
    EMBED = 'embed'
    
    # error terms
    TC_ERROR = 'tc_error' # error due to inaccurate transcoders
    PRUNE_ERROR = 'prune_error' # error due to only looking at top paths in graph
    BIAS_ERROR = 'bias_error' # account for bias terms in transcoders

class FeatureType(enum.Enum):
    NONE = 'none'
    SAE = 'sae'
    TRANSCODER = 'tc'

class ContribType(enum.Enum):
    RAW = 'raw'
    ZERO_ABLATION = 'zero_ablation'


# Component: an individual component (e.g. an attn head or a transcoder feature)
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
        attn_str = '' if self.component_type != ComponentType.ATTN else f'[{self.attn_head}]'
        
        feature_str = ''
        if self.feature_type is not None and self.feature_idx is not None:
            feature_str = f"{self.feature_type.value}[{self.feature_idx}]"
            
        token_str = ''
        if self.token is not None and show_token:
            token_str = f'@{self.token}'

        retstr = ''.join([base_str, attn_str, feature_str, token_str])
        return retstr

    def __repr__(self):
        return f'<Component object {str(self)}>'
