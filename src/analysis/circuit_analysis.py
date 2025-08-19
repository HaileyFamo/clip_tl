import copy
import functools
from typing import Optional

import numpy as np
import open_clip
import torch
from transformer_lens.utils import get_act_name, to_numpy
from vit_prisma.sae import SparseAutoencoder

from src.analysis.components import (
    Component,
    ComponentType,
    ContribType,
    FeatureFilter,
    FeatureType,
    FeatureVector,
)


@torch.no_grad()
def get_attn_head_contribs(model, cache, layer_idx, range_normal):
    """
    Calculate the contribution of each attention head from each source token to
    each destination token, adapted for OpenCLIP ViT.

    Args:
        model: the visual model (e.g., model_v)
        cache: the cache of all the layers
        layer_idx: the current layer to analyze
        range_normal: the target direction vector (from a higher-level
        feature_vector)

    Returns:
        a tensor of shape [batch, num_heads, dst_pos, src_pos], representing
        the contribution scores.
    """
    # 1. from cache, get v_acts and pattern
    # v_acts.shape: [batch, seq_len, num_heads, d_head]
    v_acts = cache[get_act_name('v', layer_idx)]

    # pattern.shape: [batch, num_heads, dst_pos, src_pos]
    pattern = cache[get_act_name('pattern', layer_idx)]

    # 2. get W_O and reshape for head-wise multiplication
    if hasattr(model.visual, 'transformer') and hasattr(
        model.visual.transformer, 'resblocks'
    ):
        attn_block = model.visual.transformer.resblocks[layer_idx].attn
    else:
        attn_block = model.visual.blocks[layer_idx].attn
        print(attn_block)
    n_heads = (
        attn_block.num_heads
        if hasattr(attn_block, 'num_heads')
        else model.visual.n_heads
    )
    d_model = (
        attn_block.embed_dim
        if hasattr(attn_block, 'embed_dim')
        else model.visual.d_model
    )
    d_head = d_model // n_heads

    W_O = attn_block.out_proj.weight.reshape(
        n_heads,
        d_head,  # d_head_out
        d_model,  # d_model
    )

    # 3. calculate contribs
    # 'bshf,hfm,b hds,m->bhds'
    # b: batch, s: src_pos, h: head, f: d_head, m: d_model, d: dst_pos
    contribs = torch.einsum(
        'bshf,hfm,bhds,m->bhds', v_acts, W_O, pattern, range_normal
    )

    return contribs


@torch.no_grad()
def get_transcoder_ixg(
    transcoder: SparseAutoencoder,
    cache: dict[str, torch.Tensor],
    range_normal: torch.Tensor,
    input_layer: int,
    input_token_idx: int,
    return_numpy: bool = True,
    is_transcoder_post_ln: bool = True,
    return_feature_activs: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get the pulledback feature for a given transcoder."""

    pulledback_feature = transcoder.W_dec @ range_normal
    if is_transcoder_post_ln:
        act_name = get_act_name('normalized', input_layer, 'ln2')
    else:
        act_name = get_act_name('resid_mid', input_layer)

    feature_activs = transcoder.encode(cache[act_name])[1][0, input_token_idx]
    pulledback_feature = pulledback_feature * feature_activs
    if return_numpy:
        pulledback_feature = to_numpy(pulledback_feature)
        feature_activs = to_numpy(feature_activs)

    if not return_feature_activs:
        return pulledback_feature
    else:
        return pulledback_feature, feature_activs


# approximate layernorms as constants when propagating feature vectors backward
# for theoretical motivation, see the LayerNorm section of
# https://www.neelnanda.io/mechanistic-interpretability/attribution-patching
@torch.no_grad()
def get_ln_constant(
    model,
    cache: dict[str, torch.Tensor],
    vector: torch.Tensor,
    layer: int,
    token: int,
    is_ln2: bool = False,
    recip: bool = False,
):
    """Get the layernorm constant for a given layer and token."""
    x_act_name = (
        get_act_name('resid_mid', layer)
        if is_ln2
        else get_act_name('resid_pre', layer)
    )
    x = cache[x_act_name][0, token]

    y_act_name = get_act_name('normalized', layer, 'ln2' if is_ln2 else 'ln1')
    y = cache[y_act_name][0, token]

    if torch.dot(vector, x) == 0:
        return torch.tensor(0.0)
    return (
        torch.dot(vector, y) / torch.dot(vector, x)
        if not recip
        else torch.dot(vector, x) / torch.dot(vector, y)
    )


@torch.no_grad()
def get_top_transcoder_features(
    model,
    transcoder: SparseAutoencoder,
    cache: dict[str, torch.Tensor],
    feature_vector: FeatureVector,
    layer: int,
    k: int = 5,
):
    """Get the top k transcoder features for a given feature vector."""

    my_token = (
        feature_vector.token
        if feature_vector.token >= 0
        else cache[get_act_name('resid_pre', 0)].shape[1] + feature_vector.token
    )
    is_transcoder_post_ln = (
        'ln2' in transcoder.cfg.hook_point
        and 'normalized' in transcoder.cfg.hook_point
    )

    # compute error
    if is_transcoder_post_ln:
        act_name = get_act_name('normalized', layer, 'ln2')
    else:
        act_name = get_act_name('resid_mid', layer)
    transcoder_out = transcoder.encode(cache[act_name])[0][0, my_token]
    if hasattr(model.visual, 'blocks'):
        mlp_out = model.visual.blocks[layer].mlp(cache[act_name])[0, my_token]
    elif hasattr(model.visual, 'transformer') and hasattr(
        model.visual.transformer, 'resblocks'
    ):
        mlp_out = model.visual.transformer.resblocks[layer].mlp(
            cache[act_name]
        )[0, my_token]

    error = torch.dot(
        feature_vector.vector, mlp_out - transcoder_out
    ) / torch.dot(feature_vector.vector, mlp_out)

    # compute pulledback feature
    pulledback_feature, feature_activs = get_transcoder_ixg(
        transcoder,
        cache,
        feature_vector.vector,
        layer,
        feature_vector.token,
        return_numpy=False,
        is_transcoder_post_ln=is_transcoder_post_ln,
    )
    top_contribs, top_indices = torch.topk(pulledback_feature, k=k)

    top_contribs_list = []
    for contrib, index in zip(top_contribs, top_indices):
        vector = transcoder.W_enc[:, index]
        vector = vector * (transcoder.W_dec @ feature_vector.vector)[index]

        if is_transcoder_post_ln:
            vector = vector * get_ln_constant(
                model, cache, vector, layer, feature_vector.token, is_ln2=True
            )

        new_component = Component(
            layer=layer,
            component_type=ComponentType.MLP,
            token=my_token,
            feature_type=FeatureType.TRANSCODER,
            feature_idx=index.item(),
        )
        top_contribs_list.append(
            FeatureVector(
                component_path=[new_component],
                vector=vector,
                layer=layer,
                sublayer='resid_mid',
                contrib=contrib.item(),
                contrib_type=ContribType.RAW,
                error=error,
            )
        )
    return top_contribs_list


@torch.no_grad()
def get_top_contribs(
    model,
    transcoders,
    cache,
    feature_vector,
    k: int = 5,
    ignore_bos: bool = False,
    only_return_all_scores: bool = False,
    cap: Optional[float] = None,
    filter: Optional[FeatureFilter] = None,
):
    """
    Get top contributions for a feature vector.
    Adapted for a ViT model. It now uses get_attn_head_contribs for attention head calculations.
    Args:
        model: the CLIP model
        transcoders: the transcoders
        cache: the cache of all the layers
        feature_vector: the feature vector
        k: the number of top contributions to return
        ignore_bos: whether to ignore the BOS token
    """
    if feature_vector.sublayer == 'mlp_out':
        return get_top_transcoder_features(
            model,
            transcoders[feature_vector.layer],
            cache,
            feature_vector,
            feature_vector.layer,
            k=k,
        )

    my_layer = feature_vector.layer

    # get MLP contribs
    all_mlp_contribs = []
    # go to all the previous layers
    mlp_max_layer = my_layer + (
        1 if feature_vector.sublayer == 'resid_post' else 0
    )
    for cur_layer in range(mlp_max_layer):
        cur_top_features = get_top_transcoder_features(
            model, transcoders[cur_layer], cache, feature_vector, cur_layer, k=k
        )
        all_mlp_contribs = all_mlp_contribs + cur_top_features

    # get attn contribs
    all_attn_contribs = []
    attn_max_layer = my_layer + (
        1
        if feature_vector.sublayer == 'resid_post'
        or feature_vector.sublayer == 'resid_mid'
        else 0
    )
    for cur_layer in range(attn_max_layer):
        attn_contribs = get_attn_head_contribs(
            model, cache, cur_layer, feature_vector.vector
        )[0, :, feature_vector.token, :]
        # if ignore_bos:
        #     attn_contribs = attn_contribs[:, 1:]

        if attn_contribs.numel() == 0:
            print(f'No attn contribs for layer {cur_layer}')
            continue

        # here we get the top k attn contribs, they are scalars
        top_attn_contribs_flattened, top_attn_contrib_indices_flattened = (
            torch.topk(
                attn_contribs.flatten(), k=np.min([k, len(attn_contribs)])
            )
        )
        top_attn_contrib_indices = np.array(
            np.unravel_index(
                to_numpy(top_attn_contrib_indices_flattened),
                attn_contribs.shape,
            )
        ).T

        # here we get the top k attn contribs, they are vectors
        for contrib, (head, src_token) in zip(
            top_attn_contribs_flattened, top_attn_contrib_indices
        ):
            # adapted for ViT
            if hasattr(model.visual, 'transformer') and hasattr(
                model.visual.transformer, 'resblocks'
            ):
                attn_block = model.visual.transformer.resblocks[cur_layer].attn
            else:  # for prisma
                attn_block = model.visual.blocks[cur_layer].attn
            d_model = (
                attn_block.embed_dim
                if hasattr(attn_block, 'embed_dim')
                else attn_block.d_model
            )
            n_heads = (
                attn_block.num_heads
                if hasattr(attn_block, 'num_heads')
                else attn_block.n_heads
            )
            d_head = d_model // n_heads
            # if ignore_bos:
            #     src_token = src_token + 1

            # W_O_head is used to project the feature vector to the head space
            # W_O_head.shape: (d_head, d_model) = (64, 768)
            W_O_head = attn_block.out_proj.weight.reshape(
                n_heads, d_head, d_model
            )[head, :, :]

            # in openclip, W_Q, W_K, W_V is in in_proj_weight, we need to use .chunk(3)[2] to separate them.
            # W_V_head.shape = (d_model, d_head) = (768, 64)
            W_V_head = (
                attn_block.in_proj_weight.chunk(3)[2]
                .reshape(d_model, n_heads, d_head)
                .permute(1, 0, 2)[head, :, :]
            )

            OV = W_V_head @ W_O_head  # shape: (d_model, d_model) = (768, 768)

            # vector = model.OV[cur_layer, head] @ feature_vector.vector
            vector = OV @ feature_vector.vector
            attn_pattern = cache[get_act_name('pattern', cur_layer)]
            vector = (
                vector * attn_pattern[0, head, feature_vector.token, src_token]
            )
            ln_constant = get_ln_constant(
                model, cache, vector, cur_layer, src_token, is_ln2=False
            )
            vector = vector * ln_constant
            if ln_constant.isnan():
                print('Nan!')

            new_component = Component(
                layer=cur_layer,
                component_type=ComponentType.ATTN,
                token=src_token,
                attn_head=head,
            )
            new_feature_vector = FeatureVector(
                component_path=feature_vector.component_path + [new_component],
                vector=vector,
                layer=cur_layer,
                sublayer='resid_pre',
                contrib=contrib.item(),
                contrib_type=ContribType.RAW,
            )
            all_attn_contribs.append(new_feature_vector)

    # get embedding contribs
    my_token = (
        feature_vector.token
        if feature_vector.token >= 0
        else cache[get_act_name('resid_pre', 0)].shape[1] + feature_vector.token
    )
    # my_token = feature_vector.token
    embedding_contrib = FeatureVector(
        component_path=feature_vector.component_path
        + [
            Component(
                layer=0,
                component_type=ComponentType.EMBED,
                token=my_token,
            )
        ],
        vector=feature_vector.vector,
        layer=0,
        sublayer='resid_pre',
        contrib=torch.dot(
            cache[get_act_name('resid_pre', 0)][0, feature_vector.token],
            feature_vector.vector,
        ).item(),
        contrib_type=ContribType.RAW,
    )

    # get top contribs from all categories
    all_contribs = all_mlp_contribs + all_attn_contribs + [embedding_contrib]

    if filter is not None:
        all_contribs = [x for x in all_contribs if filter.match(x)]

    if cap is not None:
        for i, contrib in enumerate(all_contribs):
            if contrib.contrib > cap:
                all_contribs[i].contrib = cap
                all_contribs[i].contrib_type = ContribType.ZERO_ABLATION
    all_contrib_scores = torch.tensor([x.contrib for x in all_contribs])
    if only_return_all_scores:
        return all_contrib_scores

    _, top_contrib_indices = torch.topk(
        all_contrib_scores, k=np.min([k, len(all_contrib_scores)])
    )
    return [all_contribs[i.item()] for i in top_contrib_indices]


@torch.no_grad()
def greedy_get_top_paths(
    model: open_clip.model.CLIP,
    transcoders: list[SparseAutoencoder],
    cache: dict[str, torch.Tensor],
    feature_vector: FeatureVector,
    num_iters: int = 2,
    num_branches: int = 5,
    ignore_bos: bool = True,
    do_raw_attribution: bool = False,
    filter: Optional[FeatureFilter] = None,
):
    do_cap = not do_raw_attribution  # historical name change; TODO: refactor

    all_paths = []
    new_root = copy.deepcopy(feature_vector)

    # deal with LN constant
    # TODO: this is hacky and makes the assumption that if feature_vector is a
    # transcoder feature, then it comes from the passed list of transcoders
    if new_root.component_path[-1].feature_type == FeatureType.TRANSCODER:
        tc = transcoders[new_root.layer]
        if 'ln2.hook_normalized' in tc.cfg.hook_point:
            ln_constant = get_ln_constant(
                model,
                cache,
                new_root.vector,
                new_root.layer,
                new_root.token,
                is_ln2=True,
            )
            new_root.vector *= ln_constant
        new_root.contrib = tc.encode(cache[tc.cfg.hook_point])[1][
            0, new_root.token, new_root.component_path[-1].feature_idx
        ].item()
    cur_paths = [[new_root]]
    for iter in range(num_iters):
        new_paths = []
        for path in cur_paths:
            cur_feature = path[-1]
            if cur_feature.layer == 0 and cur_feature.sublayer == 'resid_pre':
                continue

            cap = None
            if do_cap:
                # Cap feature contribs at smallest transcoder feature activation
                # This corresponds to calculating feature attribs by
                #   zero-ablating the output of the feature
                for cap_feature in path:
                    if len(cap_feature.component_path) > 0 and (
                        cap_feature.component_path[-1].feature_type
                        == FeatureType.TRANSCODER
                        or (
                            cap_feature.component_path[-1].feature_type
                            == FeatureType.SAE
                            and (cap is None or cap_feature.contrib < cap)
                        )
                    ):
                        cap = cap_feature.contrib

            cur_top_contribs = get_top_contribs(
                model,
                transcoders,
                cache,
                cur_feature,
                k=num_branches,
                ignore_bos=ignore_bos,
                cap=cap,
                filter=filter,
            )
            new_paths = new_paths + [
                path + [cur_top_contrib] for cur_top_contrib in cur_top_contribs
            ]
        _, top_new_path_indices = torch.topk(
            torch.tensor([new_path[-1].contrib for new_path in new_paths]),
            k=np.min([num_branches, len(new_paths)]),
        )
        cur_paths = [new_paths[i] for i in top_new_path_indices]
        all_paths.append(cur_paths)
    return all_paths


def print_all_paths(paths: list[list[FeatureVector]]):
    """Print all paths, each path is a list of FeatureVector.

    Example:
        Path [1][0]: mlp10tc[46014]@0: 5.4 <- mlp9tc[37227]@0: 2.7
        <- embed0@0: 2.1
        In this example, each object in the path is a FeatureVector.
        Path [1][0]: is the first path in the second list (iteration 2).
        mlp10tc[46014]@0: this is a transcoder feature (in mlp layer 10).
        The 46014 is the feature index.
        The @0: is the token index. Since we use the CLS token, the token index
        will always be 0.
    """
    if len(paths) == 0:
        return
    if type(paths[0][0]) is list:
        for i, cur_paths in enumerate(paths):
            try:
                print(f'--- Paths of size {len(cur_paths[0])} ---')
            except:
                continue
            for j, cur_path in enumerate(cur_paths):
                print(f'Path [{i}][{j}]: ', end='')
                print(
                    ' <- '.join(
                        map(
                            lambda x: x.__str__(
                                show_full=False, show_last_token=True
                            ),
                            cur_path,
                        )
                    )
                )
    else:
        for j, cur_path in enumerate(paths):
            print(f'Path [{j}]: ', end='')
            print(
                ' <- '.join(
                    map(
                        lambda x: x.__str__(
                            show_full=False, show_last_token=True
                        ),
                        cur_path,
                    )
                )
            )


def flatten_nested_list(x):
    return list(functools.reduce(lambda a, b: a + b, x))


def get_paths_via_filter(
    all_paths: list[list[FeatureVector]],
    infix_path: Optional[list[FeatureFilter]] = None,
    not_infix_path: Optional[list[FeatureFilter]] = None,
    suffix_path: Optional[list[FeatureFilter]] = None,
) -> list[list[FeatureVector]]:
    retpaths = []
    if type(all_paths[0][0]) is list:
        path_list = flatten_nested_list(all_paths)
    else:
        path_list = all_paths
    for path in path_list:
        if not_infix_path is not None:
            if len(path) < len(not_infix_path):
                continue

            match_started = False
            path_good = True
            i = 0
            for j, cur_feature in enumerate(path):
                cur_infix_filter = not_infix_path[i]

                if cur_infix_filter.match(cur_feature):
                    if not match_started:
                        if len(path[j:]) < len(not_infix_path):
                            break
                        match_started = True
                elif match_started:
                    path_good = False
                    break

                if match_started:
                    i = i + 1
                    if i >= len(not_infix_path):
                        break
            if not (match_started and path_good):
                retpaths.append(path)

        if infix_path is not None:
            if len(path) < len(infix_path):
                continue

            match_started = False
            path_good = True
            i = 0
            for j, cur_feature in enumerate(path):
                cur_infix_filter = infix_path[i]

                if cur_infix_filter.match(cur_feature):
                    if not match_started:
                        if len(path[j:]) < len(infix_path):
                            break
                        match_started = True
                elif match_started:
                    path_good = False
                    break

                if match_started:
                    i = i + 1
                    if i >= len(infix_path):
                        break
            if match_started and path_good:
                retpaths.append(path)

        if suffix_path is not None:
            if len(path) < len(suffix_path):
                continue
            path_good = True
            for i in range(1, len(suffix_path) + 1):
                cur_feature = path[-i]
                cur_suffix_filter = suffix_path[-i]
                if not cur_suffix_filter.match(cur_feature):
                    path_good = False
                    break
            if path_good:
                retpaths.append(path)
    return retpaths
