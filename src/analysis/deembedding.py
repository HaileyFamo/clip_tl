from typing import Optional, Union

import torch

from src.analysis.components import FeatureVector
from src.tuned_lens.clip_tl import CLIPTunedLens

# print colors
YELLOW = "\033[93m"  # yellow
RESET = "\033[0m"  # reset color


def get_labels_for_feature_vector(
    text_embeddings: torch.Tensor,
    labels: list[str],
    feature_vector: FeatureVector,
    lens: Optional[CLIPTunedLens] = None,
    k: int = 5,
    print_top_k: bool = False,
) -> tuple[Union[str, list[str]], Union[float, list[float]]]:
    """Get the labels for a given feature vector.

    If lens is not provided, we will use the logit lens. That is, we will
    calculate the feature vector's similarity to the text embeddings directly.
    """

    with torch.no_grad():
        # pulledback_feature = model.W_E @ feature_vector.vector
        if lens is None:
            similarity = text_embeddings @ feature_vector.vector.T
        else:
            projected_decoder_vectors = lens.unembed.project_feature(
                feature_vector.vector
            )
            similarity = text_embeddings @ projected_decoder_vectors.T
        top_k_scores, top_k_indices = torch.topk(similarity, k=k)
        top_k_words = [labels[i] for i in top_k_indices]

    if print_top_k:
        for i in range(k):
            print(
                f"{i + 1}. label: {top_k_words[i]:<20} | "
                f"prob: {top_k_scores[i].item():.4f}"
            )
    if k == 1:
        return top_k_words[0], top_k_scores[0].item()

    return top_k_words, top_k_scores.tolist()


def get_deembeddings_for_path(
    text_embeddings: torch.Tensor,
    labels: list[str],
    path: list[FeatureVector],
    lens: Optional[CLIPTunedLens] = None,
    print_path: bool = False,
) -> str:
    """Get the deembeddings for a given path.

    If lens is not provided, we will use the logit lens. That is, we will
    calculate the feature vector's similarity to the text embeddings directly.
    """

    result_parts = []
    for feature in path:
        try:
            feature_str = feature.__str__()
        except TypeError:
            feature_str = str(feature)

        words, score = get_labels_for_feature_vector(
            text_embeddings, labels, feature, lens, k=1
        )
        # use yellow to highlight word
        if isinstance(words, str):
            highlighted_deembedding = f"{YELLOW}{words}{RESET}"
        else:
            highlighted_deembedding = f'{YELLOW}{", ".join(words)}{RESET}'
        result_parts.append(f"{feature_str}({highlighted_deembedding})")

    results = " ← ".join(result_parts)
    if print_path:
        print(results)

    return results


def get_deembeddings_for_all_paths(
    text_embeddings: torch.Tensor,
    labels: list[str],
    paths: list[list[FeatureVector]] = [],
    lens: Optional[CLIPTunedLens] = None,
) -> list[str]:
    """Get the deembeddings for all paths in a circuit analysis.

    If lens is not provided, we will use the logit lens. That is, we will
    calculate the feature vector's similarity to the text embeddings directly.
    """

    if paths == []:
        return []

    results = []
    for path in paths:
        result = get_deembeddings_for_path(text_embeddings, labels, path, lens)
        results.append(result)

    return results


def print_deembeddings_for_all_paths(
    text_embeddings: torch.Tensor,
    labels: list[str],
    paths: list[list[FeatureVector]] = [],
    lens: Optional[CLIPTunedLens] = None,
) -> None:
    """Print the deembeddings for all paths in a circuit analysis."""

    results = get_deembeddings_for_all_paths(
        text_embeddings, labels, paths, lens
    )
    if results:
        print(f"--- Paths of size {len(paths[0])} ---")
        for result in results:
            print(result)
