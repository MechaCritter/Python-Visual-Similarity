"""
This module contains functions to evaluate the performance of a retrieval system.
"""

from collections.abc import Iterable

import numpy as np

from ._utils import cosine_similarity
from .typing import EmbeddingStore, MatLike

__all__ = ["top_k_map", "top_k_accuracy"]


def top_k_map(
    images: Iterable[MatLike],
    image_labels: Iterable[int],
    store: EmbeddingStore,
    path_labels_dict: dict[str, int],
    k: int | None = None,
) -> float:
    """
    Computes mean Average Precision over the queries,
    based on whether retrieved images have matching labels.

    :param images: Query images.
    :param image_labels: Corresponding labels for the query images.
    :param store: An :class:`~pyvisim.image_store.InMemoryImageEmbeddingStore`
        (or any :class:`~pyvisim.typing.EmbeddingStore`) holding the gallery
        embeddings and the encoder.
    :param path_labels_dict: dict {img_path: label}
    :param k: Number of top results to consider.
    :return: mAP
    """
    all_vectors = np.asarray(store.embeddings)
    all_paths = store.paths
    encoder = store.encoder

    APs = []
    for query_img, true_label in zip(images, image_labels, strict=True):
        query_vec = encoder.encode(query_img)
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        sims = cosine_similarity(query_vec, all_vectors)[0]

        # Sort by descending similarity
        sorted_idx = np.argsort(-sims)
        if k is not None:
            sorted_idx = sorted_idx[:k]

        sorted_paths = [all_paths[i] for i in sorted_idx]
        sorted_labels = [path_labels_dict[path] for path in sorted_paths]

        # compute average precision by counting relevant images at each rank
        relevant_count = 0
        precision_sum = 0.0
        for rank, path in enumerate(sorted_paths, start=1):
            if path_labels_dict[path] == true_label:
                relevant_count += 1
                precision_sum += relevant_count / rank

        # If there are R relevant images in the entire dataset
        # average precision = sum(precision_at_i for each relevant i) / R
        R = sum(lbl == true_label for lbl in sorted_labels)
        AP = precision_sum / R if R > 0 else 0.0

        APs.append(AP)

    return float(np.mean(APs))


def top_k_accuracy(
    images: Iterable[MatLike],
    image_labels: Iterable[int],
    store: EmbeddingStore,
    path_labels_dict: dict[str, int],
    k: int,
) -> float:
    """
    Computes top-k accuracy. For each query, we look at the top-k
    most similar results in the dataset. If any of them match the
    query's label, that query is considered correct.

    :param images: Query images.
    :param image_labels: List of true labels for each query image.
    :param store: An :class:`~pyvisim.image_store.InMemoryImageEmbeddingStore`
        (or any :class:`~pyvisim.typing.EmbeddingStore`) holding the gallery
        embeddings and the encoder.
    :param path_labels_dict: dict {path: label}.
    :param k: Number of top results to check for a correct match.
    :return: Top-k accuracy (float) in the range [0, 1].
    """
    all_paths = store.paths
    all_vectors = np.asarray(store.embeddings)
    encoder = store.encoder
    correct_count = 0
    num_images = 0

    for query_img, true_label in zip(images, image_labels, strict=True):
        num_images += 1
        q_vec = encoder.encode(query_img)
        if q_vec.ndim == 1:
            q_vec = q_vec.reshape(1, -1)

        sims = cosine_similarity(q_vec, all_vectors)[0]
        sorted_idx = np.argsort(-sims)[:k]  # top-k

        # Check if any of the top-k share the query's label
        found_match = False
        for idx in sorted_idx:
            if path_labels_dict[all_paths[idx]] == true_label:
                found_match = True
                break

        if found_match:
            correct_count += 1

    if num_images == 0:
        return 0.0
    return float(correct_count / num_images)
