"""
This module contains functions to evaluate the performance of a retrieval system.
"""

from collections import Counter
from collections.abc import Iterable

import numpy as np

from .distance import cosine_similarity
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
    :param store: An :class:`~pyvisim.retrieval.image_store.InMemoryImageEmbeddingStore`
        (or any :class:`~pyvisim.typing.EmbeddingStore`) holding the gallery
        embeddings and the embedder.
    :param path_labels_dict: dict {img_path: label}, covering every path of
        the store.
    :param k: Number of top results to consider. Each average precision is
        divided by the number of gallery images sharing the query label,
        capped at ``k``.
    :return: mAP
    """
    all_vectors = np.asarray(store.embeddings)
    all_paths = store.paths
    embedder = store.embedder
    gallery_label_counts = Counter(path_labels_dict[path] for path in all_paths)

    APs = []
    for query_img, true_label in zip(images, image_labels, strict=True):
        query_vec = embedder.embed(query_img)
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        sims = cosine_similarity(query_vec, all_vectors)[0]

        # Sort by descending similarity
        sorted_idx = np.argsort(-sims)
        if k is not None:
            sorted_idx = sorted_idx[:k]

        sorted_paths = [all_paths[i] for i in sorted_idx]

        # compute average precision by counting relevant images at each rank
        relevant_count = 0
        precision_sum = 0.0
        for rank, path in enumerate(sorted_paths, start=1):
            if path_labels_dict[path] == true_label:
                relevant_count += 1
                precision_sum += relevant_count / rank

        # With R relevant images in the gallery, the ranking can hold at most
        # min(R, k) of them, so that is what the precision sum is divided by.
        n_relevant = gallery_label_counts[true_label]
        if k is not None:
            n_relevant = min(n_relevant, k)
        AP = precision_sum / n_relevant if n_relevant > 0 else 0.0

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
    :param store: An :class:`~pyvisim.retrieval.image_store.InMemoryImageEmbeddingStore`
        (or any :class:`~pyvisim.typing.EmbeddingStore`) holding the gallery
        embeddings and the embedder.
    :param path_labels_dict: dict {path: label}.
    :param k: Number of top results to check for a correct match.
    :return: Top-k accuracy (float) in the range [0, 1].
    """
    all_paths = store.paths
    all_vectors = np.asarray(store.embeddings)
    embedder = store.embedder
    correct_count = 0
    num_images = 0

    for query_img, true_label in zip(images, image_labels, strict=True):
        num_images += 1
        q_vec = embedder.embed(query_img)
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
