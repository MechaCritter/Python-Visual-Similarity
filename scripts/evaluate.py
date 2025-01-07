"""
This module contains functions to evaluate the performance of a retrieval system.
"""

import numpy as np

from src.utils import *


def retrieve_top_k_similar(
    uploaded_image: np.ndarray,
    dataset: dict[str, np.ndarray],
    encoder,
    k: int = 5
) -> list[tuple[str, float]]:
    """
    Returns the top-k most similar images from 'dataset' to the 'uploaded_image'.

    :param uploaded_image: Query image as a NumPy array (H x W x C).
    :param dataset: A dict mapping file paths to their feature vectors (np.ndarray).
    :param encoder: An object that implements `compute_vector(img) -> np.ndarray`.
    :param k: Number of top similar images to return.
    :return: A list of (image_path, similarity_score) for the top-k matches, sorted descending by similarity.
    """
    all_vectors, all_paths = np.array(list(dataset.values())), list(dataset.keys())

    # Query vector
    query_vector = encoder.encode(uploaded_image)

    # If `query_vector` is 1D, reshape to (1, D) to work with `cosine_similarity`
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)

    scores = cosine_similarity(query_vector, all_vectors)   # (1, N)
    scores = scores[0]

    sorted_indices = np.argsort(-scores)  # highest scores first

    # Slice top-k
    top_k_indices = sorted_indices[:k]

    results = [(all_paths[i], scores[i]) for i in top_k_indices]
    return results


def compute_map_top_k(queries: list[np.ndarray],
                query_labels: list[int],
                dataset: dict[str, np.ndarray],
                dataset_labels: dict[str, int],
                encoder,
                k: int=None) -> float:
    """
    Computes mean Average Precision over the queries,
    based on whether retrieved images have matching labels.

    :param queries: List of query images (NumPy arrays).
    :param query_labels: Corresponding labels for the queries.
    :param dataset: dict {img_path: feature_vector}
    :param dataset_labels: dict {img_path: label}
    :param encoder: Object with `compute_vector(img) -> np.ndarray`.
    :param k: Number of top results to consider.
    :return: mAP (float)
    """
    all_vectors, all_paths = np.array(list(dataset.values())), list(dataset.keys())

    APs = []
    for query_img, true_label in zip(queries, query_labels):
        query_vec = encoder.encode(query_img)
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
            if dataset_labels[path] == true_label:
                relevant_count += 1
                precision_sum += (relevant_count / rank)

        # If there are R relevant images in the entire dataset
        # average precision = sum(precision_at_i for each relevant i) / R
        R = sum(lbl == true_label for lbl in dataset_labels.values())
        if R > 0:
            AP = precision_sum / R
        else:
            AP = 0.0

        APs.append(AP)

    return float(np.mean(APs))

def top_k_accuracy(
    queries: list[np.ndarray],
    query_labels: list[int],
    dataset: dict[str, np.ndarray],
    dataset_labels: dict[str, int],
    encoder,
    k: int
) -> float:
    """
    Computes top-k accuracy. For each query, we look at the top-k
    most similar results in the dataset. If any of them match the
    query's label, that query is considered correct.

    :param queries: List of query images.
    :param query_labels: List of true labels for each query.
    :param dataset: dict {path: feature_vector}.
    :param dataset_labels: dict {path: label}.
    :param encoder: An object with `compute_vector(img) -> np.ndarray`.
    :param k: Number of top results to check for a correct match.
    :return: Top-k accuracy (float) in the range [0, 1].
    """
    all_paths, all_vectors = list(dataset.keys()), np.array(list(dataset.values()))
    correct_count = 0

    for query_img, true_label in zip(queries, query_labels):
        q_vec = encoder.encode(query_img)
        if q_vec.ndim == 1:
            q_vec = q_vec.reshape(1, -1)

        sims = cosine_similarity(q_vec, all_vectors)[0]
        sorted_idx = np.argsort(-sims)[:k]               # top-k

        # Check if any of the top-k share the query's label
        found_match = False
        for idx in sorted_idx:
            if dataset_labels[all_paths[idx]] == true_label:
                found_match = True
                break

        if found_match:
            correct_count += 1

    topk_acc = correct_count / len(queries)
    return float(topk_acc)
