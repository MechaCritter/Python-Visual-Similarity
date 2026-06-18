"""
Functional building blocks shared across pyvisim's retrieval pipeline.

This module hosts :func:`retrieve_top_k_similar`, the single entry point used to
rank a gallery against one or more query images, and the :class:`Candidate` it
returns. The function works either by brute force (comparing every gallery
vector) or, when given a :class:`pyvisim.typing.SearchIndex`, by delegating the
nearest-neighbour search to that accelerated index.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ._utils import cosine_similarity
from .image_store import ImageEncodingMap
from .typing import Encoder, FloatNumpyArray, ImageInput, IntNumpyArray, SearchIndex

__all__ = ["Candidate", "retrieve_top_k_similar"]


class Candidate(NamedTuple):
    """A single retrieval result.

    :param path: Path of the matched gallery image.
    :param score: Similarity (or distance) of the match to the query. Higher
        means more similar for the brute-force and inner-product metrics; for an
        L2 index it is a distance, where lower means more similar.
    """

    path: str
    score: float


def retrieve_top_k_similar(
    query_images: ImageInput,
    dataset: ImageEncodingMap,
    encoder: Encoder,
    k: int = 5,
    *,
    index: SearchIndex | None = None,
) -> list[list[Candidate]]:
    """
    Return the top-k most similar gallery images for each query image.

    Each query image is encoded with ``encoder`` and matched against the
    ``dataset`` gallery. When ``index`` is ``None`` the search is done by brute
    force using cosine similarity; otherwise the nearest-neighbour search is
    delegated to ``index``, which accelerates the search significantly.

    :param query_images: A single image or a batch/iterable of images to use as
        queries. Anything accepted by ``encoder.encode`` is valid.
    :param dataset: An :class:`~pyvisim.image_store.ImageEncodingMap` mapping
        gallery image paths to their feature vectors.
    :param encoder: Encoder used to turn the query images into feature vectors.
    :param k: Number of top similar gallery images to return per query.
    :param index: Optional accelerated search index built over ``dataset``. When
        provided, its ids must align with ``dataset`` insertion order.
    :return: One ranked list of :class:`Candidate` matches per query image, in
        the same order as ``query_images``.
    """
    # ``encoder.encode`` returns one row per query image, in input order, so the
    # whole batch is searched at once: both the cosine matmul and FAISS are far
    # faster on one ``(M, D)`` matrix than on a per-query loop.
    query_matrix = np.asarray(encoder.encode(query_images))
    if query_matrix.ndim == 1:
        query_matrix = query_matrix.reshape(1, -1)
    if query_matrix.shape[0] == 0:
        return []

    if index is None:
        scores, ids = _brute_force_search(query_matrix, dataset, k)
        gallery_paths = list(dataset.keys())
    else:
        scores, ids = index.search(query_matrix, k)
        gallery_paths = index.paths

    return _assemble_results(gallery_paths, scores, ids)


def _brute_force_search(
    query_matrix: FloatNumpyArray,
    dataset: ImageEncodingMap,
    k: int,
) -> tuple[FloatNumpyArray, IntNumpyArray]:
    """
    Rank a gallery against a batch of query vectors using cosine similarity.

    :param query_matrix: Query feature vectors of shape ``(M, D)``.
    :param dataset: Gallery mapping of path to feature vector.
    :param k: Number of top matches to return per query.
    :return: A ``(scores, ids)`` tuple of ``(M, k)`` arrays, sorted by
        descending similarity, where ``ids`` index into ``dataset`` order.
    """
    gallery = np.asarray(list(dataset.values()))
    similarities = cosine_similarity(query_matrix, gallery)  # (M, N)
    top_ids = np.argsort(-similarities, axis=1)[:, :k]  # (M, k)
    top_scores = np.take_along_axis(similarities, top_ids, axis=1)
    return top_scores, top_ids


def _assemble_results(
    gallery_paths: list[str],
    scores: FloatNumpyArray,
    ids: IntNumpyArray,
) -> list[list[Candidate]]:
    """
    Turn batched ``(M, k)`` search results into per-query candidate lists.

    :param gallery_paths: Gallery image paths the ``ids`` index into.
    :param scores: ``(M, k)`` similarity (or distance) scores, best first.
    :param ids: ``(M, k)`` gallery ids; negative ids mark missing neighbours.
    :return: One ranked list of candidates per query row, in row order.
    """
    return [
        [
            Candidate(gallery_paths[int(image_id)], float(score))
            for score, image_id in zip(row_scores, row_ids, strict=True)
            if image_id >= 0
        ]
        for row_scores, row_ids in zip(scores, ids, strict=True)
    ]
