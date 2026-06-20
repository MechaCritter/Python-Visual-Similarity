"""
Functional building blocks shared across pyvisim's retrieval pipeline.

This module hosts :func:`retrieve_top_k_similar`, the single entry point used to
rank a gallery against one or more query images, and the :class:`Candidate` it
returns. The search is delegated to the accelerated index held by the
:class:`~pyvisim.typing.EmbeddingStore` it is given.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .typing import EmbeddingStore, FloatNumpyArray, ImageInput, IntNumpyArray

__all__ = ["Candidate", "retrieve_top_k_similar"]


class Candidate(NamedTuple):
    """A single retrieval result.

    :param path: Path of the matched gallery image.
    :param score: Similarity (or distance) of the match to the query. Higher
        means more similar for the inner-product metric; for an L2 index it is a
        distance, where lower means more similar.
    """

    path: str
    score: float


def retrieve_top_k_similar(
    query_images: ImageInput,
    store: EmbeddingStore,
    k: int = 5,
) -> list[list[Candidate]]:
    """
    Return the top-k most similar gallery images for each query image.

    Each query image is encoded with the store's encoder and matched against the
    gallery through the store's accelerated index.

    :param query_images: A single image or a batch/iterable of images to use as
        queries. Anything accepted by the store's encoder is valid.
    :param store: An :class:`~pyvisim.image_store.InMemoryImageEmbeddingStore`
        (or any :class:`~pyvisim.typing.EmbeddingStore`) holding the gallery.
    :param k: Number of top similar gallery images to return per query.
    :return: One ranked list of :class:`Candidate` matches per query image, in
        the same order as ``query_images``.
    """
    # ``encoder.encode`` returns one row per query image, in input order, so the
    # whole batch is searched at once: FAISS is far faster on one ``(M, D)``
    # matrix than on a per-query loop.
    query_matrix = np.asarray(store.encoder.encode(query_images))
    if query_matrix.ndim == 1:
        query_matrix = query_matrix.reshape(1, -1)
    if query_matrix.shape[0] == 0:
        return []

    scores, ids = store.search(query_matrix, k)
    return _assemble_results(store.paths, scores, ids)


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
