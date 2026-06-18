"""
Functional building blocks shared across pyvisim's retrieval API.

This module hosts :func:`retrieve_top_k_similar`, the single entry point used to
rank a gallery against one or more query images, and the read-only
:class:`TopKSimilarityMap` it returns. The function works either by brute force
(comparing every gallery vector) or, when given a :class:`pyvisim.typing.SearchIndex`,
by delegating the nearest-neighbour search to that accelerated index.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import NamedTuple

import numpy as np

from ._utils import cosine_similarity
from .image_store import ImageEncodingMap
from .typing import Encoder, FloatNumpyArray, IntNumpyArray, SearchIndex

__all__ = ["Candidate", "TopKSimilarityMap", "retrieve_top_k_similar"]


class Candidate(NamedTuple):
    """A single retrieval result.

    :param path: Path of the matched gallery image.
    :param score: Similarity (or distance) of the match to the query. Higher
        means more similar for the brute-force and inner-product metrics; for an
        L2 index it is a distance, where lower means more similar.
    """

    path: str
    score: float


class TopKSimilarityMap(Mapping[str, list[Candidate]]):
    """Read-only mapping from a query image path to its ranked candidates.

    Indexing the map with a query image path returns the list of
    :class:`Candidate` matches for that query, ordered best match first.

    :param results: Mapping of query path to its ranked list of candidates.
    """

    def __init__(self, results: dict[str, list[Candidate]]) -> None:
        self._results = results

    def __getitem__(self, query_path: str) -> list[Candidate]:
        return self._results[query_path]

    def __iter__(self) -> Iterator[str]:
        return iter(self._results)

    def __len__(self) -> int:
        return len(self._results)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_queries={len(self)})"


def retrieve_top_k_similar(
    query_image_paths: str | Iterable[str],
    dataset: ImageEncodingMap,
    encoder: Encoder,
    k: int = 5,
    *,
    index: SearchIndex | None = None,
) -> TopKSimilarityMap:
    """
    Return the top-k most similar gallery images for each query image.

    Each query image is encoded with ``encoder`` and matched against the
    ``dataset`` gallery. When ``index`` is ``None`` the search is done by brute
    force using cosine similarity; otherwise the nearest-neighbour search is
    delegated to ``index`` for acceleration.

    :param query_image_paths: A single image path or an iterable of image paths
        to use as queries.
    :param dataset: An :class:`~pyvisim.image_store.ImageEncodingMap` mapping
        gallery image paths to their feature vectors.
    :param encoder: Encoder used to turn each query image into a feature vector.
    :param k: Number of top similar gallery images to return per query.
    :param index: Optional accelerated search index built over ``dataset``. When
        provided, its ids must align with ``dataset`` insertion order.
    :return: A :class:`TopKSimilarityMap` keyed by query image path; each value
        is the ranked list of :class:`Candidate` matches.
    :raises FileNotFoundError: If a query image file is missing.
    :raises ValueError: If a query image cannot be decoded.
    """
    paths = (
        [query_image_paths]
        if isinstance(query_image_paths, str)
        else list(query_image_paths)
    )
    query_map = ImageEncodingMap(encoder, paths)
    query_paths = list(query_map.keys())
    if not query_paths:
        return TopKSimilarityMap({})

    # Stack the query vectors so the whole batch is searched at once: both the
    # cosine matmul and FAISS are far faster on one ``(M, D)`` matrix than on a
    # per-query loop.
    query_matrix = np.asarray(list(query_map.values()))

    if index is None:
        scores, ids = _brute_force_search(query_matrix, dataset, k)
        gallery_paths = list(dataset.keys())
    else:
        scores, ids = index.search(query_matrix, k)
        gallery_paths = index.paths

    return TopKSimilarityMap(_assemble_results(query_paths, gallery_paths, scores, ids))


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
    query_paths: list[str],
    gallery_paths: list[str],
    scores: FloatNumpyArray,
    ids: IntNumpyArray,
) -> dict[str, list[Candidate]]:
    """
    Turn batched ``(M, k)`` search results into a per-query candidate mapping.

    :param query_paths: Query image paths, one per row of ``scores``/``ids``.
    :param gallery_paths: Gallery image paths the ``ids`` index into.
    :param scores: ``(M, k)`` similarity (or distance) scores, best first.
    :param ids: ``(M, k)`` gallery ids; negative ids mark missing neighbours.
    :return: Mapping of query path to its ranked list of candidates.
    """
    results: dict[str, list[Candidate]] = {}
    for query_path, row_scores, row_ids in zip(query_paths, scores, ids, strict=True):
        results[query_path] = [
            Candidate(gallery_paths[int(image_id)], float(score))
            for score, image_id in zip(row_scores, row_ids, strict=True)
            if image_id >= 0
        ]
    return results
