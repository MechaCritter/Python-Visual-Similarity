"""
Image retrieval machine over an accelerated index.

This module defines :class:`ImageRetriever`, a thin façade that pairs an
:class:`~pyvisim.retrieval.ImageIndex` with the gallery and encoder it was built
from. It exposes a single :meth:`ImageRetriever.retrieve_top_k_similar` method
that ranks the gallery against one or more query images, delegating the actual
work to :func:`pyvisim.functional.retrieve_top_k_similar`.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..functional import TopKSimilarityMap, retrieve_top_k_similar
from .index import ImageIndex


class ImageRetriever:
    """
    Retrieve the most similar gallery images for a set of query images.

    :param index: The accelerated index to search against.
    """

    def __init__(self, index: ImageIndex) -> None:
        self._index = index

    @property
    def index(self) -> ImageIndex:
        """The underlying :class:`~pyvisim.retrieval.ImageIndex`."""
        return self._index

    def retrieve_top_k_similar(
        self,
        query_image_paths: str | Iterable[str],
        k: int = 5,
    ) -> TopKSimilarityMap:
        """
        Return the top-k most similar gallery images for each query image.

        :param query_image_paths: A single image path or an iterable of image
            paths to use as queries.
        :param k: Number of top similar gallery images to return per query.
        :return: A :class:`~pyvisim.functional.TopKSimilarityMap` keyed by query
            image path; each value is the ranked list of candidates.
        :raises FileNotFoundError: If a query image file is missing.
        :raises ValueError: If a query image cannot be decoded.
        """
        return retrieve_top_k_similar(
            query_image_paths,
            self._index.encoding_map,
            self._index.encoder,
            k=k,
            index=self._index,
        )
