"""
Image retrieval machine over an accelerated index.

This module defines :class:`ImageRetriever`, a thin façade that pairs an
:class:`~pyvisim.retrieval.ImageIndex` with the gallery and encoder it was built
from. It exposes a single :meth:`ImageRetriever.retrieve_top_k_similar` method
that ranks the gallery against one or more query images.
"""

from __future__ import annotations

from ..functional import Candidate, retrieve_top_k_similar
from ..typing import Encoder, ImageInput
from .index import ImageIndex


class ImageRetriever:
    """
    Retrieve the most similar gallery images for a set of query images.

    :param index: The accelerated index to search against.
    :param encoder: Encoder used to turn the query images into feature vectors.
        It should match the one used to build the gallery encodings.
    """

    def __init__(self, index: ImageIndex, encoder: Encoder) -> None:
        self._index = index
        self._encoder = encoder

    @property
    def index(self) -> ImageIndex:
        """The underlying :class:`~pyvisim.retrieval.ImageIndex`."""
        return self._index

    @property
    def encoder(self) -> Encoder:
        """The encoder used to turn query images into feature vectors."""
        return self._encoder

    def retrieve_top_k_similar(
        self,
        query_images: ImageInput,
        k: int = 5,
    ) -> list[list[Candidate]]:
        """
        Return the top-k most similar gallery images for each query image.

        :param query_images: A single image or a batch/iterable of images to use
            as queries. Anything accepted by the encoder is valid.
        :param k: Number of top similar gallery images to return per query.
        :return: One ranked list of :class:`~pyvisim.functional.Candidate`
            matches per query image, in the same order as ``query_images``.
        """
        return retrieve_top_k_similar(
            query_images,
            self._index.encoding_map,
            self._encoder,
            k=k,
            index=self._index,
        )
