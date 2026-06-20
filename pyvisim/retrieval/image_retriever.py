"""
Image retrieval façade over an embedding store.

This module defines :class:`ImageRetriever`, a thin wrapper around an
:class:`~pyvisim.typing.EmbeddingStore` (typically an
:class:`~pyvisim.image_store.InMemoryImageEmbeddingStore`). It exposes a single
:meth:`ImageRetriever.retrieve_top_k_similar` method that ranks the store's
gallery against one or more query images.
"""

from __future__ import annotations

from ..functional import Candidate, retrieve_top_k_similar
from ..typing import EmbeddingStore, Encoder, ImageInput


class ImageRetriever:
    """
    Retrieve the most similar gallery images for a set of query images.

    :param store: The embedding store to search against. It already bundles the
        gallery embeddings, their paths, the encoder and the accelerated index.
    """

    def __init__(self, store: EmbeddingStore) -> None:
        self._store = store

    @property
    def store(self) -> EmbeddingStore:
        """The underlying :class:`~pyvisim.typing.EmbeddingStore`."""
        return self._store

    @property
    def encoder(self) -> Encoder:
        """The encoder used to turn query images into feature vectors."""
        return self._store.encoder

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
        return retrieve_top_k_similar(query_images, self._store, k)
