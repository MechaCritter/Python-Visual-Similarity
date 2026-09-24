"""Structural type describing the embedding-store interface used by retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .embedders import Embedder
from .numeric import Float32NumpyArray, ImageInput

if TYPE_CHECKING:
    from ..image_store import Candidate


@runtime_checkable
class EmbeddingStore(Protocol):
    """
    Protocol for an in-memory gallery of image embeddings.

    An embedding store pairs a gallery of feature vectors (and their image
    paths) with the embedder that produced them and an accelerated index used to
    search the gallery.
    """

    @property
    def paths(self) -> list[str]:
        """Gallery image paths, ordered to match the embedding rows."""
        ...

    @property
    def embeddings(self) -> Float32NumpyArray:
        """The ``(N, D)`` gallery embedding matrix."""
        ...

    @property
    def embedder(self) -> Embedder:
        """The embedder that produced the gallery and embeds queries."""
        ...

    def retrieve_top_k_similar(
        self,
        query_images: ImageInput,
        k: int = 5,
    ) -> list[list[Candidate]]:
        """
        Return the top-k most similar gallery images for each query image.

        :param query_images: A single image or a batch/iterable of images to use
            as queries.
        :param k: Number of top similar gallery images to return per query.
        :return: One ranked list of :class:`~pyvisim.image_store.Candidate`
            matches per query image, in the same order as ``query_images``.
        """
        ...
