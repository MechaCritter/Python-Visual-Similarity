"""
Structural type describing the embedding-store interface used by retrieval.

This module defines :class:`EmbeddingStore`, a :class:`typing.Protocol` capturing
the surface that retrieval and evaluation rely on: the gallery embeddings and
their paths, the embedder that produced them, and an accelerated
nearest-neighbour search. :class:`pyvisim.image_store.InMemoryImageEmbeddingStore`
satisfies it structurally, so the evaluation helpers stay decoupled from the
concrete store implementation.
"""

from typing import Protocol, runtime_checkable

from .embedders import Embedder
from .numeric import Float32NumpyArray, FloatNumpyArray, IntNumpyArray


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

    def search(
        self,
        query_vectors: FloatNumpyArray,
        k: int,
    ) -> tuple[Float32NumpyArray, IntNumpyArray]:
        """
        Return the ``k`` nearest gallery vectors for each query vector.

        :param query_vectors: A ``(D,)`` vector or a ``(N, D)`` batch of query
            vectors.
        :param k: Number of nearest neighbours to return per query.
        :return: A ``(scores, ids)`` tuple of ``(N, k)`` arrays. ``ids`` index
            into :attr:`paths`; missing neighbours are reported as ``-1``.
        """
        ...
