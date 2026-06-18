"""
Structural type describing the search-index interface used by retrieval.
"""

from typing import Protocol

from .numeric import Float32NumpyArray, FloatNumpyArray, IntNumpyArray


class SearchIndex(Protocol):
    """
    Protocol for indexes that accelerate nearest-neighbour search.

    An index maps a batch of query vectors to the nearest gallery vectors. The
    integer ids returned by :meth:`search` are positions into :attr:`paths`.
    """

    @property
    def paths(self) -> list[str]:
        """Gallery image paths, ordered to match the index's internal ids."""
        ...

    def search(
        self,
        query_vectors: FloatNumpyArray,
        k: int,
    ) -> tuple[Float32NumpyArray, IntNumpyArray]:
        """
        Return the ``k`` nearest gallery vectors for each query vector.

        :param query_vectors: A ``(N, D)`` array of query vectors.
        :param k: Number of nearest neighbours to return per query.
        :return: A ``(scores, ids)`` tuple of ``(N, k)`` arrays. ``ids`` index
            into :attr:`paths`; missing neighbours are reported as ``-1``.
        """
        ...
