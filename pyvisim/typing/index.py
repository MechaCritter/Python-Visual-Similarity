"""
Types describing the search indexes used by an image store.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, TypeAlias

from .numeric import Float32NumpyArray, FloatNumpyArray, IntNumpyArray

if TYPE_CHECKING:
    import faiss

    # FAISS is an optional dependency, so only type checkers know its index type.
    FaissIndex: TypeAlias = faiss.Index


class SearchIndex(Protocol):
    """Protocol for indexes that accelerate nearest-neighbor search."""

    @property
    def vectors(self) -> Float32NumpyArray: ...

    @property
    def dim(self) -> int: ...

    def vectors_at(self, ids: Sequence[int] | IntNumpyArray) -> Float32NumpyArray:
        """
        Read the vectors stored under the given row numbers.

        :param ids: Gallery row numbers, shape ``(n,)``, at least one.
        :return: The ``(n, D)`` block of the requested vectors, in the given
            order.
        """
        ...

    def search(
        self,
        query_vectors: FloatNumpyArray,
        k: int,
    ) -> tuple[Float32NumpyArray, IntNumpyArray]: ...
