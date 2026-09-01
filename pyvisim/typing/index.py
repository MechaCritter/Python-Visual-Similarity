"""
Structural type describing the search-index interface used by an image store.
"""

from typing import Protocol

from .numeric import Float32NumpyArray, FloatNumpyArray, IntNumpyArray


class SearchIndex(Protocol):
    """Protocol for indexes that accelerate nearest-neighbour search."""

    @property
    def vectors(self) -> Float32NumpyArray: ...

    @property
    def dim(self) -> int: ...

    def search(
        self,
        query_vectors: FloatNumpyArray,
        k: int,
    ) -> tuple[Float32NumpyArray, IntNumpyArray]: ...
