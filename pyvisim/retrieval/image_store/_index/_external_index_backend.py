"""Gallery sources an external search index reads its vectors from."""

from __future__ import annotations

import abc
from typing import Any

import numpy as np

from ....typing import Float32NumpyArray, FloatNumpyArray, IntNumpyArray
from ._utils import as_decoded_gallery, as_gallery_matrix, as_read_only


class Gallery(abc.ABC):
    """Source an :class:`ExternalSearchIndex` reads its gallery vectors from."""

    @abc.abstractmethod
    def __len__(self) -> int: ...

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """Dimensionality of the vectors."""

    @abc.abstractmethod
    def read_all(self) -> Float32NumpyArray:
        """
        Read the whole gallery.

        :return: The ``(N, D)`` matrix, read-only.
        """

    @abc.abstractmethod
    def read_rows(self, rows: IntNumpyArray) -> Float32NumpyArray:
        """
        Read the vectors under the given row numbers.

        :param rows: Validated gallery row numbers, shape ``(n,)``.
        :return: The ``(n, D)`` block of the requested vectors, read-only.
        """


class GenericGallery(Gallery):
    """
    Gallery vectors the adapter keeps a copy of.

    :param vectors: The gallery vectors, shape ``(N, D)``.
    :raises ValueError: If ``vectors`` is not a non-empty 2-D matrix.
    """

    def __init__(self, vectors: FloatNumpyArray) -> None:
        self._matrix = as_read_only(as_gallery_matrix(vectors))

    def __len__(self) -> int:
        return int(self._matrix.shape[0])

    @property
    def dim(self) -> int:
        return int(self._matrix.shape[1])

    def read_all(self) -> Float32NumpyArray:
        return self._matrix

    def read_rows(self, rows: IntNumpyArray) -> Float32NumpyArray:
        return as_read_only(np.ascontiguousarray(self._matrix[rows]))


class FaissGallery(Gallery):
    """
    Gallery vectors a FAISS index decodes from its own storage on every read.

    :param index: The FAISS index holding the gallery.
    :raises ValueError: If the index is empty or cannot look its vectors up
        by row.
    """

    def __init__(self, index: Any) -> None:
        self._index = index
        self._num_vectors = _faiss_gallery_size(index)
        self._dim = int(index.d)

    def __len__(self) -> int:
        return self._num_vectors

    @property
    def dim(self) -> int:
        return self._dim

    def read_all(self) -> Float32NumpyArray:
        return as_decoded_gallery(self._index.reconstruct_n(0, self._num_vectors))

    def read_rows(self, rows: IntNumpyArray) -> Float32NumpyArray:
        return as_decoded_gallery(self._index.reconstruct_batch(rows))


def _faiss_gallery_size(index: Any) -> int:
    """
    Count the vectors of a FAISS index after checking it can look them up by row.

    :param index: The FAISS index.
    :return: Number of vectors the index holds.
    :raises ValueError: If the index is empty or cannot look its vectors up by
        row.
    """
    num_vectors = getattr(index, "ntotal", None)
    if num_vectors is not None and int(num_vectors) == 0:
        raise ValueError("Cannot build an index from an empty gallery.")
    if num_vectors is None or not _looks_up_rows(index):
        raise ValueError(
            f"{type(index).__name__} cannot look its vectors up by row, so they "
            f"must be passed explicitly: "
            f"ExternalSearchIndex(index, vectors). An IVF index "
            f"can do so after faiss.extract_index_ivf(index).make_direct_map()."
        )
    return int(num_vectors)


def _looks_up_rows(index: Any) -> bool:
    """
    Check whether a FAISS index can decode a stored vector from its row.

    :param index: The FAISS index, holding at least one vector.
    :return: ``True`` if the vector in row ``0`` could be decoded.
    """
    try:
        index.reconstruct_batch(np.zeros(1, dtype=np.int64))
    except (AttributeError, RuntimeError, TypeError):
        # Not every index keeps its vectors around: a purely compressed index
        # has none to give back, and an IVF index needs a direct map first.
        return False
    return True
