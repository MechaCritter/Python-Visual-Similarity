"""Adapter for search indexes built outside of this library."""

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Any, cast

import numpy as np

from ....typing import Float32NumpyArray, FloatNumpyArray, IntNumpyArray
from ....utils.validation import Param, validate_params
from ._utils import (
    as_decoded_gallery,
    as_gallery_matrix,
    as_id_array,
    as_query_matrix,
    as_read_only,
)

#: Name reported by an index the caller did not name.
DEFAULT_EXTERNAL_NAME = "external"


class ExternalSearchIndex:
    """
    A search index built elsewhere, adapted to the store's interface.

    .. important::
        Whether a score is a distance (lower is better) or a
        similarity (higher is better) depends on the metric the wrapped
        index was built for, and any necessary normalization of the vectors
        (for a cosine ranking) must be done by the caller, before the
        index is built.

    For more information, see the documentation:
    ``https://mechacritter.github.io/Python-Visual-Similarity/image_similarity_retrieval/image_store/external_search_index/external_search_index.html``.

    :param index: The index to search through. It must expose a
        ``search(queries, k)`` returning a ``(scores, ids)`` pair of ``(M, k)``
        arrays whose ids are row numbers into ``vectors``.
    :param vectors: The gallery vectors the index was built over, shape
        ``(N, D)``, in the order its ids refer to. The adapter keeps a copy of
        them.
    :param name: Name identifying the index, kept across a save/load round trip
        so a store can be rebuilt on a matching one. If ``None``,
        :data:`DEFAULT_EXTERNAL_NAME` is used.
    :raises AttributeError: If ``index`` has no ``search`` method.
    :raises ValueError: If ``vectors`` is not a non-empty 2-D matrix, or the
        index reports a size that does not match it.
    """

    def __init__(
        self,
        index: Any,
        vectors: FloatNumpyArray,
        *,
        name: str | None = None,
    ) -> None:
        self._adopt(index, name)
        self._gallery: _Gallery = _StoredGallery(vectors)

        indexed = getattr(index, "ntotal", None)
        if indexed is not None and int(indexed) != len(self._gallery):
            raise ValueError(
                f"The index holds {int(indexed)} vectors, but {len(self._gallery)} "
                f"were passed alongside it."
            )

    @classmethod
    def from_faiss_index(
        cls,
        index: Any,
        *,
        name: str | None = None,
    ) -> ExternalSearchIndex:
        """
        Adapt a FAISS index, reading its vectors back from it.

        .. important::
           To read from an `IVF <https://github.com/facebookresearch/faiss/wiki/Faiss-indexes>`_
           index, call ``faiss.extract_index_ivf(faiss_index).make_direct_map()`` first.
           Otherwise, reconstruction will fail. If the index cannot reconstruct its vectors
           and you need exact reconstruction, pass them to the constructor instead.

        Normalization, if any, must be done by the caller. An index built for
        ``METRIC_INNER_PRODUCT`` only ranks by cosine similarity if the vectors
        were L2-normalised before they were added, and the queries handed to
        :meth:`search` must be normalised the same way.

        :param index: The FAISS index to search through.
        :param name: Name identifying the index. If ``None``,
            :data:`DEFAULT_EXTERNAL_NAME` is used.
        :return: An :class:`ExternalSearchIndex` around ``index``.
        :raises ValueError: If the index is empty or cannot look its vectors up
            by row.
        """
        adapter = cls.__new__(cls)
        adapter._adopt(index, name)
        adapter._gallery = _FaissGallery(index)
        return adapter

    def _adopt(self, index: Any, name: str | None) -> None:
        """
        Take over the index to search through and the name identifying it.

        :param index: The index to search through.
        :param name: Name identifying the index. If ``None``,
            :data:`DEFAULT_EXTERNAL_NAME` is used.
        :raises AttributeError: If ``index`` has no ``search`` method.
        """
        if not callable(getattr(index, "search", None)):
            raise AttributeError(
                f"{type(index).__name__} has no 'search' method, so it cannot be "
                f"used as a search index."
            )
        self._index = index
        self._name = DEFAULT_EXTERNAL_NAME if name is None else str(name)

    @property
    def index(self) -> Any:
        """The wrapped index, as it was passed in."""
        return self._index

    @property
    def name(self) -> str:
        """Name identifying the index across a save/load round trip."""
        return self._name

    @property
    def vectors(self) -> Float32NumpyArray:
        """
        The ``(N, D)`` gallery matrix the index was built over, read-only.

        An adapter that :meth:`from_faiss_index` created decodes it from the
        index on every access.
        """
        return self._gallery.read_all()

    def vectors_at(self, ids: Sequence[int] | IntNumpyArray) -> Float32NumpyArray:
        """
        Read the gallery vectors stored under the given row numbers.

        :param ids: Gallery row numbers, shape ``(n,)``, at least one.
        :return: The ``(n, D)`` block of the requested vectors, read-only and in
            the given order.
        :raises ValueError: If ``ids`` is empty, not one-dimensional, holds
            non-integers, or names a row outside the gallery.
        """
        rows = as_id_array(ids, len(self))
        return self._gallery.read_rows(rows)

    @property
    def dim(self) -> int:
        """Dimensionality of the indexed vectors."""
        return self._gallery.dim

    def __len__(self) -> int:
        return len(self._gallery)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self._name!r}, "
            f"index={type(self._index).__name__}, num_vectors={len(self)}, "
            f"dim={self.dim})"
        )

    @validate_params(k=Param(int, ge=1))
    def search(
        self,
        query_vectors: FloatNumpyArray,
        k: int,
    ) -> tuple[Float32NumpyArray, IntNumpyArray]:
        """
        Return the ``k`` nearest gallery vectors for each query vector.

        The scores are whatever the wrapped index returns, unchanged.

        :param query_vectors: A ``(D,)`` vector or an ``(M, D)`` batch of query
            vectors.
        :param k: Number of nearest neighbors to return per query.
        :return: A ``(scores, ids)`` tuple of ``(M, k)`` arrays. ``ids`` are
            gallery row numbers, and missing neighbors are reported as ``-1``.
        :raises ValueError: If ``k`` is not a positive integer or the queries do
            not match the indexed dimensionality.
        """
        k = int(k)
        queries = as_query_matrix(query_vectors, self.dim)
        scores, ids = self._index.search(queries, k)
        return (
            cast(Float32NumpyArray, np.asarray(scores, dtype=np.float32)),
            cast(IntNumpyArray, np.asarray(ids, dtype=np.intp)),
        )


class _Gallery(abc.ABC):
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


class _StoredGallery(_Gallery):
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


class _FaissGallery(_Gallery):
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
