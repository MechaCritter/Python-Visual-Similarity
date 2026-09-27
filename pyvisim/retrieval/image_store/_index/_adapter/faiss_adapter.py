"""Adapter letting the image store search through a FAISS index."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeGuard

import numpy as np

from .....lazy_import import OptionalImport
from .....typing import Float32NumpyArray, FloatNumpyArray, IntNumpyArray
from .....utils.validation import Param, validate_params
from .._utils import Space, as_decoded_gallery, as_id_array, as_query_matrix

with OptionalImport(
    err_msg=(
        "FAISS is not installed. Please install it with: "
        "'uv pip install faiss-cpu' or 'pip install faiss-cpu'"
    )
) as _faiss_import:
    import faiss
    from faiss import Index as FaissIndex


def is_faiss_index(obj: object) -> TypeGuard[FaissIndex]:
    """
    Check whether an object is a FAISS index.

    :param obj: The object to check.
    :return: ``True`` if ``obj`` is a ``faiss.Index``.
    :raises ImportError: If FAISS is not installed.
    """
    _faiss_import.check()
    return isinstance(obj, FaissIndex)


class FaissIndexAdapter:
    """
    A FAISS index adapted to the interface the image store searches through.

    The adapter's search results' scores are the distances of the index's metric
    space, as the built-in indexes report them: the squared Euclidean distance
    in ``"l2"`` space and ``1 - inner_product`` in ``"ip"`` space.

    #TODO: in the future, more metrics than just L2 and inner product should
    be supported.

    :param index: The FAISS index, already holding the gallery.
    :raises ValueError: If the index was built for a metric other than
        ``METRIC_L2`` or ``METRIC_INNER_PRODUCT``, holds no vectors, or cannot
        look its vectors up by row.
    """

    def __init__(self, index: FaissIndex) -> None:
        self._space = _space_of(index)
        self._num_vectors = _num_vectors_of(index)
        self._index = index

    @property
    def faiss_index(self) -> FaissIndex:
        """The FAISS index, as it was passed in."""
        return self._index

    @property
    def name(self) -> str:
        """Name of the index type, such as ``"faiss.IndexHNSWFlat"``."""
        return f"faiss.{type(self._index).__name__}"

    @property
    def space(self) -> Space:
        """Metric space the index was built for, ``"l2"`` or ``"ip"``."""
        return self._space

    @property
    def vectors(self) -> Float32NumpyArray:
        """
        The ``(N, D)`` gallery matrix, decoded from the index on every access.

        A compressed index (product- or scalar-quantized) hands back an
        approximation of the vectors it was given.
        """
        decoded = np.empty((self._num_vectors, self.dim), dtype=np.float32)
        return as_decoded_gallery(
            self._index.reconstruct_n(0, self._num_vectors, decoded)
        )

    def vectors_at(self, ids: Sequence[int] | IntNumpyArray) -> Float32NumpyArray:
        """
        Read the gallery vectors stored under the given row numbers.

        :param ids: Gallery row numbers, shape ``(n,)``, at least one.
        :return: The ``(n, D)`` block of the requested vectors, read-only and in
            the given order.
        :raises ValueError: If ``ids`` is empty, not one-dimensional, holds
            non-integers, or names a row outside the gallery.
        """
        rows = as_id_array(ids, self._num_vectors)
        return as_decoded_gallery(
            self._index.reconstruct_batch(np.asarray(rows, dtype=np.int64))
        )

    @property
    def dim(self) -> int:
        """Dimensionality of the indexed vectors."""
        return int(self._index.d)

    def __len__(self) -> int:
        return self._num_vectors

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name!r}, "
            f"num_vectors={len(self)}, dim={self.dim}, space={self._space!r})"
        )

    @validate_params(k=Param(int, ge=1))
    def search(
        self,
        query_vectors: FloatNumpyArray,
        k: int,
    ) -> tuple[Float32NumpyArray, IntNumpyArray]:
        """
        Return the ``k`` nearest gallery vectors for each query vector.

        :param query_vectors: A ``(D,)`` vector or an ``(M, D)`` batch of query
            vectors.
        :param k: Number of nearest neighbors to return per query.
        :return: A ``(scores, ids)`` tuple of ``(M, k)`` arrays. The scores are
            distances, so lower is more similar. ``ids`` are gallery row
            numbers, and a missing neighbor is reported as the id ``-1`` with
            an infinite distance.
        :raises ValueError: If ``k`` is not a positive integer or the queries do
            not match the indexed dimensionality.
        """
        queries = as_query_matrix(query_vectors, self.dim)
        scores, labels = self._index.search(queries, int(k))
        ids = np.asarray(labels, dtype=np.intp)
        return _as_distances(scores, ids, self._space), ids


def _space_of(index: FaissIndex) -> Space:
    """
    Name the metric space a FAISS index was built for.

    :param index: The FAISS index.
    :return: ``"l2"`` for ``METRIC_L2``, ``"ip"`` for ``METRIC_INNER_PRODUCT``.
    :raises ValueError: If the index was built for any other metric.
    """
    spaces: dict[int, Space] = {
        faiss.METRIC_L2: "l2",
        faiss.METRIC_INNER_PRODUCT: "ip",
    }
    space = spaces.get(int(index.metric_type))
    if space is None:
        raise ValueError(
            f"{type(index).__name__} was built for the FAISS metric "
            f"{int(index.metric_type)}, but only METRIC_L2 and "
            f"METRIC_INNER_PRODUCT are supported."
        )
    return space


def _num_vectors_of(index: FaissIndex) -> int:
    """
    Count the vectors of a FAISS index after checking it can look them up by row.

    :param index: The FAISS index.
    :return: Number of vectors the index holds.
    :raises ValueError: If the index holds no vectors or cannot look them up by
        row.
    """
    num_vectors = int(index.ntotal)
    if num_vectors == 0:
        raise ValueError(
            "The FAISS index holds no vectors, so there is no gallery to search."
        )
    if not _looks_up_rows(index):
        raise ValueError(
            f"{type(index).__name__} cannot look its vectors up by row, so the "
            f"store cannot read its gallery back. An IVF index can do so after "
            f"faiss.extract_index_ivf(index).make_direct_map()."
        )
    return num_vectors


def _looks_up_rows(index: FaissIndex) -> bool:
    """
    Check whether a FAISS index can decode a stored vector from its row.

    :param index: The FAISS index, holding at least one vector.
    :return: ``True`` if the vector in row ``0`` could be decoded.
    """
    try:
        index.reconstruct_batch(np.zeros(1, dtype=np.int64))
    except RuntimeError:
        # Not every index keeps its vectors around: a purely compressed index
        # has none to give back, and an IVF index needs a direct map first.
        return False
    return True


def _as_distances(
    scores: Float32NumpyArray,
    ids: IntNumpyArray,
    space: Space,
) -> Float32NumpyArray:
    """
    Turn the scores of a FAISS search into distances of the index's space.

    An inner-product index scores by similarity, which becomes the
    ``1 - inner_product`` distance of the built-in ``"ip"`` space. FAISS pads a
    missing neighbor with the largest finite float, which is replaced by an
    infinite distance, as the built-in indexes report it.

    :param scores: The ``(M, k)`` scores FAISS returned.
    :param ids: The ``(M, k)`` gallery row numbers FAISS returned, ``-1`` for a
        missing neighbor.
    :param space: Metric space the index was built for.
    :return: The ``(M, k)`` distances.
    """
    distances = np.asarray(1.0 - scores if space == "ip" else scores, np.float32)
    distances[ids < 0] = np.inf
    return distances
