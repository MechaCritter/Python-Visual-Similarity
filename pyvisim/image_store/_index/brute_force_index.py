"""
Exhaustive nearest-neighbour search over a gallery.

:class:`BruteForceIndex` extends the compiled ``hnswlib`` brute-force index with
the surface the image store searches through. It compares every query against
every gallery vector, so its results are exact: the reference an approximate
index such as :class:`~pyvisim.image_store.HnswIndex` is measured against.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from ...typing import Float32NumpyArray, FloatNumpyArray, IntNumpyArray
from ._utils import (
    Space,
    as_gallery_matrix,
    as_query_matrix,
    as_read_only,
    is_explicit_thread_count,
    l2_normalize,
    pad_results,
    validate_k,
    validate_space,
)
from ._vendored import _hnswlib


class BruteForceIndex(_hnswlib.BFIndex):
    """
    Exhaustive (brute-force) index scanning the whole gallery per query.

    Every query is compared against every gallery vector, so the ranking is
    exact at a cost linear in the gallery size. It is the right choice for
    galleries small enough to scan and the baseline to measure an approximate
    index against.

    The index owns the gallery vectors: they are copied in at construction time
    and handed back as a read-only matrix, so no second copy is held.

    :param vectors: Gallery embedding vectors, shape ``(N, D)``.
    :param space: Metric space to search in, ``"cosine"``, ``"l2"`` or ``"ip"``.
        In cosine space the vectors are stored L2-normalised.
    :param num_threads: Threads used to run batched queries. ``-1`` uses every
        available core.
    :raises ValueError: If ``space`` is unknown or the gallery is not a
        non-empty 2-D matrix.
    """

    def __init__(
        self,
        vectors: FloatNumpyArray,
        *,
        space: Space = "cosine",
        num_threads: int = -1,
    ) -> None:
        validate_space(space)
        gallery = as_gallery_matrix(vectors)
        super().__init__(space, gallery.shape[1])

        self.init_index(gallery.shape[0])
        if is_explicit_thread_count(num_threads):
            self.set_num_threads(int(num_threads))
        self._vectors = self._store_gallery(gallery)

    def _store_gallery(self, gallery: Float32NumpyArray) -> Float32NumpyArray:
        """
        Add the gallery to the index and keep it as the read-only matrix.

        :param gallery: A fresh contiguous ``(N, D)`` float32 matrix.
        :return: The stored matrix, normalised in cosine space and locked
            against writes.
        """
        if self.space == "cosine":
            # The index normalises its own copy, so the matrix kept here is
            # normalised too and both describe the same gallery.
            l2_normalize(gallery)
        self.add_items(gallery, np.arange(gallery.shape[0]))
        return as_read_only(gallery)

    @property
    def vectors(self) -> Float32NumpyArray:
        """
        The ``(N, D)`` gallery matrix, read-only.

        In cosine space the vectors are L2-normalised, the form they were
        indexed in. Rebuild the index with :meth:`update` to change them.
        """
        return self._vectors

    def __len__(self) -> int:
        return int(self._vectors.shape[0])

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_vectors={len(self)}, dim={self.dim}, "
            f"space={self.space!r})"
        )

    def search(
        self,
        query_vectors: FloatNumpyArray,
        k: int,
    ) -> tuple[Float32NumpyArray, IntNumpyArray]:
        """
        Return the ``k`` nearest gallery vectors for each query vector.

        The scores are the distances of the index's metric space, so lower is
        always more similar: ``1 - cosine_similarity`` in cosine space,
        ``1 - inner_product`` in ``"ip"`` space and the squared Euclidean
        distance in ``"l2"`` space.

        :param query_vectors: A ``(D,)`` vector or an ``(M, D)`` batch of query
            vectors.
        :param k: Number of nearest neighbours to return per query.
        :return: A ``(scores, ids)`` tuple of ``(M, k)`` arrays. ``ids`` are
            gallery row numbers; a gallery holding fewer than ``k`` vectors pads
            the free columns with the id ``-1``.
        :raises ValueError: If ``k`` is not positive or the queries do not match
            the indexed dimensionality.
        """
        k = validate_k(k)
        queries = as_query_matrix(query_vectors, self.dim)
        labels, distances = self.knn_query(queries, k=min(k, len(self)))
        return pad_results(cast(IntNumpyArray, labels), distances, k)

    def update(self, vectors: FloatNumpyArray) -> None:
        """
        Replace the gallery and rebuild the index from scratch.

        :param vectors: The new gallery matrix, shape ``(N, D)``. Its
            dimensionality must match the one the index was built for.
        :raises ValueError: If the new gallery is not a non-empty 2-D matrix or
            its dimensionality differs from the indexed one.
        """
        gallery = as_gallery_matrix(vectors)
        if gallery.shape[1] != self.dim:
            raise ValueError(
                f"New vectors have dimensionality {gallery.shape[1]}, but the "
                f"index was built for {self.dim}."
            )
        self.reset_index()
        self.init_index(gallery.shape[0])
        self._vectors = self._store_gallery(gallery)
