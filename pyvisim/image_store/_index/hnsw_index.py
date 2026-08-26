"""Approximate nearest-neighbour search over an HNSW graph."""

from __future__ import annotations

from typing import cast

import numpy as np

from ...typing import Float32NumpyArray, FloatNumpyArray, IntNumpyArray
from ._bindings import _hnswlib
from ._utils import (
    Space,
    as_decoded_gallery,
    as_gallery_matrix,
    as_query_matrix,
    is_explicit_thread_count,
    pad_results,
    validate_k,
    validate_space,
)


class HnswIndex(_hnswlib.Index):
    """
    Hierarchical Navigable Small World (HNSW) graph index.

    HNSW builds a multi-layer proximity graph over the gallery and walks it
    greedily at query time, which gives fast approximate search without a
    training step. Recall is traded against speed through ``ef_construction``
    (how thoroughly the graph is built) and ``ef_search`` (how wide the walk is
    at query time).

    The graph owns the gallery vectors: they are copied into it at construction
    time and read back on demand, so no second copy is held.

    :param vectors: Gallery embedding vectors, shape ``(N, D)``.
    :param space: Metric space to build the graph for, ``"cosine"``, ``"l2"`` or
        ``"ip"``. In cosine space the vectors are stored L2-normalised.
    :param m: Number of bidirectional links created per node. Higher values
        improve recall on high-dimensional data at the cost of memory.
    :param ef_construction: Size of the candidate list used while building the
        graph. Higher values build a better graph, more slowly.
    :param ef_search: Size of the candidate list used at query time. It is
        raised to ``k`` whenever a search asks for more neighbours than that.
    :param random_seed: Seed of the level generator, which decides the layer
        each node is inserted at.
    :param num_threads: Threads used to build the graph and to run batched
        queries. ``-1`` uses every available core.
    :raises ValueError: If ``space`` is unknown or the gallery is not a
        non-empty 2-D matrix.
    """

    def __init__(
        self,
        vectors: FloatNumpyArray,
        *,
        space: Space = "cosine",
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        random_seed: int = 100,
        num_threads: int = -1,
    ) -> None:
        validate_space(space)
        gallery = as_gallery_matrix(vectors)
        super().__init__(space, gallery.shape[1])

        self._m = int(m)
        self._ef_construction = int(ef_construction)
        self._random_seed = int(random_seed)
        self._num_vectors = int(gallery.shape[0])
        self.init_index(
            max_elements=self._num_vectors,
            M=self._m,
            ef_construction=self._ef_construction,
            random_seed=self._random_seed,
        )
        self.ef = int(ef_search)
        if is_explicit_thread_count(num_threads):
            self.set_num_threads(int(num_threads))
        self.add_items(gallery, np.arange(self._num_vectors))

    @property
    def vectors(self) -> Float32NumpyArray:
        """
        The ``(N, D)`` gallery matrix, read back from the graph.

        The graph stores the vectors in its own layout, so each access decodes
        them into a fresh read-only array. In cosine space they come back
        L2-normalised, the form they were indexed in.
        """
        decoded = self.get_items(np.arange(self._num_vectors), return_type="numpy")
        return as_decoded_gallery(decoded)

    @property
    def ef_search(self) -> int:
        """Size of the candidate list used at query time."""
        return int(self.ef)

    @property
    def random_seed(self) -> int:
        """Seed of the level generator used to build the graph."""
        return self._random_seed

    def __len__(self) -> int:
        return self._num_vectors

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_vectors={len(self)}, dim={self.dim}, "
            f"space={self.space!r}, m={self.M}, "
            f"ef_construction={self.ef_construction}, ef_search={self.ef_search})"
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
        neighbours = min(k, self._num_vectors)
        if self.ef < neighbours:
            # The walk cannot return more neighbours than the candidate list it
            # keeps, so widen it to what this query asks for.
            self.ef = neighbours
        labels, distances = self.knn_query(queries, k=neighbours)
        return pad_results(cast(IntNumpyArray, labels), distances, k)

    def update(self, vectors: FloatNumpyArray) -> None:
        """
        Replace the gallery and rebuild the graph from scratch.

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
        ef_search = self.ef
        self._num_vectors = int(gallery.shape[0])
        self.reset_index()
        self.init_index(
            max_elements=self._num_vectors,
            M=self._m,
            ef_construction=self._ef_construction,
            random_seed=self._random_seed,
        )
        self.ef = ef_search
        self.add_items(gallery, np.arange(self._num_vectors))
