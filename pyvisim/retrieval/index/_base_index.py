"""
Abstract interface for the image indexes.

This module defines :class:`ImageIndex`, the base class shared by every
accelerated image index. A concrete index turns a gallery of embedding vectors
(paired with their image paths) into a trained FAISS index upon construction
(via the :meth:`ImageIndex._learn_index` hook) and exposes a batched
:meth:`ImageIndex.search`. Subclasses pick the index structure (e.g. IVF-Flat,
IVF-PQ); this base handles the shared concerns: validating the metric,
materialising the gallery matrix, optionally L2-normalising it for
inner-product search, and mapping FAISS ids back to image paths.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Any, Literal, cast

import faiss
import numpy as np

from ...typing import Float32NumpyArray, FloatNumpyArray, IntNumpyArray

#: Supported quantizer/metric choices, mapped to their FAISS metric constant.
_METRICS: dict[str, int] = {
    "l2": faiss.METRIC_L2,
    "inner_product": faiss.METRIC_INNER_PRODUCT,
}

#: Literal alias for the accepted ``quantizer`` argument.
Quantizer = Literal["l2", "inner_product"]


class ImageIndex(abc.ABC):
    """
    Abstract base for all image indexes.

    :param paths: Gallery image paths, in the same order as ``vectors``. Their
        order defines the integer ids used by the index.
    :param vectors: Gallery embedding vectors, shape ``(N, D)``. Row ``i`` is the
        encoding of ``paths[i]``.
    :param quantizer: Distance metric to build the index for. ``"l2"`` uses
        Euclidean distance; ``"inner_product"`` uses the dot product and the
        gallery vectors are L2-normalised first, so it ranks by cosine
        similarity.
    :raises ValueError: If ``quantizer`` is unknown, the gallery is empty, the
        vectors are not a 2-D matrix, or ``paths`` and ``vectors`` differ in
        length.
    """

    def __init__(
        self,
        paths: Sequence[str],
        vectors: FloatNumpyArray,
        *,
        quantizer: Quantizer = "l2",
    ) -> None:
        if quantizer not in _METRICS:
            raise ValueError(
                f"Unsupported quantizer {quantizer!r}. Supported quantizers are: "
                f"{sorted(_METRICS)}."
            )

        # Copy into a fresh contiguous float32 matrix so the in-place
        # normalisation below never mutates the caller's embeddings.
        gallery = np.array(vectors, dtype=np.float32)
        if gallery.ndim != 2:
            raise ValueError(
                "Gallery encodings must form a 2-D (num_vectors, dim) matrix."
            )
        if gallery.shape[0] == 0:
            raise ValueError("Cannot build an index from an empty gallery.")
        if len(paths) != gallery.shape[0]:
            raise ValueError(
                f"Number of paths ({len(paths)}) must match the number of "
                f"gallery vectors ({gallery.shape[0]})."
            )

        self._quantizer: Quantizer = quantizer
        self._metric = _METRICS[quantizer]
        self._paths: list[str] = list(paths)
        self._dim = int(gallery.shape[1])

        if quantizer == "inner_product":
            # Normalise before adding so dot-product search ranks by cosine.
            faiss.normalize_L2(gallery)

        # The trained FAISS index is the sole owner of the gallery vectors; the
        # ``gallery`` matrix is dropped once the index has copied it in, so the
        # embeddings are kept in memory only once (see :meth:`reconstruct`).
        self._index: Any = self._learn_index(np.ascontiguousarray(gallery))

    @abc.abstractmethod
    def _learn_index(self, vectors: Float32NumpyArray) -> Any:
        """
        Build and train the index over the gallery vectors.

        Called once during construction. Implementations train the index, add
        every gallery vector and return the ready-to-search index. The
        ``vectors`` argument is not retained by the base class.

        :param vectors: The contiguous ``(N, D)`` gallery matrix to index.
        :return: The trained index instance.
        """
        ...

    @property
    @abc.abstractmethod
    def cluster_centers(self) -> Float32NumpyArray:
        """Coordinates of the coarse-quantizer centroids, shape ``(nlist, D)``."""
        ...

    @property
    def index(self) -> Any:
        """The underlying trained index."""
        return self._index

    @property
    def paths(self) -> list[str]:
        """Gallery image paths, ordered to match the index ids."""
        return list(self._paths)

    @property
    def quantizer(self) -> str:
        """The distance metric the index was built for."""
        return self._quantizer

    @property
    def dim(self) -> int:
        """Dimensionality of the indexed feature vectors."""
        return self._dim

    def reconstruct(self) -> Float32NumpyArray:
        """
        Reconstruct the gallery embedding matrix from the trained index.

        The vectors are read back from FAISS, so no separate copy of the
        gallery is kept in memory. For an inner-product index they come back
        L2-normalised (the form in which they were indexed); for a lossy index
        such as IVF-PQ they are the decompressed approximation.

        :return: The gallery embeddings, shape ``(N, D)``, in path order.
        """
        ivf = faiss.extract_index_ivf(self._index)
        ivf.make_direct_map()
        vectors = self._index.reconstruct_n(0, self._index.ntotal)
        return cast(Float32NumpyArray, np.asarray(vectors, dtype=np.float32))

    def __len__(self) -> int:
        return len(self._paths)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_vectors={len(self)}, dim={self.dim}, "
            f"quantizer={self._quantizer!r})"
        )

    def search(
        self,
        query_vectors: FloatNumpyArray,
        k: int,
    ) -> tuple[Float32NumpyArray, IntNumpyArray]:
        """
        Return the ``k`` nearest gallery vectors for each query vector.

        :param query_vectors: A ``(D,)`` vector or a ``(N, D)`` batch of query
            vectors. For an inner-product index the queries are L2-normalised to
            match the gallery.
        :param k: Number of nearest neighbours to return per query.
        :return: A ``(scores, ids)`` tuple of ``(N, k)`` arrays. ``ids`` index
            into :attr:`paths`; missing neighbours are reported as ``-1``.
        :raises ValueError: If ``k`` is not a positive integer.
        """
        if k < 1:
            raise ValueError(f"'k' must be a positive integer, got {k}.")

        queries = np.array(query_vectors, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        queries = np.ascontiguousarray(queries)
        if self._quantizer == "inner_product":
            faiss.normalize_L2(queries)

        scores, ids = self._index.search(queries, k)
        return (
            cast(Float32NumpyArray, np.asarray(scores, dtype=np.float32)),
            cast(IntNumpyArray, np.asarray(ids, dtype=np.intp)),
        )

    def _make_quantizer(self) -> Any:
        """
        Build the coarse quantizer (a flat index) matching the chosen metric.

        :return: A FAISS flat index used as the IVF coarse quantizer.
        """
        if self._quantizer == "inner_product":
            return faiss.IndexFlatIP(self.dim)
        return faiss.IndexFlatL2(self.dim)

    def _coarse_centroids(self) -> Float32NumpyArray:
        """
        Reconstruct the coarse-quantizer centroids of an IVF index.

        :return: The centroid coordinates, shape ``(nlist, D)``.
        """
        ivf = faiss.extract_index_ivf(self._index)
        quantizer = faiss.downcast_index(ivf.quantizer)
        centroids = quantizer.reconstruct_n(0, ivf.nlist)
        return cast(Float32NumpyArray, np.asarray(centroids, dtype=np.float32))
