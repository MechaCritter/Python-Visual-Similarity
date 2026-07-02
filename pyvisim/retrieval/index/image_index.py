"""
FAISS-backed image indexes.

This module provides two inverted-file (IVF) indexes that partition the gallery
into ``nlist`` Voronoi cells and probe ``nprobe`` of them per query:

- :class:`ImageIndexIVFFlat` stores the full gallery vectors in each cell, so it
  is exact within the probed cells.
- :class:`ImageIndexIVFPQ` compresses the vectors with product quantization,
  trading a little accuracy for a much smaller memory footprint.

Two further index structures, :class:`ImageIndexHNSW` and
:class:`ImageIndexScalarQuantizer`, are sketched for upcoming releases and
currently raise :class:`NotImplementedError` when constructed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ...lazy_import import OptionalImport
from ...typing import Float32NumpyArray, FloatNumpyArray
from ._base_index import ImageIndex, Quantizer

with OptionalImport(package="faiss", extra="search"):
    import faiss


class ImageIndexIVFFlat(ImageIndex):
    """
    Inverted-file (IVF) index storing the full gallery vectors per cell.

    Detail
    ------
    Check out this documentation if you are interested in the algorithm
    details:

    - https://milvus.io/ai-quick-reference/how-do-inverted-file-ivf-indexes-work-in-vector-databases-and-what-role-do-clustering-centroids-play-in-the-search-process

    :param paths: Gallery image paths, in the same order as ``vectors``.
    :param vectors: Gallery embedding vectors, shape ``(N, D)``.
    :param quantizer: Distance metric, ``"l2"`` or ``"inner_product"`` (see
        :class:`~pyvisim.retrieval.ImageIndex`).
    :param nlist: Number of Voronoi cells to partition the gallery into.
    :param nprobe: Number of cells to scan per query at search time.
    :raises ValueError: If ``nlist`` or ``nprobe`` are out of range.
    """

    def __init__(
        self,
        paths: Sequence[str],
        vectors: FloatNumpyArray,
        *,
        quantizer: Quantizer = "l2",
        nlist: int = 100,
        nprobe: int = 1,
    ) -> None:
        self._nlist = int(nlist)
        self._nprobe = int(nprobe)
        super().__init__(paths, vectors, quantizer=quantizer)

    @property
    def nlist(self) -> int:
        """Number of Voronoi cells the gallery is partitioned into."""
        return self._nlist

    @property
    def nprobe(self) -> int:
        """Number of cells scanned per query at search time."""
        return self._nprobe

    @property
    def cluster_centers(self) -> Float32NumpyArray:
        """Coordinates of the ``nlist`` cell centroids, shape ``(nlist, D)``."""
        return self._coarse_centroids()

    def _learn_index(self, vectors: FloatNumpyArray) -> Any:
        """
        Build, train and populate the IVF-Flat index.

        :param vectors: The contiguous ``(N, D)`` gallery matrix to index.
        :return: The trained FAISS ``IndexIVFFlat``.
        :raises ValueError: If ``nlist`` exceeds the gallery size or ``nprobe``
            is not within ``[1, nlist]``.
        """
        num_vectors = vectors.shape[0]
        if not 1 <= self._nlist <= num_vectors:
            raise ValueError(
                f"'nlist' must be between 1 and the number of indexed vectors "
                f"({num_vectors}), got {self._nlist}."
            )
        if not 1 <= self._nprobe <= self._nlist:
            raise ValueError(
                f"'nprobe' must be between 1 and 'nlist' ({self._nlist}), "
                f"got {self._nprobe}."
            )

        quantizer = self._make_quantizer()
        index = faiss.IndexIVFFlat(quantizer, self.dim, self._nlist, self._metric)
        index.train(vectors)
        index.add(vectors)
        index.nprobe = self._nprobe
        return index


class ImageIndexIVFPQ(ImageIndex):
    """
    Inverted-file (IVF) index with product-quantization (PQ) compression.

    Like :class:`ImageIndexIVFFlat`, the gallery is partitioned into ``nlist``
    cells and ``nprobe`` of them are scanned per query. Within each cell the
    vectors are PQ-compressed: each vector is split into ``m`` sub-vectors, and
    every sub-vector is encoded with ``nbits`` bits. This shrinks the index at
    the cost of approximate distances.

    Detail
    ------
    Check out this documentation if you are interested in the algorithm
    details:

    - https://www.pinecone.io/learn/series/faiss/product-quantization/

    :param paths: Gallery image paths, in the same order as ``vectors``.
    :param vectors: Gallery embedding vectors, shape ``(N, D)``.
    :param quantizer: Distance metric, ``"l2"`` or ``"inner_product"`` (see
        :class:`~pyvisim.retrieval.ImageIndex`).
    :param nlist: Number of Voronoi cells to partition the gallery into.
    :param nprobe: Number of cells to scan per query at search time.
    :param m: Number of sub-vectors each vector is split into. Must divide the
        feature dimensionality.
    :param nbits: Number of bits used to encode each sub-vector. The PQ
        codebook holds ``2 ** nbits`` centroids per sub-vector.
    :raises ValueError: If ``nlist``/``nprobe`` are out of range, ``m`` does not
        divide the dimensionality, or the gallery is too small to train the PQ
        codebook.
    """

    def __init__(
        self,
        paths: Sequence[str],
        vectors: FloatNumpyArray,
        *,
        quantizer: Quantizer = "l2",
        nlist: int = 100,
        nprobe: int = 1,
        m: int = 8,
        nbits: int = 8,
    ) -> None:
        self._nlist = int(nlist)
        self._nprobe = int(nprobe)
        self._m = int(m)
        self._nbits = int(nbits)
        super().__init__(paths, vectors, quantizer=quantizer)

    @property
    def nlist(self) -> int:
        """Number of Voronoi cells the gallery is partitioned into."""
        return self._nlist

    @property
    def nprobe(self) -> int:
        """Number of cells scanned per query at search time."""
        return self._nprobe

    @property
    def m(self) -> int:
        """Number of PQ sub-vectors each vector is split into."""
        return self._m

    @property
    def nbits(self) -> int:
        """Number of bits used to encode each PQ sub-vector."""
        return self._nbits

    @property
    def cluster_centers(self) -> Float32NumpyArray:
        """Coordinates of the ``nlist`` cell centroids, shape ``(nlist, D)``."""
        return self._coarse_centroids()

    def _learn_index(self, vectors: FloatNumpyArray) -> Any:
        """
        Build, train and populate the IVF-PQ index.

        :param vectors: The contiguous ``(N, D)`` gallery matrix to index.
        :return: The trained FAISS ``IndexIVFPQ``.
        :raises ValueError: If ``nlist`` exceeds the gallery size, ``nprobe`` is
            not within ``[1, nlist]``, ``m`` does not divide the dimensionality,
            or the gallery has fewer than ``2 ** nbits`` vectors.
        """
        num_vectors = vectors.shape[0]
        if not 1 <= self._nlist <= num_vectors:
            raise ValueError(
                f"'nlist' must be between 1 and the number of indexed vectors "
                f"({num_vectors}), got {self._nlist}."
            )
        if not 1 <= self._nprobe <= self._nlist:
            raise ValueError(
                f"'nprobe' must be between 1 and 'nlist' ({self._nlist}), "
                f"got {self._nprobe}."
            )
        if self.dim % self._m != 0:
            raise ValueError(
                f"'m' ({self._m}) must divide the vector dimensionality ({self.dim})."
            )
        min_train = 39 * (2**self._nbits)
        if num_vectors < min_train:
            raise ValueError(
                f"IVF-PQ with nbits={self._nbits} needs at least {min_train} "
                f"indexed vectors for reliable PQ codebook training "
                f"(FAISS recommends ~39 points per "
                f"centroid; and given 2 ** nbits = {2**self._nbits}) "
                f"centroids, one would need 39 * {2**self._nbits} = {39 * (2**self._nbits)} "
                f"vectors. Got {num_vectors} vectors instead."
            )

        quantizer = self._make_quantizer()
        index = faiss.IndexIVFPQ(
            quantizer, self.dim, self._nlist, self._m, self._nbits, self._metric
        )
        index.train(vectors)
        index.add(vectors)
        index.nprobe = self._nprobe
        return index


class ImageIndexHNSW(ImageIndex):
    """
    Hierarchical Navigable Small World (HNSW) graph index.

    HNSW builds a multi-layer proximity graph over the gallery and walks it
    greedily at query time, giving fast and accurate approximate search without
    a training step.

    .. note::
        This index is sketched for an upcoming release and is **not yet
        implemented**: constructing it raises :class:`NotImplementedError`.

    Detail
    ------
    Check out this documentation if you are interested in the algorithm
    details:

    - https://www.pinecone.io/learn/series/faiss/hnsw/

    :param paths: Gallery image paths, in the same order as ``vectors``.
    :param vectors: Gallery embedding vectors, shape ``(N, D)``.
    :param quantizer: Distance metric, ``"l2"`` or ``"inner_product"``.
    :param m: Number of neighbours stored per node in the graph.
    :param ef_construction: Breadth of the search used while building the graph.
    :param ef_search: Breadth of the search used at query time.
    """

    def __init__(
        self,
        paths: Sequence[str],
        vectors: FloatNumpyArray,
        *,
        quantizer: Quantizer = "l2",
        m: int = 32,
        ef_construction: int = 40,
        ef_search: int = 16,
    ) -> None:
        self._m = int(m)
        self._ef_construction = int(ef_construction)
        self._ef_search = int(ef_search)
        super().__init__(paths, vectors, quantizer=quantizer)

    @property
    def cluster_centers(self) -> Float32NumpyArray:
        """Unavailable: HNSW is a graph index without coarse centroids."""
        raise NotImplementedError(
            "ImageIndexHNSW is planned for a future release and is not yet implemented."
        )

    def _learn_index(self, vectors: FloatNumpyArray) -> Any:
        """
        Build the HNSW graph index (planned for a future release).

        :param vectors: The contiguous ``(N, D)`` gallery matrix to index.
        :raises NotImplementedError: Always; this index is not yet implemented.
        """
        raise NotImplementedError(
            "ImageIndexHNSW is planned for a future release and is not yet implemented."
        )


class ImageIndexScalarQuantizer(ImageIndex):
    """
    Flat index with scalar-quantized (e.g. 8-bit integer) gallery vectors.

    A scalar quantizer compresses every vector component independently to a
    small integer type, roughly quartering the memory of ``int8`` storage while
    keeping search exhaustive within the compressed space.

    .. note::
        This index is sketched for an upcoming release and is **not yet
        implemented**: constructing it raises :class:`NotImplementedError`.

    Detail
    ------
    Check out this documentation if you are interested in the algorithm
    details:

    - https://github.com/facebookresearch/faiss/wiki/Faiss-indexes#scalar-quantizer

    :param paths: Gallery image paths, in the same order as ``vectors``.
    :param vectors: Gallery embedding vectors, shape ``(N, D)``.
    :param quantizer: Distance metric, ``"l2"`` or ``"inner_product"``.
    :param qtype: Scalar-quantizer component type, e.g. ``"8bit"``.
    """

    def __init__(
        self,
        paths: Sequence[str],
        vectors: FloatNumpyArray,
        *,
        quantizer: Quantizer = "l2",
        qtype: str = "8bit",
    ) -> None:
        self._qtype = str(qtype)
        super().__init__(paths, vectors, quantizer=quantizer)

    @property
    def cluster_centers(self) -> Float32NumpyArray:
        """Unavailable: a flat scalar-quantizer index has no coarse centroids."""
        raise NotImplementedError(
            "ImageIndexScalarQuantizer is planned for a future release and is "
            "not yet implemented."
        )

    def _learn_index(self, vectors: FloatNumpyArray) -> Any:
        """
        Build the scalar-quantizer index (planned for a future release).

        :param vectors: The contiguous ``(N, D)`` gallery matrix to index.
        :raises NotImplementedError: Always; this index is not yet implemented.
        """
        raise NotImplementedError(
            "ImageIndexScalarQuantizer is planned for a future release and is "
            "not yet implemented."
        )
