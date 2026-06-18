"""
Concrete FAISS-backed image indexes.

This module provides :class:`ImageIndexIVFFlat`, an inverted-file index that
partitions the gallery into ``nlist`` Voronoi cells and stores the full vectors
in each cell. Search probes ``nprobe`` cells, trading recall for speed.
"""

from __future__ import annotations

from typing import Any

import faiss

from ..image_store import ImageEncodingMap
from ..typing import Float32NumpyArray
from ._base_index import ImageIndex, Quantizer


class ImageIndexIVFFlat(ImageIndex):
    """
    Inverted-file (IVF) index storing the full gallery vectors per cell.

    Detail
    ------
    Check out this documentation if you are interested in the algorithm
    details:

    - https://milvus.io/ai-quick-reference/how-do-inverted-file-ivf-indexes-work-in-vector-databases-and-what-role-do-clustering-centroids-play-in-the-search-process

    :param encoding_map: Gallery mapping of image path to feature vector.
    :param quantizer: Distance metric, ``"l2"`` or ``"inner_product"`` (see
        :class:`~pyvisim.retrieval.ImageIndex`).
    :param nlist: Number of Voronoi cells to partition the gallery into.
    :param nprobe: Number of cells to scan per query at search time.
    :raises ValueError: If ``nlist`` or ``nprobe`` are out of range.
    """

    def __init__(
        self,
        encoding_map: ImageEncodingMap,
        *,
        quantizer: Quantizer = "l2",
        nlist: int = 100,
        nprobe: int = 1,
    ) -> None:
        self._nlist = int(nlist)
        self._nprobe = int(nprobe)
        super().__init__(encoding_map, quantizer=quantizer)

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

    def _learn_index(self) -> Any:
        """
        Build, train and populate the IVF-Flat index.

        :return: The trained FAISS ``IndexIVFFlat``.
        :raises ValueError: If ``nlist`` exceeds the gallery size or ``nprobe``
            is not within ``[1, nlist]``.
        """
        num_vectors = self._vectors.shape[0]
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
        index.train(self._vectors)
        index.add(self._vectors)
        index.nprobe = self._nprobe
        return index
