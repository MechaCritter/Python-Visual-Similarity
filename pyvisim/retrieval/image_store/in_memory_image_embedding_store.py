"""
In-memory image embedding storage that uses a search index to accelerate retrieval.
"""

from __future__ import annotations

import functools
import itertools
import math
import pathlib
import warnings
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar, TypeVar, cast

import numpy as np
from PIL import Image, UnidentifiedImageError

from ...base import SerializableImageEmbedder
from ...serialization import SerializerMixin, decode_array_node
from ...typing import (
    BoolNumpyArray,
    Embedder,
    Float32NumpyArray,
    Float64NumpyArray,
    FloatNumpyArray,
    ImageInput,
    IntNumpyArray,
    NumpyArray,
    SearchIndex,
    UInt8NumpyArray,
)
from ...utils.validation import Param, validate_params
from ..data import Candidate
from ._index import (
    BRUTE_FORCE_TO_HNSWLIB,
    HNSW_TO_HNSWLIB,
    BruteForceIndex,
    ExternalSearchIndex,
    HnswIndex,
    Space,
    validate_index_params,
)

# Value of ``search_index`` selecting the HNSW graph.
_HNSW = "hnsw"
# Name under which a brute-force store records its index.
_BRUTE_FORCE = "brute-force"

# Parameter table of each index the store can build, keyed by index name. It
# names the parameters the index takes and maps them onto the backend's own.
_INDEX_PARAM_TABLES: dict[str, dict[str, str]] = {
    _HNSW: HNSW_TO_HNSWLIB,
    _BRUTE_FORCE: BRUTE_FORCE_TO_HNSWLIB,
}

# Keyword argument of :meth:`InMemoryImageEmbeddingStore.load_from_disk` that
# carries a rebuilt index instead of being forwarded to the embedder.
_SEARCH_INDEX_KWARG = "search_index"

# All index classes the store uses
_GalleryIndex = ExternalSearchIndex | HnswIndex | BruteForceIndex

# Threads decoding image files while the embedder works on the previous batch.
_DEFAULT_NUM_WORKERS = 4
# Batches the decoding threads may run ahead of the embedder.
_DEFAULT_NUM_PREFETCH_BATCHES = 4

_GenericMethodT = TypeVar("_GenericMethodT", bound=Callable[..., Any])


def _requires_built_store(method: _GenericMethodT) -> _GenericMethodT:
    """Make a store method raise while the store has not been built yet."""

    @functools.wraps(method)
    def checked(store: InMemoryImageEmbeddingStore, *args: Any, **kwargs: Any) -> Any:
        if not store.is_built:
            raise RuntimeError(
                "This store holds no gallery yet. Call 'build_store()' on it to "
                "embed the images and build the index, or construct the store "
                "with 'lazy_build=False'."
            )
        return method(store, *args, **kwargs)

    return cast(_GenericMethodT, checked)


class InMemoryImageEmbeddingStore(SerializerMixin):
    """
    Embed a gallery of images and index their embeddings for fast retrieval.

    .. important::

        - The store holds no embeddings until :meth:`build_store` is called
          once. Call it right after constructing the store, or pass
          ``lazy_build=False`` and the constructor calls it. Searching an
          unbuilt store or reading its embeddings or its index raises
          :class:`RuntimeError`.
        - If ``hnsw`` is used, the memory consumption is higher than when only
          using ``brute-force``, since the ``hnsw`` index builds an additional
          graph structure based on the embeddings.
        - For **FAISS**-based indexes, the returned embeddings may not be
          exactly the same as the original embeddings due to compression or
          quantization, and for some indexes, reconstruction is impossible.
        - If you use an :class:`~pyvisim.retrieval.image_store.ExternalSearchIndex`
          instead, that index must already hold the gallery. ``image_paths``
          then assumes each path matches the corresponding row in the index.

    For more information, see the documentation:
    ``https://mechacritter.github.io/Python-Visual-Similarity/image_similarity_retrieval/image_store/in_memory_image_embedding_store/in_memory_image_embedding_store.html``.

    :param image_paths: Iterable of image file paths to embed. Duplicates are
        dropped, keeping the first occurrence.
    :param embedder: Embedder used to turn images into feature vectors. This
        can be any object that implements the ``embed`` method, including
        ``Siamese`` and ``Triplet`` networks, the ``ClipEmbedder`` and the
        ``VLAD``/``Fisher Vector`` embedders.
    :param search_index: The index to search the gallery through. Pass
        ``"hnsw"`` for the HNSW graph algorithm, ``None`` for brute-force
    :param space: Metric space the index is built for, ``"cosine"``,
        ``"l2"`` or ``"ip"``. Ignored by an external index, which
        brings its own metric. An overview below:

        .. list-table::
           :header-rows: 1
           :widths: 15 30 55

           * - ``space``
             - Score
             - Notes
           * - ``"cosine"``
             - ``1 - cosine_similarity``
             - The vectors are stored L2-normalised, so their
               magnitudes are lost.
           * - ``"ip"``
             - ``1 - inner_product``
             - Stores the vectors as given. Only ranks by cosine similarity
               if you normalised them yourself.
           * - ``"l2"``
             - Squared Euclidean distance
             - Stores the vectors as given.

    :param index_params: Optional keyword parameters forwarded to the index
        constructor. The accepted parameters per index are listed under
        :ref:`Index parameters <store-index-parameters>`; anything else is
        rejected. ``space`` belongs to the store itself and is not accepted
        here.
    :param skip_errors: If ``True``, images that cannot be read or embedded are
        skipped with a warning instead of aborting.
    :param num_workers: Threads reading the gallery image files while the
        embedder works on the previous batch. ``1`` reads them on the calling
        thread. Ignored by an external index, which embeds nothing.
    :param num_prefetch_batches: Batches of images the reading threads may run
        ahead of the embedder. Higher values hide slower file reads at the cost
        of holding more decoded images in memory. Ignored by an external index,
        which embeds nothing.
    :param lazy_build: If ``True``, the gallery is embedded and indexed by the
        first call to :meth:`build_store`. If ``False``, the constructor calls
        :meth:`build_store` itself and the store is ready to search. Ignored by
        an external index, which already holds its gallery.
    :raises ValueError: If ``search_index`` is unknown, ``num_workers`` or
        ``num_prefetch_batches`` is not positive, no image path was given, an
        external index does not hold one vector per path, or ``index_params``
        names a parameter the index does not take. With ``lazy_build=False``,
        it is also raised if no image could be embedded.
    :raises TypeError: If any provided path is not a string.

    .. _store-index-parameters:

    Index parameters
    ----------------

    ``HnswIndex`` parameters
    ~~~~~~~~~~~~~~~~~~~~~~~~

    .. list-table::
       :header-rows: 1
       :widths: 20 15 65

       * - Parameter
         - Default
         - Explanation
       * - ``graph_degree``
         - ``16``
         - Bidirectional links created per node. Higher values raise recall on
           high-dimensional data and cost memory.
       * - ``build_candidates``
         - ``200``
         - Size of the candidate list kept while building the graph. Higher
           values build a better graph, more slowly.
       * - ``search_candidates``
         - ``50``
         - Size of the candidate list kept at query time. Higher values raise
           recall and cost query time. A search for more than
           ``search_candidates`` neighbors raises it to ``k``.
       * - ``random_seed``
         - ``100``
         - Seed of the level generator, which decides the layer each vector is
           inserted at.
       * - ``num_threads``
         - ``-1``
         - Threads used to build the graph and to run batched queries. ``-1``
           uses every available core.

    ``BruteForceIndex`` parameters
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. list-table::
       :header-rows: 1
       :widths: 20 15 65

       * - Parameter
         - Default
         - Explanation
       * - ``num_threads``
         - ``-1``
         - Threads used to run batched queries. ``-1`` uses every available
           core.
    """

    __metadata_key__: ClassVar[str] = "pyvisim_store"

    __format_version__: ClassVar[int] = 4
    __state_keys__: ClassVar[frozenset[str]] = frozenset(
        {
            "index_name",
            "space",
            "index_params",
            "paths",
            "embeddings",
            "graph",
            "embedder",
        }
    )

    @validate_params(
        num_workers=Param(int, ge=1),
        num_prefetch_batches=Param(int, ge=1),
    )
    def __init__(
        self,
        image_paths: Iterable[str],
        embedder: Embedder,
        search_index: str | ExternalSearchIndex | None = None,
        *,
        space: Space = "cosine",
        index_params: dict[str, Any] | None = None,
        skip_errors: bool = False,
        num_workers: int = _DEFAULT_NUM_WORKERS,
        num_prefetch_batches: int = _DEFAULT_NUM_PREFETCH_BATCHES,
        lazy_build: bool = True,
    ) -> None:
        _validate_search_index(search_index)

        self._embedder = embedder
        self._space: Space = space
        self._index_params: dict[str, Any] = dict(index_params or {})

        self._skip_errors = skip_errors
        self._num_workers = num_workers
        self._num_prefetch_batches = num_prefetch_batches

        if isinstance(search_index, ExternalSearchIndex):
            self._paths = _validated_paths(image_paths)
            self._index: SearchIndex | None = _aligned_index(search_index, self._paths)
            self._index_name = search_index.name
            return

        self._index_name = _HNSW if search_index == _HNSW else _BRUTE_FORCE
        # Checked before a single image is read: a misspelled parameter should
        # not cost the caller a full embedding pass first.
        _validate_index_params(self._index_name, self._index_params)

        # Collecting the paths here consumes a generator exactly once and reports
        # a bad path right away, so the caller never holds a store that cannot
        # be built.
        self._paths = _validated_paths(image_paths)
        self._index = None

        if not lazy_build:
            self.build_store()

    @classmethod
    def _from_components(
        cls,
        paths: list[str],
        embedder: Embedder,
        index: _GalleryIndex,
        index_name: str,
        space: Space,
        index_params: dict[str, Any],
    ) -> InMemoryImageEmbeddingStore:
        """
        Assemble a built store around an index that already holds its gallery.

        :param paths: Gallery image paths, ordered to match the index's rows.
        :param embedder: The reconstructed embedder.
        :param index: The index holding the gallery.
        :param index_name: Name of the index.
        :param space: Metric space the index is built for.
        :param index_params: Keyword parameters the index was built with.
        :return: A populated :class:`InMemoryImageEmbeddingStore`.
        :raises ValueError: If ``index`` does not hold one vector per path.
        """
        store = cls.__new__(cls)
        store._embedder = embedder
        store._space = space
        store._index_params = dict(index_params)
        store._paths = list(paths)
        store._skip_errors = False
        store._num_workers = _DEFAULT_NUM_WORKERS
        store._num_prefetch_batches = _DEFAULT_NUM_PREFETCH_BATCHES
        store._index = _aligned_index(index, store._paths)
        store._index_name = index_name
        return store

    def build_store(self) -> None:
        """
        Embed the gallery images and build the search index over them.

        A store constructed with ``lazy_build=True`` cannot be searched until
        this has run, so call it right after the constructor. Calling it on a
        store that is already built does nothing.

        With ``skip_errors`` on, the paths whose image could not be embedded
        are dropped, so :attr:`paths` can end up shorter than the list the store
        was constructed with.

        :raises ValueError: If no image could be embedded.
        :raises FileNotFoundError: If an image is missing and ``skip_errors``
            is off.
        """
        if self._index is not None:
            return
        paths, embeddings = _embed_image_paths_and_drop_duplicates(
            self._paths,
            self._embedder,
            self._skip_errors,
            self._num_workers,
            self._num_prefetch_batches,
        )
        self._paths = paths
        # Only the index keeps the gallery vectors. The local ``embeddings``
        # matrix is freed once the index has copied it in.
        self._index = _build_index(
            self._index_name, embeddings, self._space, self._index_params
        )

    @property
    def is_built(self) -> bool:
        """Whether the gallery has been embedded and indexed."""
        return self._index is not None

    @property
    def _search_index(self) -> SearchIndex:
        """The index of a built store, read by the members that require one."""
        return cast(SearchIndex, self._index)

    @property
    def paths(self) -> list[str]:
        """
        Gallery image paths, ordered to match the embedding rows.

        Before :meth:`build_store` has run, these are the paths the store was
        constructed with, minus duplicates. The build then removes the paths
        whose image could not be embedded.
        """
        return list(self._paths)

    @property
    @_requires_built_store
    def embeddings(self) -> Float32NumpyArray:
        """
        The ``(N, D)`` gallery embedding matrix, read back from the index.

        :raises RuntimeError: If the store has not been built yet.
        """
        return self._search_index.vectors

    @functools.cached_property
    def _row_by_path(self) -> dict[str, int]:
        """Gallery row number of every path, built on first use."""
        return {path: row for row, path in enumerate(self._paths)}

    @_requires_built_store
    def embeddings_of(self, paths: Sequence[str]) -> Float32NumpyArray:
        """
        Read the embeddings of the given gallery images back from the index.

        Only the requested rows are decoded, so looking a few images up this
        way is far cheaper than slicing :attr:`embeddings`, whose every access
        decodes the whole gallery.

        :param paths: Gallery image paths, at least one, in the order their
            rows are wanted in.
        :return: The ``(len(paths), D)`` block of their embeddings, read-only.
            In cosine space they come back L2-normalised, the form they were
            indexed in.
        :raises ValueError: If no path is given or a path is not in the
            gallery.
        :raises RuntimeError: If the store has not been built yet.
        """
        if len(paths) == 0:
            raise ValueError("'paths' must name at least one gallery image, got none.")
        rows = self._row_by_path
        missing = [path for path in paths if path not in rows]
        if missing:
            raise ValueError(
                f"{len(missing)} path(s) are not in the gallery, e.g. {missing[0]!r}."
            )
        return self._search_index.vectors_at(
            np.asarray([rows[path] for path in paths], dtype=np.intp)
        )

    @property
    def embedder(self) -> Embedder:
        """The embedder used to build the gallery and to embed queries."""
        return self._embedder

    @property
    @_requires_built_store
    def index(self) -> SearchIndex:
        """
        The search index the gallery is searched through.

        :raises RuntimeError: If the store has not been built yet.
        """
        return self._search_index

    @property
    def index_name(self) -> str:
        """Name of the index the store was built with."""
        return self._index_name

    @property
    def space(self) -> str:
        """The metric space the index was built for."""
        return self._space

    @property
    def index_params(self) -> dict[str, Any]:
        """Keyword parameters forwarded to the index constructor."""
        return dict(self._index_params)

    @property
    @_requires_built_store
    def dim(self) -> int:
        """
        Dimensionality of the gallery embeddings.

        :raises RuntimeError: If the store has not been built yet.
        """
        return self._search_index.dim

    def __len__(self) -> int:
        return len(self._paths)

    def __contains__(self, path: object) -> bool:
        return path in self._paths

    def __repr__(self) -> str:
        dim = self._index.dim if self._index is not None else None
        return (
            f"{self.__class__.__name__}(num_images={len(self)}, dim={dim}, "
            f"index_name={self._index_name!r}, space={self._space!r})"
        )

    @_requires_built_store
    @validate_params(
        expansion_alpha=Param(float, ge=0, lt=math.inf),
        expansion_neighbors=Param(int, ge=1),
    )
    def retrieve_top_k_similar(
        self,
        query_images: ImageInput,
        k: int = 5,
        *,
        query_expansion: bool = False,
        expansion_alpha: float = 3.0,
        expansion_neighbors: int = 50,
    ) -> list[list[Candidate]]:
        """
        Return the top-k most similar gallery images for each query image.

        The query images are embedded with this store's embedder and matched
        against the gallery through its index.

        With ``query_expansion`` on, every query is refined by the
        alpha-weighted query expansion (αQE) of Radenović et al. [1] before the
        final search: the query is searched once, the embeddings of the
        ``expansion_neighbors`` best matches are read back from the index, and
        the query is replaced by the L2-normalised weighted average of itself
        and those matches, each match weighted by its cosine similarity to the
        query raised to ``expansion_alpha``. A match whose similarity is not
        positive weighs nothing. The results are thereby pulled towards the
        whole neighborhood the query belongs to rather than the single point
        that was embedded. ``expansion_alpha=0`` weights every match that
        resembles the query alike, which is the classic average query expansion
        (AQE). The expansion costs one extra index search per query plus the
        decoding of ``expansion_neighbors`` gallery vectors, which is why
        ``query_expansion`` is off unless set.

        The expansion is defined on L2-normalised embeddings ranked by cosine
        similarity. The weights are computed on L2-normalised copies of the
        vectors whatever ``space`` the store was built in, so a gallery of
        non-normalised embeddings searched in ``"l2"`` or ``"ip"`` space is
        averaged as if it were normalised.

        According to [1], enabling **query expansion** improved mAP substantially.

        :param query_images: A single image or a batch/iterable of images to use
            as queries. Anything accepted by the store's embedder is valid.
        :param k: Number of top similar gallery images to return per query.
        :param query_expansion: Whether to refine every query with the alpha
            query expansion before the final search.
        :param expansion_alpha: Exponent applied to the cosine similarity of
            each match to weight it in the expanded query. ``0`` weights every
            match with a positive similarity alike. [1] uses ``3``.
        :param expansion_neighbors: Number of top-ranked gallery images
            averaged into the expanded query. [1] uses ``50``.
        :return: One ranked list of :class:`Candidate` matches per query image,
            in the same order as ``query_images``.
        :raises ValueError: If ``expansion_alpha`` is not a finite non-negative
            number or ``expansion_neighbors`` is not a positive integer.
        :raises RuntimeError: If the store has not been built yet.

        References:
        ===========
        [1] F. Radenović, G. Tolias, and O. Chum, "Fine-tuning CNN Image
            Retrieval with No Human Annotation," IEEE Transactions on Pattern
            Analysis and Machine Intelligence, vol. 41, no. 7, pp. 1655-1668,
            2019.
        """
        # ``embedder.embed`` returns one row per query image, in input order, so
        # the whole batch is searched at once: an index answers one ``(M, D)``
        # matrix far faster than a per-query loop.
        query_matrix = np.asarray(self._embedder.embed(query_images))
        if query_matrix.ndim == 1:
            query_matrix = query_matrix.reshape(1, -1)
        if query_matrix.shape[0] == 0:
            return []
        if query_expansion:
            query_matrix = self._expanded_queries(
                query_matrix, expansion_alpha, expansion_neighbors
            )

        scores, ids = self._search_index.search(query_matrix, k)
        return self._ranked_candidates(scores, ids)

    def _expanded_queries(
        self,
        queries: FloatNumpyArray,
        alpha: float,
        num_neighbors: int,
    ) -> Float32NumpyArray:
        """
        Replace every query embedding by its alpha-weighted query expansion.

        The whole batch is searched once, the embeddings of every query's
        ``num_neighbors`` best matches are read back from the index in a
        single lookup, and all expansions are formed from them by one batched
        call to :func:`_alpha_query_expansion`.

        :param queries: The ``(M, D)`` query embeddings.
        :param alpha: Exponent of the similarity weights.
        :param num_neighbors: Top-ranked gallery images averaged into each
            query.
        :return: The ``(M, D)`` expanded queries, each L2-normalised.
        """
        _, ids = self._search_index.search(queries, num_neighbors)
        # A gallery smaller than ``num_neighbors`` pads the free columns with
        # the id -1, which names no vector and is left out of the average. An
        # external index may even report no neighbor at all for a query, and
        # that query is then averaged with nothing, which leaves it as it is.
        found = ids >= 0
        neighbors = (
            self._search_index.vectors_at(ids[found])
            if found.any()
            else np.empty((0, queries.shape[1]), dtype=np.float32)
        )
        blocks = _neighbor_blocks(neighbors, found, queries.shape[1])
        return _alpha_query_expansion(queries, blocks, alpha)

    def _ranked_candidates(
        self,
        scores: Float32NumpyArray,
        ids: IntNumpyArray,
    ) -> list[list[Candidate]]:
        """
        Turn the result block of a search into one ranked list per query.

        :param scores: The ``(M, k)`` scores a search returned.
        :param ids: The ``(M, k)`` gallery row numbers a search returned, with
            ``-1`` marking a missing neighbor.
        :return: One ranked list of :class:`Candidate` matches per query.
        """
        return [
            [
                Candidate(self._paths[int(image_id)], float(score))
                for score, image_id in zip(row_scores, row_ids, strict=True)
                if image_id >= 0
            ]
            for row_scores, row_ids in zip(scores, ids, strict=True)
        ]

    @_requires_built_store
    def _state(self, embeddings: Float32NumpyArray | None = None) -> dict[str, Any]:
        """
        Describe the store around its gallery.

        The image paths, index configuration and the fully serialized embedder
        are described alongside the gallery, so the store can later be rebuilt
        without access to the original images.

        :param embeddings: Embeddings to write instead of the ones the index
            holds, shape ``(N, D)``, in the order of :attr:`paths`. Ignored by
            a store on an HNSW graph.
        :return: A JSON-safe store description.
        :raises TypeError: If the embedder is not serializable.
        :raises RuntimeError: If the store has not been built yet.
        """
        if not isinstance(self._embedder, SerializableImageEmbedder):
            raise TypeError(
                f"Embedder of type {type(self._embedder).__name__!r} is not "
                "serializable, it must be a SerializableImageEmbedder."
            )
        return {
            "index_name": self._index_name,
            "space": self._space,
            "index_params": self._index_params,
            "paths": list(self._paths),
            **self._gallery_state(embeddings),
            "embedder": self._embedder.to_dict(),
        }

    def _gallery_state(self, embeddings: Float32NumpyArray | None) -> dict[str, Any]:
        """
        Describe the gallery the index holds.

        :param embeddings: Embeddings to write instead of the ones the index
            holds. Ignored by a store on an HNSW graph, whose graph holds them.
        :return: The HNSW graph under ``"graph"``, or else the embeddings under
            ``"embeddings"``, with the other key set to ``None``.
        """
        if isinstance(self._index, HnswIndex):
            return {"embeddings": None, "graph": _encoded_graph(self._index._graph())}
        if embeddings is None:
            embeddings = self.embeddings
        return {"embeddings": _array_node(embeddings), "graph": None}

    @classmethod
    def from_dict(
        cls, state: dict[str, Any], **kwargs: Any
    ) -> InMemoryImageEmbeddingStore:
        """
        Rebuild a store from a dictionary produced by :meth:`to_dict`.

        The embedder is reconstructed. A saved HNSW graph is restored as it was
        saved, and any other index is rebuilt from the saved embeddings using
        the saved parameters.

        Not everything a store is made of survives serialization. An
        :class:`~pyvisim.retrieval.image_store.ExternalSearchIndex` wraps an object this
        library cannot write to disk, so pass a rebuilt one back as the
        ``search_index`` keyword argument; a name differing from the saved one
        is reported with a warning. Without it the store falls back to an exact
        :class:`~pyvisim.retrieval.image_store.BruteForceIndex` over the saved
        embeddings. Any other keyword argument is
        forwarded to the embedder, the way an embedder's own
        :meth:`~pyvisim.serialization.SerializerMixin.load_from_disk`
        forwards it.

        :param state: A JSON-safe store description.
        :param kwargs: ``search_index`` for a rebuilt external index, plus the
            objects the embedder's file cannot hold.
        :return: A populated :class:`InMemoryImageEmbeddingStore`.
        :raises TypeError: If ``search_index`` is not an
            :class:`~pyvisim.retrieval.image_store.ExternalSearchIndex`, or the embedder
            does not take one of ``kwargs``.
        :raises ValueError: If the index does not hold one vector per saved
            path, or an array of the saved HNSW graph has the wrong size.
        """
        search_index = kwargs.pop(_SEARCH_INDEX_KWARG, None)
        saved_name = str(state["index_name"])
        search_index = _restored_external_index(search_index, saved_name)
        embedder = SerializableImageEmbedder.from_dict(state["embedder"], **kwargs)
        index: _GalleryIndex
        if search_index is not None:
            index, index_name = search_index, search_index.name
        else:
            index_name = _BRUTE_FORCE if _is_external(saved_name) else saved_name
            index = _restored_index(state, index_name)
        return cls._from_components(
            paths=list(state["paths"]),
            embedder=embedder,
            index=index,
            index_name=index_name,
            space=state["space"],
            index_params=state["index_params"],
        )

    @_requires_built_store
    def save_to_disk(
        self,
        path: str | pathlib.Path,
        embeddings: FloatNumpyArray | None = None,
    ) -> pathlib.Path:
        """
        Persist the store to a single safetensors file.

        The image paths, index configuration and the fully serialized embedder
        are written together with the gallery, so the store can later be
        rebuilt without access to the original images. A store on an HNSW
        graph writes the graph itself, which holds the embeddings, so loading
        it does not build the graph again.

        The embeddings written are the ones the index holds, which are not
        always the ones it was given. A cosine index stores them L2-normalised,
        and a compressed external index (product- or scalar-quantized) hands
        back an approximation of them. Pass ``embeddings`` to write the originals
        instead.

        :param path: Destination file path. Overwritten if it exists.
        :param embeddings: Embeddings to write instead of the index's own, shape
            ``(N, D)``, in the order of :attr:`paths`. A store on an HNSW graph
            ignores them with a :class:`FutureWarning`.
        :return: The path of the written file.
        :raises OSError: If the destination directory does not exist.
        :raises TypeError: If the embedder is not serializable.
        :raises ValueError: If ``embeddings`` does not hold one row per path.
        :raises RuntimeError: If the store has not been built yet.
        """
        path = self._resolve_save_path(path)
        if embeddings is not None and isinstance(self._index, HnswIndex):
            # 'stacklevel=3' points past this method and the decorator that
            # wraps it, at the caller of 'save_to_disk'.
            warnings.warn(
                "'embeddings' has no effect on a store on an HNSW graph, whose "
                "file holds the graph with the embeddings in it.",
                FutureWarning,
                stacklevel=3,
            )
            embeddings = None
        if embeddings is not None:
            embeddings = _validated_embeddings(embeddings, len(self._paths))
        return self._write_state(self._stamp(self._state(embeddings)), path)


def _build_index(
    index_name: str,
    embeddings: Float32NumpyArray,
    space: Space,
    index_params: dict[str, Any],
) -> HnswIndex | BruteForceIndex:
    """
    Build the configured index over a gallery embedding matrix.

    :param index_name: Name of the index to build.
    :param embeddings: The ``(N, D)`` gallery embedding matrix.
    :param space: Metric space the index is built for.
    :param index_params: Keyword parameters forwarded to the index.
    :return: The built index.
    """
    if index_name == _HNSW:
        return HnswIndex(embeddings, space=space, **index_params)
    return BruteForceIndex(embeddings, space=space, **index_params)


def _unit_rows(matrix: FloatNumpyArray) -> Float64NumpyArray:
    """
    L2-normalise every row of a matrix, leaving all-zero rows as they are.

    :param matrix: A ``(N, D)`` matrix.
    :return: The matrix with unit-length rows, as float64.
    """
    rows = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    result: Float64NumpyArray = rows / norms
    return result


def _neighbor_blocks(
    neighbors: Float32NumpyArray,
    found: BoolNumpyArray,
    dim: int,
) -> Float32NumpyArray:
    """
    Lay the vectors a search found out as one block of neighbors per query.

    The empty slots of the search are filled with the zero vector, which
    resembles no query and therefore weighs nothing in the expansion.

    :param neighbors: The ``(N, D)`` vectors of the ids the search found, in
        the row-major order of the search result.
    :param found: The ``(M, k)`` mask marking the search slots that hold an id.
    :param dim: Dimensionality of the embeddings.
    :return: The ``(M, n, D)`` block of neighbors, one ``(n, D)`` set per
        query.
    """
    num_queries, num_slots = found.shape
    # Both built-in indexes answer every query of a batch with the same number
    # of neighbors, so the vectors already lie in the order the blocks need
    # and cutting them at that shared goal is a free reshape. Only an index
    # that leaves a hole in one query's row alone has to be scattered.
    goal = int(found[0].sum()) if num_queries else 0
    if found[:, :goal].all() and not found[:, goal:].any():
        return neighbors.reshape(num_queries, goal, dim)
    blocks = np.zeros((num_queries, num_slots, dim), dtype=neighbors.dtype)
    blocks[found] = neighbors
    return blocks


def _alpha_query_expansion(
    query: FloatNumpyArray,
    neighbors: FloatNumpyArray,
    alpha: float,
) -> Float32NumpyArray:
    """
    Form the alpha-weighted query expansion (αQE) of one query or of a batch.

    Implements Section 3.5 of Radenović et al. [1]: the new query is the
    weighted average of the query and its top-ranked gallery descriptors, where
    the weight of the i-th ranked descriptor is its inner product with the query
    raised to ``alpha``. A descriptor whose inner product is not positive
    weighs nothing, whatever ``alpha``. With ``alpha=0`` every remaining
    descriptor weighs the same and the expansion is the plain average query
    expansion (AQE).

    A single ``(D,)`` query is expanded against its own ``(n, D)`` neighbors,
    an ``(M, D)`` batch of queries against the ``(M, n, D)`` block holding the
    neighbors of each of them, in one pass over the whole block.

    :param query: The ``(D,)`` query embedding, or an ``(M, D)`` batch of them.
    :param neighbors: The ``(n, D)`` embeddings of the query's top-ranked
        gallery images, or the ``(M, n, D)`` block holding one such set per
        query of a batch. A query with fewer than ``n`` neighbors pads the
        free rows with the zero vector, which weighs nothing.
    :param alpha: Exponent of the similarity weights.
    :return: The expanded queries, L2-normalised and float32, shaped like
        ``query``.

    References:
    ===========
    [1] F. Radenović, G. Tolias, and O. Chum, "Fine-tuning CNN Image Retrieval
        with No Human Annotation," IEEE Transactions on Pattern Analysis and
        Machine Intelligence, vol. 41, no. 7, pp. 1655-1668, 2019.
    """
    batched = np.ndim(query) > 1
    unit_queries = _unit_rows(np.atleast_2d(query))
    blocks = np.asarray(neighbors)
    blocks = blocks if batched else blocks[None, ...]
    # The products run in the precision the neighbors were handed over in. The
    # float32 an index returns keeps the whole block out of the float64 copy
    # that otherwise dominates the cost of a batch.
    blocks = blocks.astype(np.promote_types(blocks.dtype, np.float32), copy=False)
    # The paper works on L2-normalised descriptors, whose inner product is the
    # cosine similarity. Dividing the two products by the row norms afterwards
    # makes that hold whatever form the store indexed the vectors in, without
    # ever forming a normalised copy of the block.
    norms = _block_row_norms(blocks)
    products = blocks @ unit_queries.astype(blocks.dtype, copy=False)[:, :, None]
    similarities = np.asarray(products[:, :, 0], dtype=np.float64) / norms
    # Weight of the i-th ranked image: (f(q)^T f(i))^alpha. Only a positive
    # similarity earns a weight: a dissimilar image contributes nothing, which
    # keeps the weight real for a non-integer alpha, never pushes the query
    # away along a direction unrelated to it, and leaves alpha=0 as the plain
    # average of the images that do resemble the query.
    positive = similarities > 0.0
    weights = np.zeros_like(similarities)
    weights[positive] = similarities[positive] ** alpha
    # The query joins the average with the weight of its own similarity,
    # (f(q)^T f(q))^alpha = 1. Dividing by the total weight would not change
    # the direction, so the sum goes straight to the normalisation.
    scaled = (weights / norms).astype(blocks.dtype, copy=False)
    expanded = unit_queries + (scaled[:, None, :] @ blocks)[:, 0, :]
    unit_expanded: Float32NumpyArray = np.asarray(
        _unit_rows(expanded), dtype=np.float32
    )
    return unit_expanded if batched else unit_expanded[0]


def _block_row_norms(blocks: FloatNumpyArray) -> Float64NumpyArray:
    """
    L2-norm every row of a block of neighbors, reporting a zero row as one.

    :param blocks: The ``(M, n, D)`` block of neighbors.
    :return: The ``(M, n)`` row norms, as float64.
    """
    norms: Float64NumpyArray = np.sqrt(np.einsum("mnd,mnd->mn", blocks, blocks)).astype(
        np.float64
    )
    norms[norms == 0.0] = 1.0
    return norms


def _validate_search_index(search_index: str | ExternalSearchIndex | None) -> None:
    """Reject an index selector the store cannot build."""
    if search_index is None or isinstance(search_index, ExternalSearchIndex):
        return
    if search_index == _HNSW:
        return
    raise ValueError(
        f"Unknown search_index {search_index!r}. Pass {_HNSW!r} for an HNSW "
        f"graph, None for an exact brute-force scan, or an ExternalSearchIndex."
    )


def _validate_index_params(index_name: str, index_params: dict[str, Any]) -> None:
    """
    Reject parameters the selected index does not take.

    :param index_name: Name of the index the store builds.
    :param index_params: The parameters the caller asked it to be built with.
    :raises ValueError: If a parameter is not one the index accepts.
    """
    table = _INDEX_PARAM_TABLES.get(index_name)
    if table is None:  # an external index brings its own parameters
        return
    validate_index_params(index_params, table, index_name)


def _aligned_index(index: _GalleryIndex, paths: list[str]) -> _GalleryIndex:
    """Check that an index lines up with the gallery paths."""
    indexed = getattr(index, "ntotal", len(index))
    if int(indexed) != len(paths):
        raise ValueError(
            f"The index holds {int(indexed)} vectors, but {len(paths)} image "
            f"paths were given. They must line up one to one, in the same order."
        )
    return index


def _restored_index(
    state: dict[str, Any], index_name: str
) -> HnswIndex | BruteForceIndex:
    """
    Restore the built-in index a saved store searched through.

    :param state: A JSON-safe store description.
    :param index_name: Name of the index to restore.
    :return: The saved HNSW graph as it was saved, or else an index built over
        the saved embeddings.
    :raises ValueError: If the saved parameters name one the index does not
        take, or an array of the saved graph has the wrong size.
    """
    index_params = state["index_params"]
    _validate_index_params(index_name, index_params)
    if state["graph"] is not None:
        return HnswIndex._from_graph(
            _decoded_graph(state["graph"]),
            num_threads=index_params.get("num_threads", -1),
        )
    embeddings = np.asarray(decode_array_node(state["embeddings"]), dtype=np.float32)
    return _build_index(index_name, embeddings, state["space"], index_params)


def _array_node(array: NumpyArray) -> dict[str, Any]:
    """Wrap an array into the node the serialization layer stores as a tensor."""
    return {
        "__ndarray__": True,
        "data": array,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "order": "C",
    }


def _encoded_graph(graph: dict[str, Any]) -> dict[str, Any]:
    """Wrap every array of an exported HNSW graph into an array node."""
    return {
        name: _array_node(value) if isinstance(value, np.ndarray) else value
        for name, value in graph.items()
    }


def _decoded_graph(graph: dict[str, Any]) -> dict[str, Any]:
    """Restore the arrays of a saved HNSW graph, as nodes or as file tensors."""
    return {
        name: decode_array_node(value) if isinstance(value, dict) else value
        for name, value in graph.items()
    }


def _is_external(index_name: str) -> bool:
    """Returns `True` if a saved index name refers to an external index, `False` otherwise."""
    return index_name not in (_HNSW, _BRUTE_FORCE)


def _restored_external_index(
    search_index: Any,
    saved_name: str,
) -> ExternalSearchIndex | None:
    """Validate the index handed to :meth:`load_from_disk` against the saved one."""
    # The warnings are reported at 'stacklevel=4' so that they point at the
    # caller of 'load_from_disk', three frames up: this function, 'from_dict'
    # and 'load_from_disk' itself.
    if search_index is None:
        if _is_external(saved_name):
            warnings.warn(
                f"This store was saved on the external index {saved_name!r}, which "
                f"its file cannot hold. Pass a rebuilt one as "
                f"'{_SEARCH_INDEX_KWARG}=...' to search through it again; falling "
                f"back to an exact brute-force scan of the saved embeddings.",
                FutureWarning,
                stacklevel=4,
            )
        return None
    if not isinstance(search_index, ExternalSearchIndex):
        raise TypeError(
            f"'{_SEARCH_INDEX_KWARG}' must be an ExternalSearchIndex, got "
            f"{type(search_index).__name__}."
        )
    if search_index.name != saved_name:
        warnings.warn(
            f"This store was saved on the index {saved_name!r}, but the one passed "
            f"as '{_SEARCH_INDEX_KWARG}' is named {search_index.name!r}. Its "
            f"results may not match the ones the saved store returned.",
            FutureWarning,
            stacklevel=4,
        )
    return search_index


def _validated_embeddings(
    embeddings: FloatNumpyArray,
    num_paths: int,
) -> Float32NumpyArray:
    """
    Raises `ValueError` if the provided gallery matrix does not hold
    one row per path.
    """
    matrix = np.ascontiguousarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != num_paths:
        raise ValueError(
            f"'embeddings' must be a ({num_paths}, dim) matrix holding one row per "
            f"gallery path, got shape {matrix.shape}."
        )
    return matrix


def _validated_paths(image_paths: Iterable[str]) -> list[str]:
    """
    Collect the gallery paths without reading the images.

    Duplicates are dropped, keeping the first occurrence.
    """
    paths: list[str] = []
    seen: set[str] = set()
    for path in image_paths:
        if not isinstance(path, str):
            raise TypeError(f"Image paths must be strings, got {type(path).__name__}.")
        if path in seen:
            continue
        seen.add(path)
        paths.append(path)
    if not paths:
        raise ValueError("No image paths were given; the store would be empty.")
    return paths


def _embed_image_paths_and_drop_duplicates(
    image_paths: Iterable[str],
    embedder: Embedder,
    skip_errors: bool,
    num_workers: int = _DEFAULT_NUM_WORKERS,
    num_prefetch_batches: int = _DEFAULT_NUM_PREFETCH_BATCHES,
) -> tuple[list[str], Float32NumpyArray]:
    """
    Embed every image path into a stacked embedding matrix.

    The images are decoded and handed to the embedder one batch at a time, so
    each batch costs a single :meth:`~pyvisim.typing.Embedder.embed` call and
    only one batch of decoded images is held in memory at once. Decoding runs
    on ``num_workers`` threads that read ahead of the embedder, which hides the
    file reads behind the embedder's own work.

    Duplicate paths are dropped (keeping the first occurrence) and their type is
    validated up front.

    :param image_paths: Iterable of image file paths to embed.
    :param embedder: Embedder turning each image into a feature vector.
    :param skip_errors: If ``True``, unreadable images are skipped with a
        warning instead of raising.
    :param num_workers: Threads decoding image files. ``1`` decodes on the
        calling thread.
    :param num_prefetch_batches: Batches of images the decoding threads may read
        ahead of the embedder.
    :return: A ``(paths, embeddings)`` pair with one embedding row per path.
    :raises TypeError: If any provided path is not a string.
    :raises ValueError: If no image could be embedded.
    """
    all_paths = _validated_paths(image_paths)
    batch_size = embedder.batch_size

    if batch_size == -1:
        batch_size = max(len(all_paths), 1)

    failures: list[str] = []
    images = _decoded_images(
        all_paths,
        num_workers,
        batch_size * num_prefetch_batches,
        skip_errors,
        failures,
    )

    paths: list[str] = []
    blocks: list[Float32NumpyArray] = []
    while batch := list(itertools.islice(images, batch_size)):
        batch_paths, batch_blocks = _embed_batch(batch, embedder, skip_errors, failures)
        paths.extend(batch_paths)
        blocks.extend(batch_blocks)

    if failures:
        warnings.warn(
            f"Skipped {len(failures)} image(s) that could not be embedded.",
            FutureWarning,
            stacklevel=3,
        )
    if not paths:
        raise ValueError("No images could be embedded; the store would be empty.")

    return paths, np.ascontiguousarray(np.vstack(blocks).astype(np.float32))


def _embed_batch(
    batch: list[tuple[str, UInt8NumpyArray]],
    embedder: Embedder,
    skip_errors: bool,
    failures: list[str],
) -> tuple[list[str], list[Float32NumpyArray]]:
    """
    Embed one batch of decoded images in a single call.

    A batch the embedder rejects is retried one image at a time, so a single
    unembeddable image costs only itself rather than the whole batch.

    :param batch: The ``(path, image)`` pairs making up the batch.
    :param embedder: Embedder turning the images into feature vectors.
    :param skip_errors: If ``True``, images the embedder rejects are recorded
        in ``failures`` instead of raising.
    :param failures: Collects the paths that could not be embedded.
    :return: The embedded paths and their embedding blocks.
    :raises ValueError: If the embedder rejects an image, or does not return one
        row per image, and ``skip_errors`` is off.
    """
    paths = [path for path, _ in batch]
    images = [image for _, image in batch]
    try:
        return paths, [_embed_images(images, embedder)]
    except (ValueError, OSError):
        if not skip_errors:
            raise
    return _embed_one_by_one(batch, embedder, failures)


def _embed_one_by_one(
    batch: list[tuple[str, UInt8NumpyArray]],
    embedder: Embedder,
    failures: list[str],
) -> tuple[list[str], list[Float32NumpyArray]]:
    """
    Embed a rejected batch image by image to isolate the ones at fault.

    :param batch: The ``(path, image)`` pairs making up the batch.
    :param embedder: Embedder turning the images into feature vectors.
    :param failures: Collects the paths that could not be embedded.
    :return: The embedded paths and their embedding blocks.
    """
    paths: list[str] = []
    blocks: list[Float32NumpyArray] = []
    for path, image in batch:
        try:
            blocks.append(_embed_images([image], embedder))
        except (ValueError, OSError):
            failures.append(path)
            continue
        paths.append(path)
    return paths, blocks


def _embed_images(
    images: list[UInt8NumpyArray],
    embedder: Embedder,
) -> Float32NumpyArray:
    """
    Embed a list of decoded images into one embedding block.

    :param images: Canonical RGB images to embed.
    :param embedder: Embedder turning the images into feature vectors.
    :return: A ``(len(images), dim)`` block of embeddings.
    :raises ValueError: If the embedder does not return one row per image.
    """
    embeddings = np.asarray(embedder.embed(images), dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    if embeddings.shape[0] != len(images):
        raise ValueError(
            f"The embedder returned {embeddings.shape[0]} embeddings for "
            f"{len(images)} image(s); it must return exactly one per image."
        )
    return embeddings


def _decoded_images(
    paths: list[str],
    num_workers: int,
    prefetch: int,
    skip_errors: bool,
    failures: list[str],
) -> Iterator[tuple[str, UInt8NumpyArray]]:
    """
    Yield the decoded gallery images paired with their path, in input order.

    :param paths: Gallery image paths to decode.
    :param num_workers: Threads decoding image files.
    :param prefetch: Images the decoding threads may read ahead.
    :param skip_errors: If ``True``, unreadable images are recorded in
        ``failures`` instead of raising.
    :param failures: Collects the paths that could not be read.
    :return: An iterator over ``(path, image)`` pairs.
    :raises FileNotFoundError: If an image is missing and ``skip_errors`` is off.
    :raises ValueError: If an image cannot be decoded and ``skip_errors`` is off.
    """
    for path, decode in _decode_stream(paths, num_workers, prefetch):
        try:
            image = decode()
        except (FileNotFoundError, ValueError, OSError):
            if not skip_errors:
                raise
            failures.append(path)
            continue
        yield path, image


def _decode_stream(
    paths: list[str],
    num_workers: int,
    prefetch: int,
) -> Iterator[tuple[str, Callable[[], UInt8NumpyArray]]]:
    """Yield each path with a callable returning its decoded image.

    Deferring the decode to a callable lets a threaded and an unthreaded
    producer share one error-handling site in :func:`_decoded_images`.

    The threads stop at the decoded image and the callable copies it into an
    array on the consuming thread. The decoder itself releases the GIL and so
    runs in parallel, while the copy holds it for its whole duration: leaving
    the copy in the threads would serialize the decodes behind it."""
    if num_workers == 1:
        for path in paths:
            yield path, functools.partial(_decode_image_array, path)
        return

    remaining = iter(paths)
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        pending = deque(
            (path, pool.submit(_decode_image, path))
            for path in itertools.islice(remaining, max(prefetch, num_workers))
        )
        while pending:
            path, decoded = pending.popleft()
            for next_path in itertools.islice(remaining, 1):
                pending.append((next_path, pool.submit(_decode_image, next_path)))
            yield path, functools.partial(_awaited_array, decoded.result)


def _decode_image(path: str) -> Image.Image:
    """Read one image file into a decoded RGB image."""
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except FileNotFoundError:
        raise  # already clear and specific; let it propagate
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Could not read image {path!r}: {exc}") from exc


def _decode_image_array(path: str) -> UInt8NumpyArray:
    """Read one image file into a canonical RGB array."""
    return np.asarray(_decode_image(path))


def _awaited_array(decoded: Callable[[], Image.Image]) -> UInt8NumpyArray:
    """Wait for a threaded decode and copy its image into a canonical RGB array."""
    return np.asarray(decoded())
