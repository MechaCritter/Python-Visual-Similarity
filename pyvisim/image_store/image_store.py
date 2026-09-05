"""
In-memory image embedding storage that uses a search index to accelerate retrieval.
"""

from __future__ import annotations

import os
import pathlib
import warnings
from collections.abc import Iterable
from typing import Any, NamedTuple

import numpy as np
from PIL import Image, UnidentifiedImageError

from ..serialization import (
    embedder_from_dict,
    embedder_to_dict,
    load_state,
    save_state,
)
from ..typing import (
    Embedder,
    Float32NumpyArray,
    FloatNumpyArray,
    ImageInput,
    IntNumpyArray,
    SearchIndex,
)
from ._index import (
    BRUTE_FORCE_TO_HNSWLIB,
    HNSW_TO_HNSWLIB,
    BruteForceIndex,
    ExternalSearchIndex,
    HnswIndex,
    Space,
    validate_index_params,
)

#: Value of ``search_index`` selecting the HNSW graph.
_HNSW = "hnsw"
#: Name under which a brute-force store records its index.
_BRUTE_FORCE = "brute-force"

#: Parameter table of each index the store can build, keyed by index name. It
#: names the parameters the index takes and maps them onto the backend's own.
_INDEX_PARAM_TABLES: dict[str, dict[str, str]] = {
    _HNSW: HNSW_TO_HNSWLIB,
    _BRUTE_FORCE: BRUTE_FORCE_TO_HNSWLIB,
}

#: On-disk format version, bumped if the safetensors layout ever changes.
_STORE_FORMAT_VERSION = 2
#: Metadata key under which the store's JSON skeleton is stored on disk.
_STORE_METADATA_KEY = "pyvisim_store"
#: File suffix appended to a save path when it is missing.
_STORE_FILE_SUFFIX = ".safetensors"
#: Keyword argument of :meth:`InMemoryImageEmbeddingStore.load_from_disk` that
#: carries a rebuilt index instead of being forwarded to the embedder.
_SEARCH_INDEX_KWARG = "search_index"


class Candidate(NamedTuple):
    """A single retrieval result.

    :param path: Path of the matched gallery image.
    :param score: Score the index ranked the match by, best first. For the
        built-in indexes it is a distance, so lower means more similar.
    """

    path: str
    score: float


class InMemoryImageEmbeddingStore:
    """
    Embed a gallery of images and index their embeddings for fast retrieval.

    Each image path is read and embedded with ``embedder``; the resulting
    vectors are handed to the index selected by ``search_index``, which owns
    them from then on. The store keeps no second copy, so
    :attr:`embeddings` reads them back out of the index.

    A store built on an :class:`~pyvisim.image_store.ExternalSearchIndex`
    embeds nothing: that index already holds its gallery, so ``image_paths``
    only names its rows and must be exactly as long as the index is.

    .. important::

        Upon constructing the store, all images are embedded immediately and the
        index is built. If ``hnsw`` is used, the memory consumption is higher
        than when only using ``brute-force``, since the ``hnsw`` index builds an
        additional graph structure based on the embeddings.

    :param image_paths: Iterable of image file paths to embed. Duplicates are
        dropped, keeping the first occurrence.
    :param embedder: Embedder used to turn images into feature vectors. This
        can be any object that implements the ``embed`` method, including
        ``Siamese`` and ``Triplet`` networks, the ``ClipEmbedder`` and the
        ``VLAD``/``Fisher Vector`` embedders.
    :param search_index: The index to search the gallery through. Pass
        ``"hnsw"`` for the HNSW graph algorithm, ``None`` for brute-force
    :param space: Metric space the index is built for, ``"cosine"`` (the
        default), ``"l2"`` or ``"ip"``. Ignored by an external index, which
        brings its own metric. An overview below:

        .. list-table::
           :header-rows: 1
           :widths: 15 30 55

           * - ``space``
             - Score
             - Notes
           * - ``"cosine"``
             - ``1 - cosine_similarity``
             - The default. The vectors are stored L2-normalised, so their
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
    :raises ValueError: If ``search_index`` is unknown, ``index_params`` names a
        parameter the index does not take, no image could be embedded, or an
        external index does not hold one vector per path.
    :raises TypeError: If any provided path is not a string.

    .. _store-index-parameters:

    Index parameters
    ----------------

    The parameters are named after what they do rather than after the library
    implementing the index, so they stay the same if the backend behind an index
    ever changes.

    ``HnswIndex`` parameters
    ~~~~~~~~~~~~~~~~~~~~~~~~

    .. list-table::
       :header-rows: 1
       :widths: 20 15 65

       * - Parameter
         - Default
         - Meaning
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
           ``search_candidates`` neighbours raises it to ``k``.
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
         - Meaning
       * - ``num_threads``
         - ``-1``
         - Threads used to run batched queries. ``-1`` uses every available
           core.

    Example
    -------

    >>> from pyvisim.neural_networks import ClipEmbedder
    >>> from pyvisim.image_store import InMemoryImageEmbeddingStore
    >>>
    >>> embedder = ClipEmbedder("ViT-B/32")
    >>> store = InMemoryImageEmbeddingStore(
    ...     image_paths=["image1.jpg", "image2.jpg", "image3.jpg"],
    ...     embedder=embedder,
    ...     search_index="hnsw",
    ...     space="cosine",
    ...     index_params={"graph_degree": 32, "search_candidates": 100},
    ... )
    >>> # Save the store to disk and load it back later
    >>> store.save_to_disk("gallery.safetensors")
    >>>
    >>> # Retrieve top-k most similar images to a query image
    >>> query_image = np.array(Image.open("query.jpg"))
    >>> results = store.retrieve_top_k_similar(query_image, k=5)[0]
    >>>
    >>> for candidate in results:
    ...     print(candidate.path, candidate.score)
    """

    def __init__(
        self,
        image_paths: Iterable[str],
        embedder: Embedder,
        search_index: str | ExternalSearchIndex | None = None,
        *,
        space: Space = "cosine",
        index_params: dict[str, Any] | None = None,
        skip_errors: bool = False,
    ) -> None:
        _validate_search_index(search_index)

        self._embedder = embedder
        self._space: Space = space
        self._index_params: dict[str, Any] = dict(index_params or {})

        if isinstance(search_index, ExternalSearchIndex):
            self._paths = _validated_paths(image_paths)
            self._index: SearchIndex = _adopt_external_index(search_index, self._paths)
            self._index_name = search_index.name
            return

        self._index_name = _HNSW if search_index == _HNSW else _BRUTE_FORCE
        # Checked before a single image is read: a misspelled parameter should
        # not cost the caller a full embedding pass first.
        _validate_index_params(self._index_name, self._index_params)

        paths, embeddings = _embed_image_paths(image_paths, embedder, skip_errors)
        self._paths = paths
        # Only the index retains the gallery vectors; the local ``embeddings``
        # matrix is released once the index has copied it in.
        self._index = self._build_index(embeddings)

    @classmethod
    def _from_components(
        cls,
        paths: list[str],
        embeddings: Float32NumpyArray,
        embedder: Embedder,
        index_name: str,
        space: Space,
        index_params: dict[str, Any],
        search_index: ExternalSearchIndex | None = None,
    ) -> InMemoryImageEmbeddingStore:
        """
        Rebuild a store from already-computed components without re-embedding.

        :param paths: Gallery image paths, ordered to match ``embeddings``.
        :param embeddings: Gallery embedding matrix, shape ``(N, D)``.
        :param embedder: The reconstructed embedder.
        :param index_name: Name of the index the store was saved with.
        :param space: Metric space the index is built for.
        :param index_params: Keyword parameters forwarded to the index.
        :param search_index: A ready-made index to adopt instead of building
            one over ``embeddings``.
        :return: A populated :class:`InMemoryImageEmbeddingStore`.
        :raises ValueError: If ``search_index`` does not hold one vector per
            path, or ``index_params`` names a parameter the index does not take.
        """
        store = cls.__new__(cls)
        store._embedder = embedder
        store._space = space
        store._index_params = dict(index_params)
        store._paths = list(paths)
        if search_index is not None:
            store._index = _adopt_external_index(search_index, store._paths)
            store._index_name = search_index.name
            return store

        store._index_name = index_name
        _validate_index_params(index_name, store._index_params)
        store._index = store._build_index(
            np.ascontiguousarray(embeddings, dtype=np.float32)
        )
        return store

    def _build_index(self, embeddings: Float32NumpyArray) -> SearchIndex:
        """Build the configured index over the gallery embeddings."""
        if self._index_name == _HNSW:
            return HnswIndex(embeddings, space=self._space, **self._index_params)
        return BruteForceIndex(embeddings, space=self._space, **self._index_params)

    @property
    def paths(self) -> list[str]:
        """Gallery image paths, ordered to match the embedding rows."""
        return list(self._paths)

    @property
    def embeddings(self) -> Float32NumpyArray:
        """The ``(N, D)`` gallery embedding matrix, read back from the index."""
        return self._index.vectors

    @property
    def embedder(self) -> Embedder:
        """The embedder used to build the gallery and to embed queries."""
        return self._embedder

    @property
    def index(self) -> SearchIndex:
        """The search index the gallery is searched through."""
        return self._index

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
    def dim(self) -> int:
        """Dimensionality of the gallery embeddings."""
        return self._index.dim

    def __len__(self) -> int:
        return len(self._paths)

    def __contains__(self, path: object) -> bool:
        return path in self._paths

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_images={len(self)}, dim={self.dim}, "
            f"index_name={self._index_name!r}, space={self._space!r})"
        )

    def search(
        self,
        query_vectors: FloatNumpyArray,
        k: int,
    ) -> tuple[Float32NumpyArray, IntNumpyArray]:
        """
        Return the ``k`` nearest gallery vectors for each query vector.

        :param query_vectors: A ``(D,)`` vector or a ``(N, D)`` batch of query
            feature vectors.
        :param k: Number of nearest neighbours to return per query.
        :return: A ``(scores, ids)`` tuple of ``(N, k)`` arrays; ``ids`` index
            into :attr:`paths` and missing neighbours are reported as ``-1``.
        """
        return self._index.search(query_vectors, k)

    def retrieve_top_k_similar(
        self,
        query_images: ImageInput,
        k: int = 5,
    ) -> list[list[Candidate]]:
        """
        Return the top-k most similar gallery images for each query image.

        The query images are embedded with this store's embedder and matched
        against the gallery through its index.

        :param query_images: A single image or a batch/iterable of images to use
            as queries. Anything accepted by the store's embedder is valid.
        :param k: Number of top similar gallery images to return per query.
        :return: One ranked list of :class:`Candidate` matches per query image,
            in the same order as ``query_images``.
        """
        # ``embedder.embed`` returns one row per query image, in input order, so
        # the whole batch is searched at once: an index answers one ``(M, D)``
        # matrix far faster than a per-query loop.
        query_matrix = np.asarray(self._embedder.embed(query_images))
        if query_matrix.ndim == 1:
            query_matrix = query_matrix.reshape(1, -1)
        if query_matrix.shape[0] == 0:
            return []

        scores, ids = self.search(query_matrix, k)
        return [
            [
                Candidate(self._paths[int(image_id)], float(score))
                for score, image_id in zip(row_scores, row_ids, strict=True)
                if image_id >= 0
            ]
            for row_scores, row_ids in zip(scores, ids, strict=True)
        ]

    def save_to_disk(
        self,
        path: str | pathlib.Path,
        vectors: FloatNumpyArray | None = None,
    ) -> pathlib.Path:
        """
        Persist the store to a single ``.safetensors`` file.

        The embeddings, image paths, index configuration and the fully
        serialised embedder are written together, so the store can later be
        rebuilt without access to the original images.

        The embeddings written are the ones the index holds, which are not
        always the ones it was given. A cosine index stores them L2-normalised,
        and a compressed external index (product- or scalar-quantized) hands
        back an approximation of them. Pass ``vectors`` to write the originals
        instead.

        :param path: Destination file path. The ``.safetensors`` suffix is
            appended if missing. Overwritten if it exists.
        :param vectors: Embeddings to write instead of the index's own, shape
            ``(N, D)``, in the order of :attr:`paths`.
        :return: The path of the written file.
        :raises OSError: If the destination directory does not exist.
        :raises TypeError: If the embedder is not serialisable.
        :raises ValueError: If ``vectors`` does not hold one row per path.
        """
        path = pathlib.Path(path)
        if path.suffix != _STORE_FILE_SUFFIX:
            path = path.with_name(path.name + _STORE_FILE_SUFFIX)
        parent = os.path.dirname(os.path.abspath(path))
        if not os.path.isdir(parent):
            raise OSError(f"Destination directory does not exist: {parent!r}.")

        embeddings = (
            self.embeddings
            if vectors is None
            else _validated_vectors(vectors, len(self._paths))
        )
        state: dict[str, Any] = {
            "format_version": _STORE_FORMAT_VERSION,
            "store_class": type(self).__name__,
            "index_name": self._index_name,
            "space": self._space,
            "index_params": self._index_params,
            "paths": list(self._paths),
            "embeddings": {
                "__ndarray__": True,
                "data": embeddings,
                "dtype": str(embeddings.dtype),
                "shape": list(embeddings.shape),
                "order": "C",
            },
            "embedder": embedder_to_dict(self._embedder),
        }
        save_state(state, path, _STORE_METADATA_KEY)
        return path

    @classmethod
    def load_from_disk(
        cls,
        path: str | pathlib.Path,
        **kwargs: Any,
    ) -> InMemoryImageEmbeddingStore:
        """
        Rebuild a store from a :meth:`save_to_disk` file.

        The embedder is reconstructed and the index is rebuilt from the saved
        embeddings using the saved parameters.

        Not everything a store is made of survives serialization. An
        :class:`~pyvisim.image_store.ExternalSearchIndex` wraps an object this
        library cannot write to disk, so pass a rebuilt one back as the
        ``search_index`` keyword argument; a name differing from the saved one
        is reported with a warning. Without it the store falls back to an exact
        :class:`~pyvisim.image_store.BruteForceIndex` over the saved
        embeddings, also with a warning. Any other keyword argument is
        forwarded to the embedder, the way an embedder's own
        :meth:`~pyvisim._base_classes.SerializableImageEmbedder.load_from_disk`
        forwards it.

        :param path: Path to a ``.safetensors`` file written by
            :meth:`save_to_disk`.
        :param kwargs: ``search_index`` for a rebuilt external index, plus the
            objects the embedder's file cannot hold.
        :return: A populated :class:`InMemoryImageEmbeddingStore`.
        :raises FileNotFoundError: If ``path`` does not exist.
        :raises TypeError: If ``search_index`` is not an
            :class:`~pyvisim.image_store.ExternalSearchIndex`, or the embedder
            does not take one of ``kwargs``.
        :raises ValueError: If the file was not written by this class.
        """
        path = pathlib.Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No such store file: {str(path)!r}.")

        state = load_state(path, _STORE_METADATA_KEY)
        if state.get("store_class") != cls.__name__:
            raise ValueError(
                f"File {str(path)!r} was not written by {cls.__name__}.save_to_disk."
            )

        search_index = kwargs.pop(_SEARCH_INDEX_KWARG, None)
        index_name = str(state["index_name"])
        search_index = _restored_external_index(search_index, index_name)
        embedder = embedder_from_dict(state["embedder"], **kwargs)
        return cls._from_components(
            paths=list(state["paths"]),
            embeddings=np.asarray(state["embeddings"], dtype=np.float32),
            embedder=embedder,
            index_name=_BRUTE_FORCE
            if search_index is None and _is_external(index_name)
            else index_name,
            space=state["space"],
            index_params=state["index_params"],
            search_index=search_index,
        )


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


def _adopt_external_index(
    search_index: ExternalSearchIndex,
    paths: list[str],
) -> ExternalSearchIndex:
    """Check that an external index lines up with the gallery paths."""
    indexed = getattr(search_index, "ntotal", len(search_index))
    if int(indexed) != len(paths):
        raise ValueError(
            f"The index holds {int(indexed)} vectors, but {len(paths)} image "
            f"paths were given. They must line up one to one, in the same order."
        )
    return search_index


def _is_external(index_name: str) -> bool:
    """Returns `True` if a saved index name refers to an external index, `False` otherwise."""
    return index_name not in (_HNSW, _BRUTE_FORCE)


def _restored_external_index(
    search_index: Any,
    saved_name: str,
) -> ExternalSearchIndex | None:
    """Validate the index handed to :meth:`load_from_disk` against thggfsssssfffssssddddddgrsh gn"""
    if search_index is None:
        if _is_external(saved_name):
            warnings.warn(
                f"This store was saved on the external index {saved_name!r}, which "
                f"its file cannot hold. Pass a rebuilt one as "
                f"'{_SEARCH_INDEX_KWARG}=...' to search through it again; falling "
                f"back to an exact brute-force scan of the saved embeddings.",
                FutureWarning,
                stacklevel=3,
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
            stacklevel=3,
        )
    return search_index


def _validated_vectors(
    vectors: FloatNumpyArray,
    num_paths: int,
) -> Float32NumpyArray:
    """
    Raises `ValueError` if the provided gallery matrix does not hold
    one row per path.
    """
    matrix = np.ascontiguousarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != num_paths:
        raise ValueError(
            f"'vectors' must be a ({num_paths}, dim) matrix holding one row per "
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


def _embed_image_paths(
    image_paths: Iterable[str],
    embedder: Embedder,
    skip_errors: bool,
) -> tuple[list[str], Float32NumpyArray]:
    """
    Embed every image path into a stacked embedding matrix.

    Duplicate paths are dropped (keeping the first occurrence) and their type is
    validated up front.

    :param image_paths: Iterable of image file paths to embed.
    :param embedder: Embedder turning each image into a feature vector.
    :param skip_errors: If ``True``, unreadable images are skipped with a
        warning instead of raising.
    :return: A ``(paths, embeddings)`` pair with one embedding row per path.
    :raises TypeError: If any provided path is not a string.
    :raises ValueError: If no image could be embedded.
    """
    # TODO: take batch size attr from the embedder & embed in batches instead of one by one
    # (if hasattr(embedder, "batch_size"): batch_size = embedder.batch_size)
    paths: list[str] = []
    vectors: list[Float32NumpyArray] = []
    failures: list[str] = []
    for path in _validated_paths(image_paths):
        try:
            vectors.append(_embed_single_path(path, embedder))
        except (FileNotFoundError, ValueError, OSError):
            if not skip_errors:
                raise
            failures.append(path)
            continue
        paths.append(path)

    if failures:
        warnings.warn(
            f"Skipped {len(failures)} image(s) that could not be embedded.",
            FutureWarning,
            stacklevel=3,
        )
    if not paths:
        raise ValueError("No images could be embedded; the store would be empty.")

    embeddings = np.ascontiguousarray(np.stack(vectors).astype(np.float32))
    return paths, embeddings


def _embed_single_path(path: str, embedder: Embedder) -> Float32NumpyArray:
    """Open one image and return its flattened embedding."""
    try:
        with Image.open(path) as image:
            rgb_image = np.asarray(image.convert("RGB"))
    except FileNotFoundError:
        raise  # already clear and specific; let it propagate
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Could not read image {path!r}: {exc}") from exc
    return np.asarray(embedder.embed(rgb_image), dtype=np.float32).flatten()
