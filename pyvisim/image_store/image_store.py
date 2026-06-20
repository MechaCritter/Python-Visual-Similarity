"""
In-memory image embedding storage that uses index search to accelerate retrieval.
"""

from __future__ import annotations

import os
import pathlib
import warnings
from collections.abc import Iterable
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from ..encoders._reconstruct import encoder_from_dict, encoder_to_dict
from ..encoders._serialization import load_state, save_state
from ..functional import Candidate, retrieve_top_k_similar
from ..retrieval.index import (
    ImageIndex,
    ImageIndexHNSW,
    ImageIndexIVFFlat,
    ImageIndexIVFPQ,
    ImageIndexScalarQuantizer,
)
from ..retrieval.index._base_index import Quantizer
from ..typing import Encoder, Float32NumpyArray, ImageInput, IntNumpyArray

#: Index-type strings mapped to the index class that implements them.
_INDEX_REGISTRY: dict[str, type[ImageIndex]] = {
    "ivf-flat": ImageIndexIVFFlat,
    "ivf-pq": ImageIndexIVFPQ,
    "hnsw": ImageIndexHNSW,
    "int8": ImageIndexScalarQuantizer,
}

#: On-disk format version, bumped if the safetensors layout ever changes.
_STORE_FORMAT_VERSION = 1
#: Metadata key under which the store's JSON skeleton is stored on disk.
_STORE_METADATA_KEY = "pyvisim_store"
#: File suffix appended to a save path when it is missing.
_STORE_FILE_SUFFIX = ".safetensors"


class InMemoryImageEmbeddingStore:
    """
    Encode a gallery of images and index their embeddings for fast retrieval.

    Each image path is read and encoded with ``encoder``; the resulting vectors
    are stacked into an in-memory matrix and indexed with the structure selected
    by ``index_type``.

    :param image_paths: Iterable of image file paths to encode. Duplicates are
        dropped, keeping the first occurrence.
    :param encoder: Encoder used to turn images into feature vectors. To make the
        store serialisable it must implement ``to_dict`` (every
        :class:`~pyvisim.encoders.ImageEncoderBase` subclass and
        :class:`~pyvisim.encoders.Pipeline` does).
    :param index_type: Index structure to build, one of ``"ivf-flat"``,
        ``"ivf-pq"``, ``"hnsw"`` or ``"int8"``.
    :param quantizer: Distance metric the index is built for, ``"l2"`` or
        ``"inner_product"`` (see :class:`~pyvisim.retrieval.ImageIndex`).
    :param index_params: Optional keyword parameters forwarded to the index
        constructor (e.g. ``{"nlist": 100, "nprobe": 8}``).
    :param skip_errors: If ``True``, images that cannot be read or encoded are
        skipped with a warning instead of aborting.
    :raises ValueError: If ``index_type`` is unknown or no image could be
        encoded.
    :raises TypeError: If any provided path is not a string.
    :raises NotImplementedError: If ``index_type`` selects an index that is not
        yet implemented (``"hnsw"`` or ``"int8"``).
    """

    def __init__(
        self,
        image_paths: Iterable[str],
        encoder: Encoder,
        index_type: str = "ivf-flat",
        *,
        quantizer: Quantizer = "l2",
        index_params: dict[str, Any] | None = None,
        skip_errors: bool = False,
    ) -> None:
        if index_type not in _INDEX_REGISTRY:
            raise ValueError(
                f"Unknown index_type {index_type!r}. Supported index types are: "
                f"{sorted(_INDEX_REGISTRY)}."
            )

        self._encoder = encoder
        self._index_type = index_type
        self._quantizer: Quantizer = quantizer
        self._index_params: dict[str, Any] = dict(index_params or {})

        paths, embeddings = _encode_image_paths(image_paths, encoder, skip_errors)
        self._paths = paths
        # Only the index retains the gallery vectors; the local ``embeddings``
        # matrix is released once the index has copied it in.
        self._index = self._build_index(embeddings)

    @classmethod
    def _from_components(
        cls,
        paths: list[str],
        embeddings: Float32NumpyArray,
        encoder: Encoder,
        index_type: str,
        quantizer: Quantizer,
        index_params: dict[str, Any],
    ) -> InMemoryImageEmbeddingStore:
        """
        Rebuild a store from already-computed components without re-encoding.

        :param paths: Gallery image paths, ordered to match ``embeddings``.
        :param embeddings: Gallery embedding matrix, shape ``(N, D)``.
        :param encoder: The reconstructed encoder.
        :param index_type: Index structure to rebuild.
        :param quantizer: Distance metric the index is built for.
        :param index_params: Keyword parameters forwarded to the index.
        :return: A populated :class:`InMemoryImageEmbeddingStore`.
        """
        store = cls.__new__(cls)
        store._encoder = encoder
        store._index_type = index_type
        store._quantizer = quantizer
        store._index_params = dict(index_params)
        store._paths = list(paths)
        store._index = store._build_index(
            np.ascontiguousarray(embeddings, dtype=np.float32)
        )
        return store

    def _build_index(self, embeddings: Float32NumpyArray) -> ImageIndex:
        """
        Build the configured index over the gallery embeddings.

        :param embeddings: The ``(N, D)`` gallery matrix to index.
        :return: A trained :class:`~pyvisim.retrieval.ImageIndex`.
        """
        index_cls = _INDEX_REGISTRY[self._index_type]
        return index_cls(
            self._paths,
            embeddings,
            quantizer=self._quantizer,
            **self._index_params,
        )

    @property
    def paths(self) -> list[str]:
        """Gallery image paths, ordered to match the embedding rows."""
        return list(self._paths)

    @property
    def embeddings(self) -> Float32NumpyArray:
        """
        The ``(N, D)`` gallery embedding matrix, reconstructed from the index.

        The vectors are read back from the index on each access, so no separate
        copy is held in memory. For an ``"inner_product"`` store they come back
        L2-normalised, and for a lossy index (``"ivf-pq"``) they are the
        decompressed approximation.
        """
        return self._index.reconstruct()

    @property
    def encoder(self) -> Encoder:
        """The encoder used to build the gallery and to encode queries."""
        return self._encoder

    @property
    def index(self) -> ImageIndex:
        """The accelerated :class:`~pyvisim.retrieval.ImageIndex`."""
        return self._index

    @property
    def index_type(self) -> str:
        """The index-structure string the store was built with."""
        return self._index_type

    @property
    def quantizer(self) -> str:
        """The distance metric the index was built for."""
        return self._quantizer

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
            f"index_type={self._index_type!r}, quantizer={self._quantizer!r})"
        )

    def search(
        self,
        query_vectors: Float32NumpyArray,
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

        The query images are encoded with this store's encoder and matched
        against the gallery through its accelerated index.

        :param query_images: A single image or a batch/iterable of images to use
            as queries. Anything accepted by the store's encoder is valid.
        :param k: Number of top similar gallery images to return per query.
        :return: One ranked list of :class:`~pyvisim.functional.Candidate`
            matches per query image, in the same order as ``query_images``.
        """
        return retrieve_top_k_similar(query_images, self, k)

    def save_to_disk(self, path: str | pathlib.Path) -> pathlib.Path:
        """
        Persist the store to a single ``.safetensors`` file.

        The embeddings, image paths, index configuration and the fully
        serialised encoder are written together, so the store can later be
        rebuilt without access to the original images.

        :param path: Destination file path. The ``.safetensors`` suffix is
            appended if missing. Overwritten if it exists.
        :return: The path of the written file.
        :raises OSError: If the destination directory does not exist.
        :raises TypeError: If the encoder is not serialisable.
        """
        path = pathlib.Path(path)
        if path.suffix != _STORE_FILE_SUFFIX:
            path = path.with_name(path.name + _STORE_FILE_SUFFIX)
        parent = os.path.dirname(os.path.abspath(path))
        if not os.path.isdir(parent):
            raise OSError(f"Destination directory does not exist: {parent!r}.")

        embeddings = self.embeddings
        state: dict[str, Any] = {
            "format_version": _STORE_FORMAT_VERSION,
            "store_class": type(self).__name__,
            "index_type": self._index_type,
            "quantizer": self._quantizer,
            "index_params": self._index_params,
            "paths": list(self._paths),
            "embeddings": {
                "__ndarray__": True,
                "data": embeddings,
                "dtype": str(embeddings.dtype),
                "shape": list(embeddings.shape),
                "order": "C",
            },
            "encoder": encoder_to_dict(self._encoder),
        }
        save_state(state, path, _STORE_METADATA_KEY)
        return path

    @classmethod
    def load_from_disk(cls, path: str | pathlib.Path) -> InMemoryImageEmbeddingStore:
        """
        Rebuild a store from a :meth:`save_to_disk` file.

        The encoder is reconstructed and the index is re-trained from the saved
        embeddings using the saved parameters; no image is re-encoded.

        :param path: Path to a ``.safetensors`` file written by
            :meth:`save_to_disk`.
        :return: A populated :class:`InMemoryImageEmbeddingStore`.
        :raises FileNotFoundError: If ``path`` does not exist.
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

        embeddings = np.asarray(state["embeddings"], dtype=np.float32)
        encoder = encoder_from_dict(state["encoder"])
        return cls._from_components(
            paths=list(state["paths"]),
            embeddings=embeddings,
            encoder=encoder,
            index_type=state["index_type"],
            quantizer=state["quantizer"],
            index_params=state["index_params"],
        )


def _encode_image_paths(
    image_paths: Iterable[str],
    encoder: Encoder,
    skip_errors: bool,
) -> tuple[list[str], Float32NumpyArray]:
    """
    Encode every image path into a stacked embedding matrix.

    Duplicate paths are dropped (keeping the first occurrence) and their type is
    validated up front.

    :param image_paths: Iterable of image file paths to encode.
    :param encoder: Encoder turning each image into a feature vector.
    :param skip_errors: If ``True``, unreadable images are skipped with a
        warning instead of raising.
    :return: A ``(paths, embeddings)`` pair with one embedding row per path.
    :raises TypeError: If any provided path is not a string.
    :raises ValueError: If no image could be encoded.
    """
    paths: list[str] = []
    encodings: list[Float32NumpyArray] = []
    seen: set[str] = set()
    failures: list[str] = []
    for path in image_paths:
        if not isinstance(path, str):
            raise TypeError(f"Image paths must be strings, got {type(path).__name__}.")
        if path in seen:
            continue
        seen.add(path)
        try:
            encodings.append(_encode_single_path(path, encoder))
        except (FileNotFoundError, ValueError, OSError):
            if not skip_errors:
                raise
            failures.append(path)
            continue
        paths.append(path)

    if failures:
        warnings.warn(
            f"Skipped {len(failures)} image(s) that could not be encoded.",
            FutureWarning,
            stacklevel=3,
        )
    if not paths:
        raise ValueError("No images could be encoded; the store would be empty.")

    embeddings = np.ascontiguousarray(np.stack(encodings).astype(np.float32))
    return paths, embeddings


def _encode_single_path(path: str, encoder: Encoder) -> Float32NumpyArray:
    """
    Open one image and return its flattened embedding.

    :param path: Filesystem path of the image to encode.
    :param encoder: Encoder turning the image into a feature vector.
    :return: The flattened embedding vector for the image.
    :raises FileNotFoundError: If the image file is missing.
    :raises ValueError: If the image cannot be decoded.
    """
    try:
        with Image.open(path) as image:
            rgb_image = np.asarray(image.convert("RGB"))
    except FileNotFoundError:
        raise  # already clear and specific; let it propagate
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Could not read image {path!r}: {exc}") from exc
    return np.asarray(encoder.encode(rgb_image), dtype=np.float32).flatten()
