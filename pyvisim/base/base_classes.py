import abc
import contextlib
import inspect
from collections.abc import Iterator, Sequence
from typing import Any, ClassVar, cast

import numpy as np

from .._utils import get_similarity_func
from ..lazy_import import is_tensor
from ..serialization import SerializerMixin
from ..typing import (
    Float32NumpyArray,
    Float64NumpyArray,
    FloatNumpyArray,
    ImageInput,
    IntNumpyArray,
    MatLike,
    SimilarityFunc,
    UInt8NumpyArray,
)
from ..utils.image_utils import grayscale_dims, iter_image_batches, iter_images


def _l2_normalize(vectors: FloatNumpyArray) -> FloatNumpyArray:
    """
    Scales every row of ``vectors`` to unit L2 length.

    A row of length zero carries no direction to preserve and is left as it is
    instead of being divided by zero.

    :param vectors: An ``(N, D)`` array of embeddings.
    :return: An ``(N, D)`` array whose non-zero rows have unit L2 norm.
    """
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return cast(
        FloatNumpyArray,
        np.divide(vectors, norms, out=vectors.copy(), where=norms > 0),
    )


class SimilarityMetric(abc.ABC):
    """
    Abstract base for all similarity metrics.

    All concrete similarity metric classes must inherit from this class.

    Every metric processes its input in batches. What one batch holds depends
    on the metric (images, image pairs, ...).

    Setting ``batch_size=-1`` would treat the whole input as a single batch.

    :param batch_size: Maximum number of images processed in a single batch.
        Set to ``-1`` to process all images as a single batch.
    :raises ValueError: If ``batch_size`` is neither ``-1`` nor a positive
        integer.
    """

    def __init__(self, batch_size: int = 16) -> None:
        self._batch_size: int
        # Assign via the property setter to trigger validation.
        self.batch_size = batch_size

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int) -> None:
        self._batch_size = self._validate_batch_size(batch_size)

    def set_batch_size(self, batch_size: int) -> None:
        """
        Sets the number of items processed per batch.

        :param batch_size: Maximum number of images processed in a single
            batch. Set to ``-1`` to process all images as a single batch.
        :raises ValueError: If ``batch_size`` is neither ``-1`` nor a positive
            integer.
        """
        self.batch_size = batch_size

    @staticmethod
    def _validate_batch_size(batch_size: int) -> int:
        """
        Raises ValueError if ``batch_size`` is neither ``-1`` nor a positive integer.
        """
        if not isinstance(batch_size, int):
            raise ValueError(
                f"batch_size must be an integer, got {type(batch_size).__name__}."
            )
        if batch_size != -1 and batch_size < 1:
            raise ValueError(
                "batch_size must be a positive integer or -1 (process the "
                f"whole input as one batch), got {batch_size}."
            )
        return batch_size

    @abc.abstractmethod
    def similarity_score(
        self,
        images1: ImageInput,
        images2: ImageInput,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> FloatNumpyArray:
        """
        Compute the similarity scores matrix between two (batches of) images.

        :param images1: First (batch of) image(s) as ``MatLike`` (NumPy array,
            torch tensor or array-like).
        :param images2: Second (batch of) image(s) as ``MatLike``.
        :param dims: Axis-label string, one character per array axis in order:
            ``"H"`` = height (rows), ``"W"`` = width (columns), ``"C"`` = channels
            (e.g. RGB), ``"B"`` = batch size. For example, ``"HWC"`` is height ×
            width × channels (NumPy/OpenCV single-image layout, **default**);
            ``"CHW"`` is channels × height × width (PyTorch single-image layout);
            ``"BCHW"`` is batch × channels × height × width (PyTorch batched layout).
            See :mod:`pyvisim.typing`.
        :param value_range: The ``(low, high)`` range the input values live in;
            converted into the canonical ``[0, 255]`` range.
        :return: The similarity score matrix of shape ``(len(images1), len(images2))``.
        """
        pass


class FeatureExtractorBase(abc.ABC):
    """
    Abstract interface for extracting features from images.

    A feature extractor transforms an image (NumPy array) into a
    set of feature vectors (NumPy array).
    """

    @abc.abstractmethod
    def __call__(
        self,
        image: MatLike,
        /,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> Float32NumpyArray:
        """
        Extracts features from an image.

        :param image: Input image as ``MatLike`` (NumPy array, torch tensor or
            array-like). It is normalized to a canonical ``uint8`` ``(H, W, C)``
            image before extraction.
        :param dims: Axis-label string, one character per array axis in order:
            ``"H"`` = height (rows), ``"W"`` = width (columns), ``"C"`` = channels.
            For example, ``"HWC"`` is height × width × channels (NumPy/OpenCV
            layout, **default**); ``"CHW"`` is channels × height × width (PyTorch
            layout). See :mod:`pyvisim.typing`.
        :param value_range: The ``(low, high)`` range the input values live in;
            converted into the canonical ``[0, 255]`` range.
        :return: Feature descriptors (NumPy array).
        """
        pass

    def extract_batch(
        self,
        images: Sequence[MatLike],
        /,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> list[Float32NumpyArray]:
        """
        Extracts features from a batch of images.

        Returns one ``(N_i, D)`` feature array per image, in input order, since
        the number of descriptors an image yields varies from image to image.
        This default implementation extracts one image at a time; extractors
        that can do the whole batch in one go (e.g. a single forward pass
        through a neural network) override it.

        :param images: Batch of images, each a ``MatLike`` (NumPy array, torch
            tensor or array-like) normalized to a canonical ``uint8``
            ``(H, W, C)`` image before extraction.
        :param dims: Axis-label string, one character per array axis in order:
            ``"H"`` = height (rows), ``"W"`` = width (columns), ``"C"`` = channels.
            It applies to every image of the batch. See :mod:`pyvisim.typing`.
        :param value_range: The ``(low, high)`` range the input values live in;
            converted into the canonical ``[0, 255]`` range.
        :return: One ``(N_i, D)`` feature array per input image.
        """
        return [self(image, dims=dims, value_range=value_range) for image in images]

    @property
    @abc.abstractmethod
    def output_dim(self) -> int:
        """
        The dimensionality (D) of each feature vector, i.e., shape[1] of the output.
        """
        pass

    def to_dict(self) -> dict[str, Any]:
        """
        Serialise this feature extractor into a JSON-safe configuration dict.

        The dict captures the extractor's class name and the keyword arguments
        needed to rebuild an equivalent instance (see
        :func:`pyvisim.features.feature_extractor_from_dict`).

        :return: A mapping ``{"__class__": str, "config": dict}``.
        """
        return {
            "__class__": type(self).__name__,
            "config": self._serialization_config(),
        }

    def _serialization_config(self) -> dict[str, Any]:
        """
        Return the JSON-safe constructor arguments needed to rebuild this extractor.

        Extractors without constructor arguments return an empty mapping.
        Subclasses override this hook when they carry reconstructable
        parameters.

        :return: A JSON-safe mapping of constructor arguments.
        """
        return {}


class ImageEmbedderBase(SimilarityMetric):
    """
    Base class for all image embedders.

    An image embedder turns an image into a vector representation that can be
    used for indexing, retrieval, clustering or classification.

    :param similarity_func: Name of the built-in similarity metric to use. One of
        ``"cosine"`` (default), ``"euclidean"``, ``"l1"`` or ``"manhattan"``.
    :param normalize: Whether :meth:`embed` L2-normalizes the embeddings it
        returns, so that they can be compared directly with a dot product.
    :param batch_size: Maximum number of images processed in a single batch.
        Set to ``-1`` to process all images as a single batch.
    :raises ValueError: If ``similarity_func`` is not a supported similarity
        metric, ``normalize`` is not a boolean, or ``batch_size`` is neither
        ``-1`` nor a positive integer.
    """

    def __init__(
        self,
        similarity_func: str = "cosine",
        *,
        normalize: bool = True,
        batch_size: int = 16,
    ):
        # Set important attributes via setters to trigger error handling
        super().__init__(batch_size=batch_size)
        self._similarity_func: SimilarityFunc
        self._similarity_func_name: str
        self._normalize: bool
        self.similarity_func = similarity_func
        self.normalize = normalize

    @property
    def normalize(self) -> bool:
        """Whether the embeddings returned by :meth:`embed` are L2-normalized."""
        return self._normalize

    @normalize.setter
    def normalize(self, normalize: bool) -> None:
        """Sets whether :meth:`embed` L2-normalizes the embeddings it returns."""
        if not isinstance(normalize, bool):
            raise ValueError(
                f"normalize must be a boolean, got {type(normalize).__name__}."
            )
        self._normalize = normalize

    @property
    def similarity_func(self) -> SimilarityFunc:
        """The resolved similarity function callable."""
        return self._similarity_func

    @similarity_func.setter
    def similarity_func(self, name: str) -> None:
        """
        Resolves and stores a built-in similarity metric by name.

        :param name: One of ``"cosine"``, ``"euclidean"``, ``"l1"`` or
            ``"manhattan"``.
        :raises ValueError: If ``name`` is not a supported similarity metric.
        """
        self._similarity_func = get_similarity_func(name)
        self._similarity_func_name = name

    @property
    def similarity_func_name(self) -> str:
        """The name of the configured similarity metric (e.g. ``"cosine"``)."""
        return self._similarity_func_name

    def embed(
        self,
        images: ImageInput,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> FloatNumpyArray:
        """
        Embeds one or more images into a batch of vector representations.

        Each image is normalized to a canonical ``uint8`` ``(H, W, C)`` array
        before feature extraction, so NumPy arrays, torch tensors and other
        array-like inputs are all accepted. When a batch axis is present (via
        ``dims``), every image in the batch is embedded. The resulting vectors
        are L2-normalized row by row when :attr:`normalize` is True.

        :param images: A single ``MatLike`` image, a batched array, or an
            iterable of images. Consider using an iterator for large datasets.
        :param dims: Axis-label string, one character per array axis in order:
            ``"H"`` = height (rows), ``"W"`` = width (columns), ``"C"`` = channels
            (e.g. RGB), ``"B"`` = batch size. For example, ``"HWC"`` is height ×
            width × channels (NumPy/OpenCV single-image layout, **default**);
            ``"CHW"`` is channels × height × width (PyTorch single-image layout);
            ``"BCHW"`` is batch × channels × height × width (PyTorch batched layout).
            See :mod:`pyvisim.typing`.
        :param value_range: The ``(low, high)`` range the input values live in;
            converted into the canonical ``[0, 255]`` range.
        :return: vector representations of the given images, L2-normalized
            row by row if :attr:`normalize` is True.
        :raises ValueError: If ``images`` holds no image.
        """
        embeddings = [
            self._embed(batch)
            for batch in iter_image_batches(
                images, self.batch_size, dims=dims, value_range=value_range
            )
        ]
        if not embeddings:
            raise ValueError("Expected at least one image, got none.")
        vectors = np.vstack(embeddings)
        return _l2_normalize(vectors) if self._normalize else vectors

    @abc.abstractmethod
    def _embed(self, images: list[UInt8NumpyArray]) -> FloatNumpyArray:
        """
        Embeds one batch of images, without the L2 normalization.

        Every subclass has to implement this method

        :param images: One batch of at most :attr:`batch_size` canonical
            ``uint8`` images of shape ``(H, W[, C])``.
        :return: vector representations of the given images without L2 normalization.
        """
        raise NotImplementedError

    def similarity_score(
        self,
        images1: ImageInput,
        images2: ImageInput,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> Float32NumpyArray:
        vector1 = self.embed(images1, dims=dims, value_range=value_range)
        vector2 = self.embed(images2, dims=dims, value_range=value_range)
        result = self.similarity_func(vector1, vector2)
        return np.asarray(result, dtype=np.float32)

    def __repr__(self) -> str:
        return (
            self.__class__.__name__ + f"(similarity_func={self.similarity_func_name})"
        )


class SerializableImageEmbedder(ImageEmbedderBase, SerializerMixin):
    """
    Base for embedders that persist to a ``.embedder`` file.

    Adds the serialization contract of
    :class:`~pyvisim.serialization.SerializerMixin` on top of
    :class:`ImageEmbedderBase`: subclasses describe themselves as a JSON-safe
    state via :meth:`~pyvisim.serialization.SerializerMixin._state` /
    :meth:`~pyvisim.serialization.SerializerMixin.from_dict`, and the mixin
    turns that state into a file and back. Both the classic embedders and the
    neural ones use this path, so a ``.embedder`` file is always a
    `safetensors <https://github.com/huggingface/safetensors>`_ file.

    :param similarity_func: Name of the built-in similarity metric to use. One of
        ``"cosine"`` (default), ``"euclidean"``, ``"l1"`` or ``"manhattan"``.
    :param normalize: Whether :meth:`embed` L2-normalizes the embeddings it
        returns.
    :param batch_size: Maximum number of images processed in a single batch.
        Set to ``-1`` to process all images as a single batch.
    """

    #: Suffix of the files written by :meth:`save_to_disk`.
    __file_format__: ClassVar[str] = ".embedder"
    #: Metadata key under which the embedder JSON skeleton is stored.
    __metadata_key__: ClassVar[str] = "pyvisim_embedder"
    __class_key__: ClassVar[str] = "embedder_class"

    #: Keys a serialised state must contain to be a valid embedder file.
    #: Subclasses extend this with their own required keys.
    __state_keys__: ClassVar[frozenset[str]] = frozenset(
        {"similarity_func", "normalize", "batch_size"}
    )

    #: Every subclass defined so far, keyed by class name.
    _subclasses_by_name: ClassVar[dict[str, type["SerializableImageEmbedder"]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Registers a subclass under its name for :meth:`from_dict`."""
        super().__init_subclass__(**kwargs)
        SerializableImageEmbedder._subclasses_by_name[cls.__name__] = cls

    @classmethod
    def from_dict(
        cls, state: dict[str, Any], **kwargs: Any
    ) -> "SerializableImageEmbedder":
        """
        Rebuilds the embedder a state dictionary describes.

        Called on :class:`SerializableImageEmbedder` itself, it hands the state
        to the ``from_dict`` of the class named under :attr:`__class_key__`, so
        a state can be rebuilt without knowing which embedder wrote it.

        :param state: A JSON-safe embedder description.
        :param kwargs: Objects the state cannot describe, forwarded to the
            embedder's own ``from_dict``.
        :return: The reconstructed embedder.
        :raises ValueError: If ``state`` names no concrete embedder class.
        :raises NotImplementedError: If called on a subclass that does not
            implement its own ``from_dict``.
        """
        if cls is not SerializableImageEmbedder:
            raise NotImplementedError(f"{cls.__name__} does not implement from_dict.")
        embedder_cls = cls._subclass_named(state.get(cls.__class_key__))
        return embedder_cls.from_dict(state, **kwargs)

    @classmethod
    def _subclass_named(cls, name: Any) -> type["SerializableImageEmbedder"]:
        """
        Looks up a concrete embedder class by name.

        The shipped embedders register themselves when their package is
        imported, so both packages are imported before a name is reported as
        unknown. The neural embedders need the ``nn`` extra and are skipped
        without it.

        :param name: The class name recorded in a state.
        :return: The embedder class of that name.
        :raises ValueError: If no concrete embedder class has that name.
        """
        if name not in cls._subclasses_by_name:
            from .. import classic  # noqa: F401

            with contextlib.suppress(ImportError):
                from .. import neural_networks  # noqa: F401
        embedder_cls = cls._subclasses_by_name.get(name)
        if embedder_cls is None or inspect.isabstract(embedder_cls):
            known = sorted(
                known_name
                for known_name, known_cls in cls._subclasses_by_name.items()
                if not inspect.isabstract(known_cls)
            )
            raise ValueError(
                f"Cannot reconstruct embedder of class {name!r}. "
                f"Known classes are: {known}."
            )
        return embedder_cls


def _stack_image_batch(
    images: ImageInput,
    dims: str,
    value_range: tuple[float, float],
) -> Float64NumpyArray:
    """
    Normalize ``images`` and stack them into one ``(N, H, W, C)`` float batch.

    Every image is first converted to the canonical ``uint8`` ``(H, W[, C])``
    layout in ``[0, 255]`` (see :mod:`pyvisim.typing`), then the batch is
    stacked and cast to ``float64`` for numerically stable arithmetic.
    Grayscale images receive a singleton channel axis.

    :param images: A single ``MatLike`` image, a batched array, or an iterable
        of images.
    :param dims: Axis-label string describing the input axes (see
        :mod:`pyvisim.typing`).
    :param value_range: The ``(low, high)`` range the input values live in.
    :return: A ``(N, H, W, C)`` ``float64`` array with values in ``[0, 255]``.
    :raises InvalidImageError: If an input cannot be converted to a numeric
        array.
    :raises ValueError: If no image is given or the images differ in shape.
    """
    if isinstance(images, np.ndarray) or is_tensor(images):
        # A single channel-less array (e.g. a 2-D grayscale image with the
        # default "HWC") keeps working, like elsewhere in the library.
        dims = grayscale_dims(images, dims)
    canonical = list(iter_images(images, dims=dims, value_range=value_range))
    if not canonical:
        raise ValueError("Expected at least one image, got none.")
    shapes = {image.shape for image in canonical}
    if len(shapes) > 1:
        raise ValueError(
            "All images in a batch must have the same shape to be compared "
            f"pixel-wise, got shapes {sorted(shapes)}."
        )
    batch = np.stack(canonical).astype(np.float64)
    if batch.ndim == 3:
        batch = batch[..., np.newaxis]
    return batch


def _iter_pair_chunks(
    n_rows: int, n_cols: int, batch_size: int
) -> Iterator[tuple[IntNumpyArray, IntNumpyArray]]:
    """
    Yield ``(rows, cols)`` index arrays covering an ``n_rows x n_cols`` grid.

    Pairs are enumerated in row-major order and grouped into chunks of at most
    ``batch_size`` pairs; ``-1`` yields every pair in a single chunk.

    :param n_rows: Number of images in the first batch.
    :param n_cols: Number of images in the second batch.
    :param batch_size: Maximum number of image pairs processed in a single
        batch. Set to ``-1`` to process all images as a single batch.
    :return: An iterator of ``(rows, cols)`` integer index arrays.
    """
    n_pairs = n_rows * n_cols
    chunk = n_pairs if batch_size == -1 else batch_size
    for start in range(0, n_pairs, chunk):
        flat = np.arange(start, min(start + chunk, n_pairs), dtype=np.intp)
        yield flat // n_cols, flat % n_cols


class DenseMetricBase(SimilarityMetric, abc.ABC):
    """
    Base class for metrics that score two aligned pixel grids directly.

    Concrete subclasses implement ``_score_pairs``, which receives two stacked
    ``float64`` batches of identical shape and returns one score per pair.

    :param batch_size: Maximum number of image pairs processed in a single
        batch. Set to ``-1`` to process all images as a single batch.
    :raises ValueError: If ``batch_size`` is neither ``-1`` nor positive.
    """

    def similarity_score(
        self,
        image1: ImageInput,
        image2: ImageInput,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> FloatNumpyArray:
        """
        Compute the pairwise score matrix between two image batches.

        Every image is normalized to the canonical ``uint8`` ``(H, W[, C])``
        layout in ``[0, 255]`` first, so the metric always operates on the
        same value scale regardless of the input dtype or range. The pairs are
        scored in chunks of at most :attr:`batch_size` pairs.

        :param image1: First (batch of) image(s) as ``MatLike`` (NumPy array,
            torch tensor or array-like).
        :param image2: Second (batch of) image(s) as ``MatLike``.
        :param dims: Axis-label string, one character per array axis in order:
            ``"H"`` = height (rows), ``"W"`` = width (columns), ``"C"`` = channels
            (e.g. RGB), ``"B"`` = batch size. For example, ``"HWC"`` is height ×
            width × channels (NumPy/OpenCV single-image layout, **default**);
            ``"CHW"`` is channels × height × width (PyTorch single-image layout);
            ``"BCHW"`` is batch × channels × height × width (PyTorch batched layout).
            See :mod:`pyvisim.typing`.
        :param value_range: The ``(low, high)`` range the input values live in;
            converted into the canonical ``[0, 255]`` range.
        :return: A ``(N, M)`` matrix scoring every image of ``image1`` against
            every image of ``image2``.
        :raises InvalidImageError: If an input cannot be converted to a numeric
            array.
        :raises ValueError: If a batch is empty, the two batches hold images of
            different shapes, or the images are too small for the metric.
        """
        batch1 = _stack_image_batch(image1, dims, value_range)
        batch2 = _stack_image_batch(image2, dims, value_range)
        if batch1.shape[1:] != batch2.shape[1:]:
            raise ValueError(
                "image1 and image2 must contain images of the same shape, "
                f"got {batch1.shape[1:]} vs {batch2.shape[1:]}."
            )
        self._validate_image_shape(batch1.shape[1], batch1.shape[2])
        scores = np.empty((batch1.shape[0], batch2.shape[0]), dtype=np.float64)
        for rows, cols in _iter_pair_chunks(
            batch1.shape[0], batch2.shape[0], self._batch_size
        ):
            scores[rows, cols] = self._score_pairs(batch1[rows], batch2[cols])
        return scores

    def _validate_image_shape(self, height: int, width: int) -> None:
        """
        Hook for subclasses to reject images too small for the metric.

        The default accepts any size.

        :param height: Height of the images, in pixels.
        :param width: Width of the images, in pixels.
        :raises ValueError: If the images cannot be scored by this metric.
        """

    @abc.abstractmethod
    def _score_pairs(
        self, images1: Float64NumpyArray, images2: Float64NumpyArray
    ) -> Float64NumpyArray:
        """
        Score aligned image pairs.

        :param images1: ``(B, H, W, C)`` ``float64`` batch, one image per pair.
        :param images2: ``(B, H, W, C)`` ``float64`` batch, aligned with
            ``images1``.
        :return: A ``(B,)`` array holding one score per pair.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(batch_size={self.batch_size})"
