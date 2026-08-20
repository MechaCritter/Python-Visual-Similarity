import abc
import logging
from typing import Any

import numpy as np

from ._utils import get_similarity_func
from .typing import (
    Float32NumpyArray,
    FloatNumpyArray,
    ImageInput,
    MatLike,
    SimilarityFunc,
)


class SimilarityMetric(abc.ABC):
    """
    Abstract base for all similarity metrics.

    All concrete similarity metric classes must inherit from this class.
    """

    _logger = logging.getLogger("Similarity_Metrics")

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

    _logger = logging.getLogger("Feature_Extractor")

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

        Stateless extractors (e.g. SIFT, RootSIFT) need no configuration and
        return an empty mapping. Subclasses override this hook when they carry
        reconstructable parameters.

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
    """

    def __init__(self, similarity_func: str = "cosine"):
        # Set important attributes via setters to trigger error handling
        self._similarity_func: SimilarityFunc
        self._similarity_func_name: str
        self.similarity_func = similarity_func

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

    @abc.abstractmethod
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
        ``dims``), every image in the batch is embedded.

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
        :return: vector representations of the given images
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
