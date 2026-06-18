import abc
import logging
from typing import Any

from .typing import Float32NumpyArray, FloatNumpyArray, ImageInput, MatLike


class SimilarityMetric(abc.ABC):
    """
    Abstract base for all similarity encoders.

    All concrete similarity metric classes must inherit from this class.
    """

    _logger = logging.getLogger("Similarity_Metrics")

    @abc.abstractmethod
    def similarity_score(
        self,
        image1: ImageInput,
        image2: ImageInput,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> FloatNumpyArray:
        """
        Compute a similarity score between two images.

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
        :return: A similarity score matrix
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
