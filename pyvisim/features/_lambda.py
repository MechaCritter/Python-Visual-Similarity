from collections.abc import Callable
from typing import Any

from .._base_classes import FeatureExtractorBase
from ..typing import Float32NumpyArray, MatLike, UInt8NumpyArray
from ._utils import _check_output_shape, _to_single_image


class Lambda(FeatureExtractorBase):
    """
    Lambda feature extractor that allows passing any user-defined
    function to extract features from images.

    The function must accept a single argument (image as NumPy array),
    and output fixed-size feature vectors from each image.
    """

    def __init__(
        self, func: Callable[[UInt8NumpyArray], Float32NumpyArray], output_dim: int
    ):
        """
        Initializes the Lambda feature extractor.
        :param func:
        :param output_dim:
        """
        super().__init__()
        if not callable(func):
            raise ValueError(
                f"Argument func must be a callable object, got {type(func)} instead"
            )
        self._output_dim = output_dim
        self.func = func

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def _serialization_config(self) -> dict[str, Any]:
        """
        Lambda extractors wrap an arbitrary user function and cannot be
        serialised.

        :raises TypeError: Always; pass ``feature_extractor`` explicitly when
            loading an encoder that used a Lambda extractor.
        """
        raise TypeError(
            "Lambda feature extractors wrap a user-defined function and cannot "
            "be serialised. Provide 'feature_extractor' explicitly when loading."
        )

    @_check_output_shape
    def __call__(
        self,
        image: MatLike,
        /,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> Float32NumpyArray:
        """
        Extracts features from an image using the user-defined function.

        :param image: Input image as ``MatLike``.
        :param dims: Axis-label string, one character per array axis in order:
            ``"H"`` = height (rows), ``"W"`` = width (columns), ``"C"`` = channels.
            For example, ``"HWC"`` is height × width × channels (NumPy/OpenCV
            layout, **default**); ``"CHW"`` is channels × height × width (PyTorch
            layout). See :mod:`pyvisim.typing`.
        :param value_range: The ``(low, high)`` range the input values live in.
        :return: Feature descriptors returned by ``func``.
        """
        image = _to_single_image(image, dims=dims, value_range=value_range)
        return self.func(image)
