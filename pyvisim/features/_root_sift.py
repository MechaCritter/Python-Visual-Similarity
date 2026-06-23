import cv2
import numpy as np

from .._base_classes import FeatureExtractorBase
from ..typing import Float32NumpyArray, MatLike
from ._utils import _check_output_shape, _to_single_image


class RootSIFT(FeatureExtractorBase):
    """
    Scale-Invariant Feature Transform with Hellinger kernel (RootSIFT) normalizer.

    References:
    ===========
    [1] Arandjelovic, R., & Zisserman, A. (2012). Three things everyone should know to improve object retrieval.
    """

    def __init__(self) -> None:
        super().__init__()
        self._output_dim = 128

    @property
    def output_dim(self) -> int:
        return self._output_dim

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
        Extracts RootSIFT features from an image.

        :param image: Input image as ``MatLike``.
        :param dims: Axis-label string, one character per array axis in order:
            ``"H"`` = height (rows), ``"W"`` = width (columns), ``"C"`` = channels.
            For example, ``"HWC"`` is height × width × channels (NumPy/OpenCV
            layout, **default**); ``"CHW"`` is channels × height × width (PyTorch
            layout). See :mod:`pyvisim.typing`.
        :param value_range: The ``(low, high)`` range the input values live in.
        :return: ``(num_keypoints, 128)`` array of RootSIFT descriptors.
        """
        image = _to_single_image(image, dims=dims, value_range=value_range)
        sift = cv2.SIFT.create()
        _, descriptors = sift.detectAndCompute(image, None)
        if descriptors is not None:
            descriptors = descriptors / (descriptors.sum(axis=1, keepdims=True) + 1e-7)
            return np.asarray(np.sqrt(descriptors))
        return descriptors

    def __repr__(self) -> str:
        return f"RootSIFT(output_dim={self.output_dim})"
