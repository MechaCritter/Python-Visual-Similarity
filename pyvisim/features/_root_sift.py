import numpy as np

from ..typing import Float32NumpyArray, MatLike
from ._sift import SIFT
from ._utils import _check_output_shape

__all__ = ["RootSIFT"]


class RootSIFT(SIFT):
    """
    Scale-Invariant Feature Transform with Hellinger kernel (RootSIFT) normalizer.

    References:
    ===========
    [1] Arandjelovic, R., & Zisserman, A. (2012). Three things everyone should know to improve object retrieval.
    """

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
            For example, ``"HWC"`` is height × width × channels (NumPy
            layout, **default**); ``"CHW"`` is channels × height × width (PyTorch
            layout). See :mod:`pyvisim.typing`.
        :param value_range: The ``(low, high)`` range the input values live in.
        :return: ``(num_keypoints, output_dim)`` array of RootSIFT descriptors.
        """
        descriptors = super().__call__(image, dims=dims, value_range=value_range)
        descriptors = descriptors / (descriptors.sum(axis=1, keepdims=True) + 1e-7)
        return np.asarray(np.sqrt(descriptors), dtype=np.float32)

    def __repr__(self) -> str:
        return f"RootSIFT(output_dim={self.output_dim})"
