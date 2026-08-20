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
        descriptors = super().__call__(image, dims=dims, value_range=value_range)
        descriptors = descriptors / (descriptors.sum(axis=1, keepdims=True) + 1e-7)
        return np.asarray(np.sqrt(descriptors), dtype=np.float32)

    def __repr__(self) -> str:
        return f"RootSIFT(output_dim={self.output_dim})"
