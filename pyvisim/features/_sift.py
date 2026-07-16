from typing import Any

import numpy as np
from PIL import Image

from .._base_classes import FeatureExtractorBase
from ..typing import Float32NumpyArray, MatLike, UInt8NumpyArray
from ._utils import _check_output_shape, _to_single_image
from ._vendored.sift.sift import SIFT as _SIFT

__all__ = ["SIFT"]


class SIFT(FeatureExtractorBase, _SIFT):
    """
    Scale-Invariant Feature Transform (SIFT) feature extractor.

    References:
    ===========
    [1] Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints.
    """

    def __init__(
        self,
        upsampling: int = 2,
        n_octaves: int = 8,
        n_scales: int = 3,
        sigma_min: float = 1.6,
        sigma_in: float = 0.5,
        c_dog: float = 0.04 / 3,
        c_edge: float = 10,
        n_bins: int = 36,
        lambda_ori: float = 1.5,
        c_max: float = 0.8,
        lambda_descr: float = 6,
        n_hist: int = 4,
        n_ori: int = 8,
    ) -> None:
        params: dict[str, Any] = {
            "upsampling": upsampling,
            "n_octaves": n_octaves,
            "n_scales": n_scales,
            "sigma_min": sigma_min,
            "sigma_in": sigma_in,
            "c_dog": c_dog,
            "c_edge": c_edge,
            "n_bins": n_bins,
            "lambda_ori": lambda_ori,
            "c_max": c_max,
            "lambda_descr": lambda_descr,
            "n_hist": n_hist,
            "n_ori": n_ori,
        }
        _SIFT.__init__(self, **params)  # type: ignore[no-untyped-call]
        self._output_dim = n_hist**2 * n_ori

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @staticmethod
    def _to_grayscale(image: UInt8NumpyArray) -> UInt8NumpyArray:
        """
        Collapse a canonical ``uint8`` image to the 2-D grayscale layout.

        :param image: A ``uint8`` array of shape ``(H, W)`` or ``(H, W, C)``.
        :return: A ``uint8`` array of shape ``(H, W)``.
        """
        if image.ndim == 2:
            return image
        return np.asarray(Image.fromarray(image).convert("L"))

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
        Extracts SIFT features from an image.

        :param image: Input image as ``MatLike``.
        :param dims: Axis-label string, one character per array axis in order:
            ``"H"`` = height (rows), ``"W"`` = width (columns), ``"C"`` = channels.
            For example, ``"HWC"`` is height × width × channels (NumPy
            layout, **default**); ``"CHW"`` is channels × height × width (PyTorch
            layout). See :mod:`pyvisim.typing`.
        :param value_range: The ``(low, high)`` range the input values live in.
        :return: ``(num_keypoints, output_dim)`` array of SIFT descriptors.
        """
        canonical = _to_single_image(image, dims=dims, value_range=value_range)
        grayscale = self._to_grayscale(canonical)
        try:
            self.detect_and_extract(grayscale)  # type: ignore[no-untyped-call]
        except RuntimeError:
            # The vendored detector raises when an image yields no keypoints;
            # the extractor contract is an empty descriptor matrix instead.
            return np.zeros((0, self.output_dim), dtype=np.float32)
        return np.asarray(self.descriptors, dtype=np.float32)

    def __repr__(self) -> str:
        return f"SIFT(output_dim={self.output_dim})"
