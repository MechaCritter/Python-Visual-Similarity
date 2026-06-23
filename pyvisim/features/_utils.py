from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

import numpy as np

from .._base_classes import FeatureExtractorBase
from ..typing import (
    Float32NumpyArray,
    MatLike,
    UInt8NumpyArray,
    _to_image_list,
)

ExtractorCallT = TypeVar("ExtractorCallT", bound=Callable[..., Any])


def grayscale_dims(image: MatLike, dims: str) -> str:
    """
    Drop the channel label from ``dims`` for a single-channel (grayscale) image.

    A grayscale image carries no channel axis, so an array with exactly one
    fewer dimension than a channel-bearing ``dims`` (e.g. a 2-D array with the
    default ``"HWC"``) is treated as single-channel and the ``"C"`` label is
    removed. This keeps the canonical ``(H, W)`` grayscale layout working with
    the channel-bearing default, matching the NumPy-only behaviour the library
    accepted before ``dims`` were introduced.

    :param image: The image whose axis count is inspected.
    :param dims: The requested axis-label string.
    :return: ``dims`` with ``"C"`` removed when ``image`` is single-channel,
        otherwise ``dims`` unchanged.
    """
    normalized = dims.upper()
    if "C" not in normalized:
        return dims
    if np.ndim(image) == len(normalized) - 1:
        return normalized.replace("C", "")
    return dims


def _to_single_image(
    image: MatLike,
    dims: str = "HWC",
    value_range: tuple[float, float] = (0.0, 255.0),
) -> UInt8NumpyArray:
    """
    Normalize a single ``MatLike`` image into one canonical array.

    :param image: A NumPy array, a PyTorch tensor, or any array-like object.
    :param dims: Axis-label string, one character per array axis in order:
        ``"H"`` = height (rows), ``"W"`` = width (columns), ``"C"`` = channels
        (e.g. RGB), ``"B"`` = batch size. For example, ``"HWC"`` is height ×
        width × channels (NumPy/OpenCV single-image layout, **default**);
        ``"CHW"`` is channels × height × width (PyTorch single-image layout);
        ``"BCHW"`` is batch × channels × height × width (PyTorch batched layout).
        A single-channel (grayscale) image may be passed as a 2-D array with the
        default ``"HWC"``; the channel label is dropped automatically.
    :param value_range: The ``(low, high)`` range the input values live in
        (default ``(0.0, 255.0)``).
    :return: A ``uint8`` image of shape ``(H, W[, C])`` in ``[0, 255]``.
    :raises ValueError: If the input expands to anything other than one image.
    """
    images = _to_image_list(image, grayscale_dims(image, dims), value_range)
    if len(images) != 1:
        raise ValueError(
            f"Expected a single image, but the input expands to {len(images)} images. "
            "Feature extractors operate on one image at a time; use an encoder for "
            "batches."
        )
    return images[0]


def _check_output_shape(  # noqa: UP047
    func: ExtractorCallT,
) -> ExtractorCallT:
    """
    Ensures the feature extractor output is a 2D NumPy array of shape
    (num_vectors, self.output_dim).

    Input normalization (``MatLike`` conversion plus ``dims``/``value_range``
    handling) is performed inside each wrapped ``__call__``; this decorator
    only validates the output.
    """

    @wraps(func)
    def wrapper(
        self: FeatureExtractorBase, image: MatLike, /, *args: Any, **kwargs: Any
    ) -> Float32NumpyArray:
        feat_vecs = func(self, image, *args, **kwargs)
        if feat_vecs is None:
            print("No feature vectors found. Returning empty array.")
            return np.zeros((0, self.output_dim), dtype=np.float32)

        if not isinstance(feat_vecs, np.ndarray):
            raise ValueError(
                f"Expected output to be a NumPy array, got {type(feat_vecs)} instead."
            )

        if feat_vecs.ndim != 2:
            raise ValueError(
                f"Feature extractor output must be 2D. Got shape {feat_vecs.shape}."
            )

        if feat_vecs.shape[1] != self.output_dim:
            raise ValueError(
                f"Expected feat_vecs.shape[1] == {self.output_dim}, "
                f"but got {feat_vecs.shape[1]}."
            )

        return feat_vecs

    return cast(ExtractorCallT, wrapper)
