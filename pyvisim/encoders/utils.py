from collections.abc import Iterable, Iterator

import numpy as np
import torch

from .._errors import InvalidImageError
from ..features._features import grayscale_dims
from ..typing import ImageInput, UInt8NumpyArray, _to_image_list


def iter_images(
    images: ImageInput,
    dims: str = "HWC",
    value_range: tuple[float, float] = (0.0, 255.0),
) -> Iterator[UInt8NumpyArray]:
    """
    Yield canonical per-image arrays from a single object or an iterable.

    A single (possibly batched) ``MatLike`` array/tensor is normalized and its
    images are yielded. Any other iterable is treated as a collection of
    ``MatLike`` images, each of which is normalized in turn; if its elements
    carry a batch axis (per ``dims``), every image is still yielded, so a batch
    size is handled gracefully.

    :param images: A single ``MatLike`` object or an iterable of ``MatLike`` images.
    :param dims: Axis-label string, one character per array axis in order:
        ``"H"`` = height (rows), ``"W"`` = width (columns), ``"C"`` = channels
        (e.g. RGB), ``"B"`` = batch size. For example, ``"HWC"`` is height ×
        width × channels (NumPy/OpenCV single-image layout, **default**);
        ``"CHW"`` is channels × height × width (PyTorch single-image layout);
        ``"BCHW"`` is batch × channels × height × width (PyTorch batched layout).
    :param value_range: The ``(low, high)`` range the input values live in
        (default ``(0.0, 255.0)``).
    :return: An iterator over ``uint8`` images of shape ``(H, W[, C])``.
    :raises InvalidImageError: If a string/bytes object is passed as an image.
    """
    if not isinstance(images, (np.ndarray, torch.Tensor, Iterable)):
        raise InvalidImageError(
            f"Expected image array(s), but got a {type(images).__name__} object."
        )
    if isinstance(images, (np.ndarray, torch.Tensor)):
        yield from _to_image_list(images, dims, value_range)
        return
    if isinstance(images, Iterable):
        for image in images:
            yield from _to_image_list(image, grayscale_dims(image, dims), value_range)
        return
    yield from _to_image_list(images, dims, value_range)
