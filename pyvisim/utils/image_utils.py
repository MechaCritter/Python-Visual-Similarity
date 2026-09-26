import itertools
from collections.abc import Iterable, Iterator
from typing import cast

import numpy as np

from .._errors import InvalidImageError
from ..lazy_import import is_tensor
from ..typing import ImageInput, MatLike, NumpyArray, UInt8NumpyArray

_VALID_DIM_CHARS = frozenset("BHWC")
_CANONICAL_AXIS_ORDER = "BHWC"
_DEFAULT_VALUE_RANGE: tuple[float, float] = (0.0, 255.0)


def _to_ndarray(data: MatLike) -> NumpyArray:
    """Convert any ``MatLike`` object into a numerical NumPy array."""
    if is_tensor(data):
        return data.detach().cpu().numpy()
    if isinstance(data, np.ndarray):
        array = data
    else:
        try:
            array = np.asarray(data)
        except (ValueError, TypeError) as exc:
            raise InvalidImageError(
                f"Could not convert object of type {type(data).__name__!r} "
                "to a NumPy array."
            ) from exc
    if not np.issubdtype(array.dtype, np.number) and not np.issubdtype(
        array.dtype, np.bool_
    ):
        raise InvalidImageError(
            f"Expected a numeric array, but got an array with dtype {array.dtype!r}."
        )
    return array


def _validate_dims(dims: str, ndim: int) -> str:
    """
    Validate a ``dims`` string against the number of array dimensions.

    :param dims: Axis-label string, one character per axis (case-insensitive).
        ``"H"`` = height, ``"W"`` = width, ``"C"`` = channels, ``"B"`` = batch.
        For example, ``"HWC"`` is height × width × channels.
    :param ndim: Number of dimensions of the array the labels describe.
    :return: The normalized (upper-cased) ``dims`` string.
    :raises ValueError: If the string is malformed or inconsistent with ``ndim``.
    """
    normalized = dims.upper()
    if len(normalized) != ndim:
        raise ValueError(
            f"'dims' string {dims!r} describes {len(normalized)} axis/axes, "
            f"but the array has {ndim} dimension(s)."
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"'dims' string {dims!r} contains duplicate axis labels.")
    invalid = set(normalized) - _VALID_DIM_CHARS
    if invalid:
        raise ValueError(
            f"'dims' string {dims!r} contains invalid axis labels {sorted(invalid)}. "
            "Only 'B' (batch), 'H' (height), 'W' (width) and 'C' (channels) are allowed."
        )
    if "H" not in normalized or "W" not in normalized:
        raise ValueError(
            f"'dims' string {dims!r} must contain both 'H' (height) and 'W' (width)."
        )
    return normalized


def _to_uint8(array: NumpyArray, value_range: tuple[float, float]) -> UInt8NumpyArray:
    """
    Rescale an array from ``value_range`` into the canonical ``[0, 255]`` uint8 range.

    :param array: Numeric array to rescale.
    :param value_range: The ``(low, high)`` range the input values live in.
    :return: A ``uint8`` array with values clipped to ``[0, 255]``.
    :raises ValueError: If ``value_range`` is not strictly increasing.
    """
    low, high = value_range
    if high <= low:
        raise ValueError(
            f"'value_range' must be an increasing (low, high) tuple, got {value_range}."
        )
    if value_range == _DEFAULT_VALUE_RANGE and array.dtype == np.uint8:
        return cast(UInt8NumpyArray, array)
    scaled = (array.astype(np.float64) - low) / (high - low) * 255.0
    scaled = np.clip(scaled, 0.0, 255.0)
    return scaled.astype(np.uint8)


def _split_into_images(array: UInt8NumpyArray, dims: str) -> list[UInt8NumpyArray]:
    """
    Reorder an array into ``(B, H, W[, C])`` and split it along the batch axis.

    :param array: Array whose axes are described by ``dims``.
    :param dims: Validated, normalized ``dims`` string.
    :return: A list of contiguous per-image arrays of shape ``(H, W[, C])``.
    """
    target = "".join(axis for axis in _CANONICAL_AXIS_ORDER if axis in dims)
    reordered = np.transpose(array, [dims.index(axis) for axis in target])
    if "B" not in dims:
        reordered = reordered[np.newaxis, ...]
    return [
        np.ascontiguousarray(reordered[index]) for index in range(reordered.shape[0])
    ]


def _to_image_list(
    images: MatLike,
    dims: str = "HWC",
    value_range: tuple[float, float] = (0.0, 255.0),
) -> list[UInt8NumpyArray]:
    """
    Normalize a single ``MatLike`` object into canonical per-image arrays.

    The input may carry a batch axis (when ``dims`` contains ``"B"``), in which
    case it is split into the individual images it holds. Check out the
    documentation here for more informaton:
    https://mechacritter.github.io/Python-Visual-Similarity/typing/index.html#keyword-arguments-for-image-data

    :param images: A NumPy array, a PyTorch tensor, or any array-like object.
    :param dims: Axis-label string, one character per array axis in order:
        ``"H"`` = height (rows), ``"W"`` = width (columns), ``"C"`` = channels
        (e.g. RGB), ``"B"`` = batch size. For example, ``"HWC"`` is height ×
        width × channels (NumPy/OpenCV single-image layout);
        ``"CHW"`` is channels × height × width (PyTorch single-image layout);
        ``"BCHW"`` is batch × channels × height × width (PyTorch batched layout).
    :param value_range: The ``(low, high)`` range the input values live in.
    :return: A list of ``uint8`` images of shape ``(H, W[, C])`` in ``[0, 255]``.
    :raises InvalidImageError: If the input cannot be converted to a numeric array.
    :raises ValueError: If ``dims`` or ``value_range`` are invalid.
    """
    array = _to_ndarray(images)
    normalized_dims = _validate_dims(dims, array.ndim)
    array = _to_uint8(array, value_range)
    return _split_into_images(array, normalized_dims)


def grayscale_dims(image: MatLike, dims: str) -> str:
    """
    Drop the channel label from ``dims`` for a single-channel (grayscale) image.

    A grayscale image carries no channel axis, so an array with exactly one
    fewer dimension than a channel-bearing ``dims`` (e.g. a 2-D array with
    ``"HWC"``) is treated as single-channel and the ``"C"`` label is
    removed. This keeps the canonical ``(H, W)`` grayscale layout working with
    a channel-bearing ``dims``, matching the NumPy-only behaviour the library
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


def to_single_image(
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
        width × channels (NumPy/OpenCV single-image layout);
        ``"CHW"`` is channels × height × width (PyTorch single-image layout);
        ``"BCHW"`` is batch × channels × height × width (PyTorch batched layout).
        A single-channel (grayscale) image may be passed as a 2-D array with
        ``dims="HWC"``, and the channel label is dropped automatically.
    :param value_range: The ``(low, high)`` range the input values live in.
    :return: A ``uint8`` image of shape ``(H, W[, C])`` in ``[0, 255]``.
    :raises InvalidImageError: If the input cannot be converted to a numeric array.
    :raises ValueError: If ``dims`` or ``value_range`` are invalid, or if the
        input expands to anything other than one image.
    """
    images = _to_image_list(image, grayscale_dims(image, dims), value_range)
    if len(images) != 1:
        raise ValueError(
            f"Expected a single image, but the input expands to {len(images)} images."
        )
    return images[0]


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
        width × channels (NumPy/OpenCV single-image layout);
        ``"CHW"`` is channels × height × width (PyTorch single-image layout);
        ``"BCHW"`` is batch × channels × height × width (PyTorch batched layout).
    :param value_range: The ``(low, high)`` range the input values live in.
    :return: An iterator over ``uint8`` images of shape ``(H, W[, C])``.
    :raises InvalidImageError: If a string/bytes object is passed as an image.
    """
    if not (isinstance(images, (np.ndarray, Iterable)) or is_tensor(images)):
        raise InvalidImageError(
            f"Expected image array(s), but got a {type(images).__name__} object."
        )
    if isinstance(images, np.ndarray) or is_tensor(images):
        yield from _to_image_list(images, dims, value_range)
        return
    if isinstance(images, Iterable):
        for image in images:
            yield from _to_image_list(image, grayscale_dims(image, dims), value_range)
        return
    yield from _to_image_list(images, dims, value_range)


def iter_image_batches(
    images: ImageInput,
    batch_size: int,
    dims: str = "HWC",
    value_range: tuple[float, float] = (0.0, 255.0),
) -> Iterator[list[UInt8NumpyArray]]:
    """
    Yield canonical per-image arrays in batches of at most ``batch_size``.

    The images are normalized by :func:`iter_images` and grouped lazily, so an
    iterable input is consumed one batch at a time and never held in memory as
    a whole. The last batch may be shorter than ``batch_size``; an input
    holding no image yields no batch at all.

    ``batch_size=-1`` yields the whole input as a single batch.

    :param images: A single ``MatLike`` object or an iterable of ``MatLike`` images.
    :param batch_size: Maximum number of images processed in a single batch.
        Set to ``-1`` to process all images as a single batch.
    :param dims: Axis-label string, one character per array axis in order:
        ``"H"`` = height (rows), ``"W"`` = width (columns), ``"C"`` = channels
        (e.g. RGB), ``"B"`` = batch size. See :mod:`pyvisim.typing`.
    :param value_range: The ``(low, high)`` range the input values live in.
    :return: An iterator over lists of at most ``batch_size`` ``uint8`` images
        of shape ``(H, W[, C])``.
    :raises ValueError: If ``batch_size`` is neither ``-1`` nor a positive integer.
    :raises InvalidImageError: If a string/bytes object is passed as an image.
    """
    if batch_size != -1 and batch_size < 1:
        raise ValueError(
            "batch_size must be a positive integer or -1 (process the whole "
            f"input as one batch), got {batch_size}."
        )
    image_iterator = iter_images(images, dims=dims, value_range=value_range)
    if batch_size == -1:
        if batch := list(image_iterator):
            yield batch
        return
    while batch := list(itertools.islice(image_iterator, batch_size)):
        yield batch
