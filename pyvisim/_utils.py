from typing import cast

import cv2
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity as cs
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

from .typing import (
    Float64NumpyArray,
    FloatNumpyArray,
    SimilarityFunc,
    UInt8NumpyArray,
)


def read_image_rgb(path: str) -> UInt8NumpyArray:
    """
    Read an image from disk and convert it to RGB.

    :param path: Path to the image file.
    :return: Image as a NumPy array (H, W, C) in RGB order.
    :raises FileNotFoundError: If the image cannot be read from the given path.
    """
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at '{path}'.")
    return cast(UInt8NumpyArray, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _as_2d_array(value: FloatNumpyArray) -> FloatNumpyArray:
    """
    Coerce a similarity-function input into a 2-D NumPy array.

    Torch tensors are moved to CPU and converted to NumPy, and a single
    1-D vector is reshaped into a ``(1, D)`` row.

    :param value: A NumPy array or a torch tensor.
    :return: A 2-D NumPy array of shape ``(N, D)``.
    """
    if isinstance(value, torch.Tensor):
        value = value.cpu().numpy()
    return value.reshape(1, -1) if value.ndim == 1 else value


def cosine_similarity(x: FloatNumpyArray, y: FloatNumpyArray) -> Float64NumpyArray:
    """
    Compute the pairwise cosine similarity between two matrices.

    Higher values mean more similar.

    :param x: First matrix of shape ``(N, D)`` (or ``(D,)``).
    :param y: Second matrix of shape ``(M, D)`` (or ``(D,)``).
    :return: Cosine similarity matrix of shape ``(N, M)``.
    :raises ValueError: If either input has fewer than 2 features.
    """
    x = _as_2d_array(x)
    y = _as_2d_array(y)
    if x.shape[-1] <= 1 or y.shape[-1] <= 1:
        raise ValueError(
            f"Cosine similarity requires at least 2 features. Got {x.shape[-1]} features for x and {y.shape[-1]} features for y."
        )

    return np.asarray(cs(x, y))


def euclidean_similarity(x: FloatNumpyArray, y: FloatNumpyArray) -> Float64NumpyArray:
    """
    Compute the pairwise Euclidean (L2) distance between two matrices.

    Lower values mean more similar.

    :param x: First matrix of shape ``(N, D)`` (or ``(D,)``).
    :param y: Second matrix of shape ``(M, D)`` (or ``(D,)``).
    :return: Euclidean distance matrix of shape ``(N, M)``.
    """
    return np.asarray(euclidean_distances(_as_2d_array(x), _as_2d_array(y)))


def manhattan_similarity(x: FloatNumpyArray, y: FloatNumpyArray) -> Float64NumpyArray:
    """
    Compute the pairwise Manhattan (L1) distance between two matrices.

    Lower values mean more similar. ``"l1"`` and ``"manhattan"`` are aliases
    for this metric.

    :param x: First matrix of shape ``(N, D)`` (or ``(D,)``).
    :param y: Second matrix of shape ``(M, D)`` (or ``(D,)``).
    :return: Manhattan distance matrix of shape ``(N, M)``.
    """
    return np.asarray(manhattan_distances(_as_2d_array(x), _as_2d_array(y)))


#: Mapping of supported similarity-function names to their implementations.
#: ``"l1"`` and ``"manhattan"`` are aliases of the same metric.
SIMILARITY_FUNCTIONS: dict[str, SimilarityFunc] = {
    "cosine": cosine_similarity,
    "euclidean": euclidean_similarity,
    "l1": manhattan_similarity,
    "manhattan": manhattan_similarity,
}


def get_similarity_func(name: str) -> SimilarityFunc:
    """
    Resolve a similarity-function name to its implementation.

    Only the built-in metrics are supported; user-defined similarity
    functions are no longer accepted.

    :param name: One of ``"cosine"``, ``"euclidean"``, ``"l1"`` or ``"manhattan"``.
    :return: The matching similarity function.
    :raises ValueError: If ``name`` is not a supported metric.
    """
    try:
        return SIMILARITY_FUNCTIONS[name]
    except (KeyError, TypeError):
        raise ValueError(
            f"Unsupported similarity function {name!r}. Supported metrics are: "
            f"{sorted(SIMILARITY_FUNCTIONS)}. Custom similarity functions are no "
            "longer supported."
        ) from None
