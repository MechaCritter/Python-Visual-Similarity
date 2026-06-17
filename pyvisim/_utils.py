from typing import cast

import cv2
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity as cs

from .typing import (
    Float64NumpyArray,
    FloatNumpyArray,
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


def cosine_similarity(x: FloatNumpyArray, y: FloatNumpyArray) -> Float64NumpyArray:
    """
    Compute the cosine similarity between two matrices.

    :param x: First matrix
    :param y: Second matrix

    :return: Cosine similarity matrix
    """
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    x = x.reshape(1, -1) if len(x.shape) == 1 else x
    y = y.reshape(1, -1) if len(y.shape) == 1 else y
    if x.shape[-1] <= 1 or y.shape[-1] <= 1:
        raise ValueError(
            f"Cosine similarity requires at least 2 features. Got {x.shape[-1]} features for x and {y.shape[-1]} features for y."
        )

    return np.asarray(cs(x, y))
