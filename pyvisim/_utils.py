from typing import cast

import numpy as np
from PIL import Image, UnidentifiedImageError

from . import distance
from .typing import (
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
    try:
        with Image.open(path) as image:
            rgb_image = image.convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError, OSError) as error:
        raise FileNotFoundError(f"Could not read image at '{path}'.") from error
    return cast(UInt8NumpyArray, np.asarray(rgb_image))


SIMILARITY_FUNCTIONS: dict[str, SimilarityFunc] = {
    "cosine": distance.cosine_similarity,
    "euclidean": distance.euclidean_distances,
    "l1": distance.manhattan_distances,
    "manhattan": distance.manhattan_distances,
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
