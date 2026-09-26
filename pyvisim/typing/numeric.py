"""Numeric and image types used across pyvisim."""

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

#: Generic NumPy array of any dtype.
NumpyArray = npt.NDArray[np.generic]
#: NumPy array of uint8 values (canonical image layout).
UInt8NumpyArray = npt.NDArray[np.uint8]
#: NumPy array of float32 values (feature descriptors and embeddings).
Float32NumpyArray = npt.NDArray[np.float32]
#: NumPy array of float64 values (model outputs computed in double precision).
Float64NumpyArray = npt.NDArray[np.float64]
#: NumPy array of any floating-point dtype.
FloatNumpyArray = npt.NDArray[np.floating[Any]]
#: NumPy array of platform-native signed integers (cluster labels / indices).
IntNumpyArray = npt.NDArray[np.intp]
#: Boolean matrix.
BoolNumpyArray = npt.NDArray[np.bool_]

#: A similarity function: maps two batches of feature vectors of shapes
#: ``(N, D)`` and ``(M, D)`` to an ``(N, M)`` similarity matrix.
SimilarityFunc = Callable[[FloatNumpyArray, FloatNumpyArray], FloatNumpyArray]

#: Anything that can be turned into a numerical NumPy array: a NumPy array, a
#: PyTorch tensor, or any array-like object (e.g. nested lists of numbers).
if TYPE_CHECKING:
    import torch

    MatLike = torch.Tensor | npt.ArrayLike
else:
    MatLike = npt.ArrayLike

#: A single image or a collection of images. Either one ``MatLike`` object
#: (optionally carrying a batch axis) or an iterable of ``MatLike`` images.
ImageInput = MatLike | Iterable[MatLike]
