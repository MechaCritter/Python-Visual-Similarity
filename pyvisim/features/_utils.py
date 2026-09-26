from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

import numpy as np

from ..base import FeatureExtractorBase
from ..typing import Float32NumpyArray, MatLike

ExtractorCallT = TypeVar("ExtractorCallT", bound=Callable[..., Any])


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
