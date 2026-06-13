"""Principal Component Analysis model used by the image encoders."""

from typing import Any

import numpy as np
from sklearn.decomposition import PCA as _SklearnPCA

from ._base_clustering import _SklearnModelBase


class PCA(_SklearnModelBase):
    """
    Principal Component Analysis model, used by the image encoders to
    reduce the dimensionality of local features. It is backed by
    :class:`sklearn.decomposition.PCA`.

    :param n_components: Number of components to keep.
    :param pca_params: Additional keyword arguments forwarded verbatim to
        :class:`sklearn.decomposition.PCA` (e.g. ``whiten``, ``random_state``).
    """

    _sklearn_class = _SklearnPCA

    def __init__(self, n_components: int, **pca_params: Any) -> None:
        super().__init__(_SklearnPCA(n_components=n_components, **pca_params))

    @property
    def n_components(self) -> int:
        """
        Number of components of the fitted PCA.

        :raises NotFittedError: If the underlying estimator is not fitted.
        """
        self._check_is_fitted()
        return int(self._model.n_components_)

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Projects the given features onto the principal components.

        :param features: Feature matrix of shape (n_samples, n_features).
        :return: Reduced features of shape (n_samples, n_components).
        :raises NotFittedError: If the underlying estimator is not fitted.
        """
        self._check_is_fitted()
        return np.asarray(self._model.transform(features))
