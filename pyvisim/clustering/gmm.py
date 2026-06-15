"""Gaussian Mixture Model used by the Fisher Vector encoder."""

from typing import Any

import numpy as np
from sklearn.mixture import GaussianMixture

from ..typing import Float64NumpyArray, FloatNumpyArray
from ._base_clustering import ClusteringModelBase


class GaussianMixtureModel(ClusteringModelBase):
    """
    Gaussian Mixture clustering model, used by the Fisher Vector encoder.
    It is backed by :class:`sklearn.mixture.GaussianMixture`.

    Only diagonal covariance matrices are supported: the Fisher Vector
    computation relies on them, and training is much faster.

    :param n_components: Number of mixture components.
    :param gmm_params: Additional keyword arguments forwarded verbatim to
        :class:`sklearn.mixture.GaussianMixture` (e.g. ``random_state``).
    :raises ValueError: If a ``covariance_type`` other than ``"diag"`` is requested.
    """

    _sklearn_class = GaussianMixture

    def __init__(self, n_components: int = 256, **gmm_params: Any) -> None:
        covariance_type = gmm_params.pop("covariance_type", "diag")
        if covariance_type != "diag":
            raise ValueError(
                f"{type(self).__name__} only supports covariance_type='diag', "
                f"got {covariance_type!r}."
            )
        super().__init__(
            GaussianMixture(
                n_components=n_components, covariance_type="diag", **gmm_params
            )
        )

    def _validate_sklearn_model(self) -> None:
        if self._model.covariance_type != "diag":
            raise ValueError(
                f"{type(self).__name__} only supports covariance_type='diag', "
                f"got {self._model.covariance_type!r}."
            )

    @property
    def n_clusters(self) -> int:
        """Number of mixture components of the GMM."""
        return int(self._model.n_components)

    @property
    def weights(self) -> Float64NumpyArray:
        """
        Mixture weights of each component, shape (n_components,).

        :raises NotFittedError: If the underlying estimator is not fitted.
        """
        self._check_is_fitted()
        return np.asarray(self._model.weights_)

    @property
    def means(self) -> Float64NumpyArray:
        """
        Mean of each mixture component, shape (n_components, n_features).

        :raises NotFittedError: If the underlying estimator is not fitted.
        """
        self._check_is_fitted()
        return np.asarray(self._model.means_)

    @property
    def covariances(self) -> Float64NumpyArray:
        """
        Diagonal covariance of each component, shape (n_components, n_features).

        :raises NotFittedError: If the underlying estimator is not fitted.
        """
        self._check_is_fitted()
        return np.asarray(self._model.covariances_)

    def predict_proba(self, features: FloatNumpyArray) -> Float64NumpyArray:
        """
        Evaluates the components' posterior probability for each feature vector.

        :param features: Feature matrix of shape (n_samples, n_features).
        :return: Posterior probabilities, shape (n_samples, n_components).
        :raises NotFittedError: If the underlying estimator is not fitted.
        """
        self._check_is_fitted()
        return np.asarray(self._model.predict_proba(features))
