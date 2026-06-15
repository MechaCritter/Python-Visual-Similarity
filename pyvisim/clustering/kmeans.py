"""K-Means clustering class used by the VLAD encoder."""

from typing import Any

import numpy as np
from sklearn.cluster import KMeans as _SklearnKMeans

from ..typing import FloatNumpyArray, IntNumpyArray
from ._base_clustering import ClusteringModelBase


class KMeans(ClusteringModelBase):
    """
    K-Means clustering model, used by the VLAD encoder. It is
    backed by :class:`sklearn.cluster.KMeans`.

    :param n_clusters: Number of clusters to form.
    :param kmeans_params: Additional keyword arguments forwarded verbatim to
        :class:`sklearn.cluster.KMeans` (e.g. ``random_state``, ``n_init``).
    """

    _sklearn_class = _SklearnKMeans

    def __init__(self, n_clusters: int = 256, **kmeans_params: Any) -> None:
        super().__init__(_SklearnKMeans(n_clusters=n_clusters, **kmeans_params))

    @property
    def n_clusters(self) -> int:
        """Number of clusters of the K-Means model."""
        return int(self._model.n_clusters)

    @property
    def cluster_centers(self) -> FloatNumpyArray:
        """
        Coordinates of the cluster centers, shape (n_clusters, n_features).

        :raises NotFittedError: If the underlying estimator is not fitted.
        """
        self._check_is_fitted()
        return np.asarray(self._model.cluster_centers_)

    def predict(self, features: FloatNumpyArray) -> IntNumpyArray:
        """
        Predicts the closest cluster for each feature vector.

        :param features: Feature matrix of shape (n_samples, n_features).
        :return: Cluster index of each sample, shape (n_samples,).
        :raises NotFittedError: If the underlying estimator is not fitted.
        """
        self._check_is_fitted()
        return np.asarray(self._model.predict(features))
