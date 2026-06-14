"""Tests for :class:`pyvisim.clustering.KMeans`."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from pyvisim.clustering import KMeans


@pytest.fixture
def fitted_kmeans() -> KMeans:
    """A reproducible ``KMeans(8)`` fitted on ``(200, 5)`` data.

    :returns: a fitted :class:`pyvisim.clustering.KMeans` instance.
    """
    rng = np.random.default_rng(0)
    model = KMeans(8, random_state=0, n_init=3)
    model.fit(rng.random((200, 5)))
    return model


def test_kmeans_n_clusters_before_fit() -> None:
    """``n_clusters`` reflects the constructor argument without fitting."""
    assert KMeans(8).n_clusters == 8


def test_kmeans_is_fitted_false_before_fit() -> None:
    """A freshly built KMeans reports ``is_fitted is False``."""
    assert KMeans(8).is_fitted is False


def test_kmeans_cluster_centers_before_fit_raises() -> None:
    """Accessing ``cluster_centers`` before fitting raises ``NotFittedError``."""
    with pytest.raises(NotFittedError):
        _ = KMeans(8).cluster_centers


def test_kmeans_predict_before_fit_raises() -> None:
    """Calling ``predict`` before fitting raises ``NotFittedError``."""
    rng = np.random.default_rng(0)
    with pytest.raises(NotFittedError):
        KMeans(8).predict(rng.random((10, 5)))


def test_kmeans_cluster_centers_shape(fitted_kmeans: KMeans) -> None:
    """Cluster centers have shape ``(n_clusters, n_features)``."""
    assert fitted_kmeans.cluster_centers.shape == (8, 5)


def test_kmeans_n_features_in_after_fit(fitted_kmeans: KMeans) -> None:
    """A fitted KMeans reports the number of input features it saw."""
    assert fitted_kmeans.n_features_in == 5


def test_kmeans_predict_labels_in_range(fitted_kmeans: KMeans) -> None:
    """``predict`` returns one label per sample, each in ``[0, n_clusters)``."""
    rng = np.random.default_rng(1)
    labels = fitted_kmeans.predict(rng.random((50, 5)))
    assert labels.shape == (50,)
    assert labels.min() >= 0
    assert labels.max() < 8


def test_kmeans_predict_is_ndarray(fitted_kmeans: KMeans) -> None:
    """``predict`` returns a NumPy array."""
    rng = np.random.default_rng(1)
    assert isinstance(fitted_kmeans.predict(rng.random((50, 5))), np.ndarray)


def test_kmeans_random_state_reproducible() -> None:
    """A fixed ``random_state`` yields identical cluster centers."""
    rng = np.random.default_rng(0)
    data = rng.random((200, 5))
    first = KMeans(8, random_state=0, n_init=3)
    second = KMeans(8, random_state=0, n_init=3)
    first.fit(data)
    second.fit(data)
    assert np.allclose(first.cluster_centers, second.cluster_centers)


def test_kmeans_from_sklearn_wrong_type_raises() -> None:
    """``_from_sklearn`` rejects objects that are not the expected estimator."""
    with pytest.raises(TypeError):
        KMeans._from_sklearn(object())
