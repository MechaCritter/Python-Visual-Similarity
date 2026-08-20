"""Tests for :class:`pyvisim.classic._clustering.KMeans`."""

from __future__ import annotations

import numpy as np
import pytest

from pyvisim._errors import NotFittedError
from pyvisim.classic._clustering import KMeans


@pytest.fixture
def fitted_kmeans() -> KMeans:
    """A reproducible ``KMeans(8)`` fitted on ``(200, 5)`` data.

    :returns: a fitted :class:`pyvisim.classic._clustering.KMeans` instance.
    """
    rng = np.random.default_rng(0)
    model = KMeans(8, rng=0)
    model.fit(rng.random((200, 5)))
    return model


def test_kmeans_n_clusters_before_fit() -> None:
    """``n_clusters`` reflects the constructor argument without fitting."""
    assert KMeans(8).n_clusters == 8


def test_kmeans_is_fitted_false_before_fit() -> None:
    """A freshly built KMeans reports ``is_fitted is False``."""
    assert KMeans(8).is_fitted is False


def test_kmeans_nonpositive_n_clusters_raises() -> None:
    """A non-positive ``n_clusters`` is rejected at construction."""
    with pytest.raises(ValueError, match="n_clusters"):
        KMeans(0)


def test_kmeans_nonpositive_n_init_raises() -> None:
    """A non-positive ``n_init`` is rejected at construction."""
    with pytest.raises(ValueError, match="n_init"):
        KMeans(8, n_init=0)


def test_kmeans_fewer_samples_than_clusters_raises() -> None:
    """Fitting on fewer samples than clusters raises ``ValueError``."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="Cannot form"):
        KMeans(8, rng=0).fit(rng.random((5, 3)))


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


def test_kmeans_predict_matches_nearest_whitened_centroid(
    fitted_kmeans: KMeans,
) -> None:
    """``predict`` assigns each sample to its nearest centroid in whitened space."""
    rng = np.random.default_rng(2)
    samples = rng.random((50, 5))
    scale = fitted_kmeans._scale
    distances = np.linalg.norm(
        (samples / scale)[:, None, :]
        - (fitted_kmeans.cluster_centers / scale)[None, :, :],
        axis=2,
    )
    assert np.array_equal(fitted_kmeans.predict(samples), distances.argmin(axis=1))


def test_kmeans_fit_whitens_training_data() -> None:
    """Fitting stores the per-feature std of the training data as the scale."""
    rng = np.random.default_rng(0)
    data = rng.random((300, 3)) * np.array([1.0, 100.0, 0.01])
    model = KMeans(4, rng=0)
    model.fit(data)
    assert np.allclose(model._scale, data.std(axis=0))
    assert model.cluster_centers.shape == (4, 3)


def test_kmeans_fit_zero_variance_feature() -> None:
    """A constant feature gets scale 1 instead of a division by zero."""
    rng = np.random.default_rng(0)
    data = rng.random((200, 3))
    data[:, 1] = 7.0
    model = KMeans(4, rng=0)
    model.fit(data)
    assert model._scale[1] == 1.0
    assert np.isfinite(model.cluster_centers).all()
    assert model.predict(data).shape == (200,)


def test_kmeans_rng_reproducible() -> None:
    """A fixed ``rng`` seed yields identical cluster centers."""
    rng = np.random.default_rng(0)
    data = rng.random((200, 5))
    first = KMeans(8, rng=0)
    second = KMeans(8, rng=0)
    first.fit(data)
    second.fit(data)
    assert np.allclose(first.cluster_centers, second.cluster_centers)


def test_kmeans_fit_integer_data() -> None:
    """Integer input is cast to float before fitting."""
    rng = np.random.default_rng(0)
    model = KMeans(4, rng=0)
    model.fit(rng.integers(0, 255, size=(100, 3)))
    assert model.cluster_centers.shape == (4, 3)


def test_kmeans_to_dict_from_dict_roundtrip(fitted_kmeans: KMeans) -> None:
    """A serialised model reloads with identical centroids and predictions."""
    rng = np.random.default_rng(3)
    samples = rng.random((30, 5))
    restored = KMeans.from_dict(fitted_kmeans.to_dict())
    assert restored.n_clusters == fitted_kmeans.n_clusters
    assert np.array_equal(restored.cluster_centers, fitted_kmeans.cluster_centers)
    assert np.array_equal(restored._scale, fitted_kmeans._scale)
    assert np.array_equal(restored.predict(samples), fitted_kmeans.predict(samples))


def test_kmeans_from_dict_legacy_sklearn_state() -> None:
    """States serialised from the old sklearn-backed model still load."""
    centers = np.arange(15, dtype=np.float64).reshape(3, 5)
    legacy = {
        "__class__": "KMeans",
        "__module__": "sklearn.cluster._kmeans",
        "state": {
            "n_clusters": 3,
            "n_features_in_": 5,
            "random_state": 0,
            "n_init": 3,
            "cluster_centers_": {
                "__ndarray__": True,
                "data": centers.tolist(),
                "dtype": "float64",
                "shape": [3, 5],
                "order": "C",
            },
        },
    }
    model = KMeans.from_dict(legacy)
    assert model.is_fitted
    assert model.n_clusters == 3
    assert model.n_features_in == 5
    assert model._n_init == 3  # legacy states share the "n_init" key
    assert np.array_equal(model.cluster_centers, centers)
    # The sklearn-backed model trained without whitening: the scale is ones.
    assert np.array_equal(model._scale, np.ones(5))


def test_kmeans_from_dict_legacy_auto_n_init() -> None:
    """A legacy state with ``n_init='auto'`` resolves to a single seeding."""
    centers = np.arange(10, dtype=np.float64).reshape(2, 5)
    legacy = {
        "__class__": "KMeans",
        "state": {
            "n_clusters": 2,
            "n_init": "auto",
            "cluster_centers_": {
                "__ndarray__": True,
                "data": centers.tolist(),
                "dtype": "float64",
                "shape": [2, 5],
                "order": "C",
            },
        },
    }
    assert KMeans.from_dict(legacy)._n_init == 1


def test_kmeans_from_sklearn_maps_params() -> None:
    """``from_sklearn`` translates the scikit-learn constructor arguments."""
    from sklearn.cluster import KMeans as SklearnKMeans

    sklearn_model = SklearnKMeans(n_clusters=8, tol=1e-3, n_init=5, random_state=42)
    model = KMeans.from_sklearn(sklearn_model)
    assert model.n_clusters == 8
    assert model._n_init == 5
    assert model._thresh == pytest.approx(1e-3)
    assert model._rng == 42
    assert model.is_fitted is False


def test_kmeans_from_sklearn_auto_n_init_maps_to_one() -> None:
    """scikit-learn's ``n_init='auto'`` resolves to a single seeding."""
    from sklearn.cluster import KMeans as SklearnKMeans

    assert KMeans.from_sklearn(SklearnKMeans(n_clusters=4))._n_init == 1


def test_kmeans_from_sklearn_adopts_fitted_centroids() -> None:
    """``from_sklearn`` adopts the centroids of a fitted estimator."""
    from sklearn.cluster import KMeans as SklearnKMeans

    rng = np.random.default_rng(0)
    data = rng.random((200, 5))
    sklearn_model = SklearnKMeans(n_clusters=8, random_state=0).fit(data)
    model = KMeans.from_sklearn(sklearn_model)
    assert model.is_fitted
    assert np.array_equal(model.cluster_centers, sklearn_model.cluster_centers_)
    assert np.array_equal(model.predict(data), sklearn_model.predict(data))


def test_kmeans_from_sklearn_random_init_raises() -> None:
    """``init='random'`` is rejected; k-means++ is the only initialization."""
    from sklearn.cluster import KMeans as SklearnKMeans

    sklearn_model = SklearnKMeans(n_clusters=2, init="random")
    with pytest.raises(ValueError, match="Unsupported scikit-learn init"):
        KMeans.from_sklearn(sklearn_model)


def test_kmeans_from_sklearn_array_init_raises() -> None:
    """An array ``init`` has no equivalent here and is rejected."""
    from sklearn.cluster import KMeans as SklearnKMeans

    sklearn_model = SklearnKMeans(n_clusters=2, init=np.zeros((2, 5)))
    with pytest.raises(ValueError, match="Unsupported scikit-learn init"):
        KMeans.from_sklearn(sklearn_model)


def test_kmeans_from_sklearn_random_state_instance_raises() -> None:
    """A ``RandomState`` instance cannot be mapped to SciPy's ``rng``."""
    from sklearn.cluster import KMeans as SklearnKMeans

    sklearn_model = SklearnKMeans(n_clusters=2, random_state=np.random.RandomState(0))
    with pytest.raises(TypeError, match="random_state"):
        KMeans.from_sklearn(sklearn_model)


def test_kmeans_from_sklearn_wrong_type_raises() -> None:
    """``from_sklearn`` rejects objects that are not the expected estimator."""
    with pytest.raises(TypeError, match="sklearn.cluster.KMeans"):
        KMeans.from_sklearn(object())


def test_kmeans_from_dict_wrong_class_raises() -> None:
    """``from_dict`` rejects dictionaries serialised by other model types."""
    with pytest.raises(ValueError, match="expects a serialised"):
        KMeans.from_dict({"__class__": "PCA", "state": {}})


def test_kmeans_from_dict_missing_centers_raises() -> None:
    """``from_dict`` rejects states without stored centroids."""
    with pytest.raises(ValueError, match="cluster_centers_"):
        KMeans.from_dict({"__class__": "KMeans", "state": {"n_clusters": 3}})
