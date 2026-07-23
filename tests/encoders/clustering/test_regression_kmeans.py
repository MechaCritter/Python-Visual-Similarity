"""Regression tests for :class:`pyvisim.encoders._clustering.KMeans` vs sklearn.

These mirror the quality checks of ``notebooks/benchmark_kmeans.ipynb`` (the
runtime comparison is intentionally left out): on Gaussian blobs the
SciPy-backed :class:`~pyvisim.encoders._clustering.KMeans` must recover the
ground-truth partition just as well as :class:`sklearn.cluster.KMeans`. With
``n_init=10`` for both, the label-quality metrics stay within one percentage
point of scikit-learn and the two implementations partition the data almost
identically.

Two blob shapes are exercised: the notebook's 2-D toy problem (fast) and a
128-dimensional variant (marked ``slow``).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from pyvisim.encoders._clustering import KMeans

#: Number of seeded restarts kept for both implementations (see the notebook).
N_INIT = 10
#: Number of blob samples drawn for each configuration.
N_SAMPLES = 2_000
#: Largest allowed absolute gap in a label-quality metric (one percentage point).
QUALITY_ABS_TOL = 0.01
#: Minimum Adjusted Rand Index between the two implementations' partitions.
AGREEMENT_MIN = 0.98

#: ``(n_features, centers, cluster_std, seed, n_clusters)`` per configuration.
#: The 2-D case reuses the notebook's well-separated toy blobs; the
#: 128-D case is marked ``slow``.
BLOB_CONFIGS = [
    pytest.param(2, 5, 1.1, 3, 5, id="dim2"),
    pytest.param(128, 16, 1.0, 0, 16, marks=pytest.mark.slow, id="dim128"),
]


def _inertia(data: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> float:
    """Sum of squared distances of each sample to its assigned centroid.

    :param data: the clustered samples, shape ``(n_samples, n_features)``.
    :param centers: the cluster centroids, shape ``(n_clusters, n_features)``.
    :param labels: the assignment of each sample to a centroid.
    :returns: the k-means objective for this assignment.
    """
    return float(((data - centers[labels]) ** 2).sum())


def _fit_both(data: np.ndarray, n_clusters: int) -> tuple[KMeans, SklearnKMeans]:
    """Fit both implementations on the same data with matching parameters.

    :param data: the training samples.
    :param n_clusters: the number of clusters to fit.
    :returns: the fitted ``(pyvisim, sklearn)`` models.
    """
    pyvisim_model = KMeans(n_clusters, n_init=N_INIT, rng=0)
    pyvisim_model.fit(data)
    sklearn_model = SklearnKMeans(n_clusters=n_clusters, n_init=N_INIT, random_state=0)
    sklearn_model.fit(data)
    return pyvisim_model, sklearn_model


@pytest.mark.parametrize(
    ("n_features", "centers", "cluster_std", "seed", "n_clusters"),
    BLOB_CONFIGS,
)
def test_kmeans_label_quality_matches_sklearn(
    n_features: int,
    centers: int,
    cluster_std: float,
    seed: int,
    n_clusters: int,
) -> None:
    """Ground-truth ARI/NMI/homogeneity stay within 1% of scikit-learn."""
    data, labels_true = make_blobs(
        n_samples=N_SAMPLES,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=seed,
    )
    data = data.astype(np.float64)
    pyvisim_model, sklearn_model = _fit_both(data, n_clusters)
    labels_pv = pyvisim_model.predict(data)
    labels_sk = sklearn_model.predict(data)

    metrics = (
        adjusted_rand_score,
        normalized_mutual_info_score,
        homogeneity_score,
    )
    for metric in metrics:
        score_pv = metric(labels_true, labels_pv)
        score_sk = metric(labels_true, labels_sk)
        assert abs(score_pv - score_sk) <= QUALITY_ABS_TOL, (
            f"{metric.__name__}: pyvisim={score_pv:.4f} sklearn={score_sk:.4f}"
        )


@pytest.mark.parametrize(
    ("n_features", "centers", "cluster_std", "seed", "n_clusters"),
    BLOB_CONFIGS,
)
def test_kmeans_partition_agrees_with_sklearn(
    n_features: int,
    centers: int,
    cluster_std: float,
    seed: int,
    n_clusters: int,
) -> None:
    """The two implementations produce an almost identical partition."""
    data, _ = make_blobs(
        n_samples=N_SAMPLES,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=seed,
    )
    data = data.astype(np.float64)
    pyvisim_model, sklearn_model = _fit_both(data, n_clusters)
    agreement = adjusted_rand_score(
        pyvisim_model.predict(data), sklearn_model.predict(data)
    )
    assert agreement >= AGREEMENT_MIN


@pytest.mark.parametrize(
    ("n_features", "centers", "cluster_std", "seed", "n_clusters"),
    BLOB_CONFIGS,
)
def test_kmeans_inertia_and_silhouette_match_sklearn(
    n_features: int,
    centers: int,
    cluster_std: float,
    seed: int,
    n_clusters: int,
) -> None:
    """The k-means objective and silhouette track scikit-learn's."""
    data, _ = make_blobs(
        n_samples=N_SAMPLES,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=seed,
    )
    data = data.astype(np.float64)
    pyvisim_model, sklearn_model = _fit_both(data, n_clusters)
    labels_pv = pyvisim_model.predict(data)
    labels_sk = sklearn_model.predict(data)

    inertia_pv = _inertia(data, pyvisim_model.cluster_centers, labels_pv)
    inertia_sk = _inertia(data, sklearn_model.cluster_centers_, labels_sk)
    assert inertia_pv == pytest.approx(inertia_sk, rel=0.02)

    subset = np.random.default_rng(seed).choice(len(data), 1_000, replace=False)
    silhouette_pv = silhouette_score(data[subset], labels_pv[subset])
    silhouette_sk = silhouette_score(data[subset], labels_sk[subset])
    assert abs(silhouette_pv - silhouette_sk) <= QUALITY_ABS_TOL
