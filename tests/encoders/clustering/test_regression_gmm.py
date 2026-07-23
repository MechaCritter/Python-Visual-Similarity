"""Regression tests for :class:`DiagCovarGaussianMixture` vs sklearn.

These mirror the quality checks of ``notebooks/benchmark_gmm.ipynb`` (the
runtime comparison is intentionally left out): the NumPy/SciPy EM
implementation must estimate the mixture as well as
:class:`sklearn.mixture.GaussianMixture` with ``covariance_type="diag"``. With
``n_init=10`` and identical ``tol`` / ``max_iter`` / ``reg_covar`` for both, the
held-out log-likelihood and the label-quality metrics stay within one
percentage point of scikit-learn, the two implementations assign held-out
samples almost identically, and the recorded EM lower bound never decreases.

Two blob shapes are exercised: a 2-D problem (fast) and a 128-dimensional
variant (marked ``slow``).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture as SklearnGaussianMixture
from sklearn.model_selection import train_test_split

from pyvisim.encoders._clustering import DiagCovarGaussianMixture

#: Number of seeded restarts kept for both implementations.
N_INIT = 10
#: Number of blob samples drawn for each configuration.
N_SAMPLES = 2_000
#: Shared EM hyper-parameters, matching the notebook.
MAX_ITER = 100
TOL = 1e-3
REG_COVAR = 1e-6
#: Largest allowed absolute gap in a label-quality metric (one percentage point).
QUALITY_ABS_TOL = 0.01
#: Minimum Adjusted Rand Index between the two implementations' assignments.
AGREEMENT_MIN = 0.98

#: ``(n_features, centers, cluster_std, seed, n_components)`` per configuration.
BLOB_CONFIGS = [
    pytest.param(2, 5, 1.1, 3, 5, id="dim2"),
    pytest.param(128, 16, 1.0, 0, 16, marks=pytest.mark.slow, id="dim128"),
]


def _fit_both(
    train: np.ndarray, n_components: int
) -> tuple[DiagCovarGaussianMixture, SklearnGaussianMixture]:
    """Fit both mixtures on the same data with matching hyper-parameters.

    :param train: the training samples.
    :param n_components: the number of mixture components to fit.
    :returns: the fitted ``(pyvisim, sklearn)`` models.
    """
    pyvisim_model = DiagCovarGaussianMixture(
        n_components,
        n_init=N_INIT,
        max_iter=MAX_ITER,
        tol=TOL,
        reg_covar=REG_COVAR,
        rng=0,
    )
    pyvisim_model.fit(train)
    sklearn_model = SklearnGaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        n_init=N_INIT,
        init_params="k-means++",
        max_iter=MAX_ITER,
        tol=TOL,
        reg_covar=REG_COVAR,
        random_state=0,
    )
    sklearn_model.fit(train)
    return pyvisim_model, sklearn_model


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize(
    ("n_features", "centers", "cluster_std", "seed", "n_components"),
    BLOB_CONFIGS,
)
def test_gmm_label_quality_matches_sklearn(
    n_features: int,
    centers: int,
    cluster_std: float,
    seed: int,
    n_components: int,
) -> None:
    """Held-out ARI/NMI/homogeneity stay within 1% of scikit-learn."""
    data, labels_true = make_blobs(
        n_samples=N_SAMPLES,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=seed,
    )
    data = data.astype(np.float64)
    train, test, _, test_labels = train_test_split(
        data, labels_true, test_size=0.25, random_state=seed
    )
    pyvisim_model, sklearn_model = _fit_both(train, n_components)
    labels_pv = pyvisim_model.predict(test)
    labels_sk = sklearn_model.predict(test)

    metrics = (
        adjusted_rand_score,
        normalized_mutual_info_score,
        homogeneity_score,
    )
    for metric in metrics:
        score_pv = metric(test_labels, labels_pv)
        score_sk = metric(test_labels, labels_sk)
        assert abs(score_pv - score_sk) <= QUALITY_ABS_TOL, (
            f"{metric.__name__}: pyvisim={score_pv:.4f} sklearn={score_sk:.4f}"
        )


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize(
    ("n_features", "centers", "cluster_std", "seed", "n_components"),
    BLOB_CONFIGS,
)
def test_gmm_held_out_log_likelihood_matches_sklearn(
    n_features: int,
    centers: int,
    cluster_std: float,
    seed: int,
    n_components: int,
) -> None:
    """The held-out mean log-likelihood matches scikit-learn's ``score``."""
    data, labels_true = make_blobs(
        n_samples=N_SAMPLES,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=seed,
    )
    data = data.astype(np.float64)
    train, test, _, _ = train_test_split(
        data, labels_true, test_size=0.25, random_state=seed
    )
    pyvisim_model, sklearn_model = _fit_both(train, n_components)
    assert pyvisim_model.score(test) == pytest.approx(
        sklearn_model.score(test), rel=0.01
    )


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize(
    ("n_features", "centers", "cluster_std", "seed", "n_components"),
    BLOB_CONFIGS,
)
def test_gmm_assignment_agrees_with_sklearn(
    n_features: int,
    centers: int,
    cluster_std: float,
    seed: int,
    n_components: int,
) -> None:
    """The two mixtures assign held-out samples almost identically."""
    data, labels_true = make_blobs(
        n_samples=N_SAMPLES,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=seed,
    )
    data = data.astype(np.float64)
    train, test, _, _ = train_test_split(
        data, labels_true, test_size=0.25, random_state=seed
    )
    pyvisim_model, sklearn_model = _fit_both(train, n_components)
    agreement = adjusted_rand_score(
        pyvisim_model.predict(test), sklearn_model.predict(test)
    )
    assert agreement >= AGREEMENT_MIN


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize(
    ("n_features", "centers", "cluster_std", "seed", "n_components"),
    BLOB_CONFIGS,
)
def test_gmm_silhouette_matches_sklearn(
    n_features: int,
    centers: int,
    cluster_std: float,
    seed: int,
    n_components: int,
) -> None:
    """The silhouette of the hard assignments tracks scikit-learn's."""
    data, _ = make_blobs(
        n_samples=N_SAMPLES,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=seed,
    )
    data = data.astype(np.float64)
    pyvisim_model, sklearn_model = _fit_both(data, n_components)
    subset = np.random.default_rng(seed).choice(len(data), 1_000, replace=False)
    silhouette_pv = silhouette_score(data[subset], pyvisim_model.predict(data[subset]))
    silhouette_sk = silhouette_score(data[subset], sklearn_model.predict(data[subset]))
    assert abs(silhouette_pv - silhouette_sk) <= QUALITY_ABS_TOL


def test_gmm_em_lower_bound_is_monotonic() -> None:
    """The recorded EM training log-likelihood never decreases (notebook §4)."""
    data, _ = make_blobs(
        n_samples=5_000, n_features=32, centers=16, cluster_std=3.0, random_state=0
    )
    data = data.astype(np.float64)
    model = DiagCovarGaussianMixture(16, max_iter=80, tol=0.0, rng=0)
    with pytest.warns(FutureWarning, match="did not converge"):
        model.fit(data)
    lower_bounds = np.asarray(model._fit_lower_bounds)
    tolerance = 1e-8 * max(1.0, float(np.abs(lower_bounds).max()))
    assert (np.diff(lower_bounds) >= -tolerance).all()
