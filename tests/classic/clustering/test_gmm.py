"""Tests for :class:`pyvisim.classic._clustering.DiagCovarGaussianMixture`."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.mixture import GaussianMixture

from pyvisim._errors import NotFittedError
from pyvisim.classic._clustering import DiagCovarGaussianMixture


def _blob_data(seed: int = 0, n_samples: int = 300, n_features: int = 5) -> np.ndarray:
    """Deterministic well-separated blobs around four corners of a hypercube.

    :param seed: seed of the generator drawing the samples.
    :param n_samples: number of samples to draw.
    :param n_features: number of features per sample.
    :returns: a ``(n_samples, n_features)`` float64 matrix.
    """
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-10.0, 10.0, size=(4, n_features))
    assignments = rng.integers(4, size=n_samples)
    return centers[assignments] + rng.normal(scale=0.5, size=(n_samples, n_features))


@pytest.fixture
def fitted_gmm() -> DiagCovarGaussianMixture:
    """A ``DiagCovarGaussianMixture(4)`` fitted on deterministic ``(300, 5)`` data.

    :returns: a fitted :class:`pyvisim.classic._clustering.DiagCovarGaussianMixture`.
    """
    model = DiagCovarGaussianMixture(4, rng=0)
    model.fit(_blob_data())
    return model


# Construction and validation


def test_gmm_non_diag_covariance_raises() -> None:
    """Requesting a non-diagonal covariance type raises ``ValueError``."""
    with pytest.raises(ValueError, match=r"only supports covariance_type='diag'"):
        DiagCovarGaussianMixture(4, covariance_type="full")


def test_gmm_diag_covariance_accepted() -> None:
    """``covariance_type='diag'`` is accepted for scikit-learn compatibility."""
    assert DiagCovarGaussianMixture(4, covariance_type="diag").n_clusters == 4


def test_gmm_n_clusters_equals_n_components() -> None:
    """``n_clusters`` mirrors ``n_components`` and works before fitting."""
    assert DiagCovarGaussianMixture(6).n_clusters == 6


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_components": 0}, "n_components"),
        ({"n_init": 0}, "n_init"),
        ({"max_iter": 0}, "max_iter"),
        ({"tol": -1.0}, "tol"),
        ({"reg_covar": -1.0}, "reg_covar"),
    ],
)
def test_gmm_invalid_constructor_args_raise(kwargs: dict, match: str) -> None:
    """Non-positive counts and negative thresholds are rejected at construction."""
    kwargs.setdefault("n_components", 4)
    with pytest.raises(ValueError, match=match):
        DiagCovarGaussianMixture(**kwargs)


def test_gmm_is_fitted_false_before_fit() -> None:
    """A freshly built model reports ``is_fitted is False``."""
    assert DiagCovarGaussianMixture(4).is_fitted is False


def test_gmm_weights_before_fit_raises() -> None:
    """Accessing ``weights`` before fitting raises ``NotFittedError``."""
    with pytest.raises(NotFittedError):
        _ = DiagCovarGaussianMixture(4).weights


def test_gmm_means_before_fit_raises() -> None:
    """Accessing ``means`` before fitting raises ``NotFittedError``."""
    with pytest.raises(NotFittedError):
        _ = DiagCovarGaussianMixture(4).means


def test_gmm_covariances_before_fit_raises() -> None:
    """Accessing ``covariances`` before fitting raises ``NotFittedError``."""
    with pytest.raises(NotFittedError):
        _ = DiagCovarGaussianMixture(4).covariances


def test_gmm_predict_proba_before_fit_raises() -> None:
    """Calling ``predict_proba`` before fitting raises ``NotFittedError``."""
    rng = np.random.default_rng(0)
    with pytest.raises(NotFittedError):
        DiagCovarGaussianMixture(4).predict_proba(rng.random((10, 5)))


def test_gmm_fewer_samples_than_components_raises() -> None:
    """Fitting on fewer samples than components raises ``ValueError``."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="Cannot fit"):
        DiagCovarGaussianMixture(8, rng=0).fit(rng.random((5, 3)))


def test_gmm_empty_features_raise(fitted_gmm: DiagCovarGaussianMixture) -> None:
    """An empty feature matrix (0 samples) is rejected with a clear message."""
    with pytest.raises(ValueError, match="0 samples"):
        fitted_gmm.predict_proba(np.empty((0, 5)))


def test_gmm_non_2d_features_raise() -> None:
    """A 1-D feature array is rejected with a clear message."""
    with pytest.raises(ValueError, match="2D feature matrix"):
        DiagCovarGaussianMixture(2, rng=0).fit(np.zeros(10))


# Fitted-state shapes and invariants


def test_gmm_weights_shape_and_sum(fitted_gmm: DiagCovarGaussianMixture) -> None:
    """Mixture weights have shape ``(k,)``, are positive and sum to one."""
    assert fitted_gmm.weights.shape == (4,)
    assert fitted_gmm.weights.sum() == pytest.approx(1.0)
    assert (fitted_gmm.weights > 0).all()


def test_gmm_means_shape(fitted_gmm: DiagCovarGaussianMixture) -> None:
    """Component means have shape ``(k, D)``."""
    assert fitted_gmm.means.shape == (4, 5)


def test_gmm_covariances_shape_and_positive(
    fitted_gmm: DiagCovarGaussianMixture,
) -> None:
    """Diagonal covariances have shape ``(k, D)`` and are strictly positive."""
    assert fitted_gmm.covariances.shape == (4, 5)
    assert (fitted_gmm.covariances > 0).all()


def test_gmm_n_features_in_after_fit(fitted_gmm: DiagCovarGaussianMixture) -> None:
    """A fitted model reports the number of input features it saw."""
    assert fitted_gmm.n_features_in == 5


def test_gmm_predict_proba_shape_and_rows_sum(
    fitted_gmm: DiagCovarGaussianMixture,
) -> None:
    """``predict_proba`` returns ``(N, k)`` posteriors whose rows sum to one."""
    rng = np.random.default_rng(1)
    proba = fitted_gmm.predict_proba(rng.random((30, 5)))
    assert proba.shape == (30, 4)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_gmm_predict_matches_argmax_proba(
    fitted_gmm: DiagCovarGaussianMixture,
) -> None:
    """``predict`` returns the argmax of the posterior probabilities."""
    rng = np.random.default_rng(2)
    samples = rng.random((30, 5))
    proba = fitted_gmm.predict_proba(samples)
    assert np.array_equal(fitted_gmm.predict(samples), proba.argmax(axis=1))


def test_gmm_predict_proba_feature_mismatch_raises(
    fitted_gmm: DiagCovarGaussianMixture,
) -> None:
    """A feature count differing from training is rejected."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="Expected 5 features"):
        fitted_gmm.predict_proba(rng.random((10, 3)))


def test_gmm_rng_reproducible() -> None:
    """A fixed ``rng`` seed yields identical mixture parameters."""
    data = _blob_data()
    first = DiagCovarGaussianMixture(4, rng=0)
    second = DiagCovarGaussianMixture(4, rng=0)
    first.fit(data)
    second.fit(data)
    assert np.array_equal(first.weights, second.weights)
    assert np.array_equal(first.means, second.means)
    assert np.array_equal(first.covariances, second.covariances)


def test_gmm_fit_integer_data() -> None:
    """Integer input is cast to float64 before fitting."""
    rng = np.random.default_rng(0)
    model = DiagCovarGaussianMixture(4, rng=0)
    model.fit(rng.integers(0, 255, size=(100, 3)))
    assert model.means.shape == (4, 3)
    assert model.means.dtype == np.float64


# Numerical edge cases


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_gmm_lower_bound_monotonically_non_decreasing() -> None:
    """The training log-likelihood never decreases across EM iterations.

    A decrease would mean a bug in the EM updates (typically a sign error
    in the variance update or a stale broadcast).
    """
    data = _blob_data(seed=3, n_samples=500, n_features=8)
    model = DiagCovarGaussianMixture(6, max_iter=200, tol=0.0, rng=0)
    model.fit(data)
    lower_bounds = np.asarray(model._fit_lower_bounds)
    assert len(lower_bounds) == 200  # tol=0 never converges: all iterations ran
    tolerance = 1e-8 * max(1.0, float(np.abs(lower_bounds).max()))
    assert (np.diff(lower_bounds) >= -tolerance).all()


def test_gmm_variance_floor_on_duplicate_points() -> None:
    """Components collapsing onto duplicated points keep floored variances."""
    rng = np.random.default_rng(0)
    data = np.repeat(rng.random((5, 3)), 20, axis=0)  # 5 unique points
    model = DiagCovarGaussianMixture(4, reg_covar=1e-6, rng=0)
    model.fit(data)
    assert (model.covariances >= 1e-6).all()
    assert np.isfinite(model.covariances).all()


def test_gmm_zero_variance_feature() -> None:
    """A constant feature yields a floored variance instead of a zero division."""
    rng = np.random.default_rng(0)
    data = rng.random((200, 3))
    data[:, 1] = 7.0
    model = DiagCovarGaussianMixture(4, rng=0)
    model.fit(data)
    assert (model.covariances[:, 1] >= model._reg_covar).all()
    assert np.isfinite(model.predict_proba(data)).all()


def test_gmm_dying_component_stays_finite() -> None:
    """With more components than distinct clusters, empty components stay finite."""
    rng = np.random.default_rng(0)
    data = np.vstack([rng.normal(-5.0, 0.1, (50, 2)), rng.normal(5.0, 0.1, (50, 2))])
    model = DiagCovarGaussianMixture(8, rng=0)
    model.fit(data)
    assert np.isfinite(model.weights).all()
    assert (model.weights > 0).all()
    assert model.weights.sum() == pytest.approx(1.0)
    assert np.isfinite(model.means).all()
    assert (model.covariances > 0).all()


def test_gmm_e_step_underflow_far_samples(
    fitted_gmm: DiagCovarGaussianMixture,
) -> None:
    """Samples far from every component keep normalised, finite posteriors."""
    far = np.full((5, 5), 1e6)
    proba = fitted_gmm.predict_proba(far)
    assert np.isfinite(proba).all()
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_gmm_not_converged_warns() -> None:
    """Stopping before convergence raises a ``FutureWarning``."""
    data = _blob_data()
    model = DiagCovarGaussianMixture(4, max_iter=1, tol=0.0, rng=0)
    with pytest.warns(FutureWarning, match="did not converge"):
        model.fit(data)


def test_gmm_score_is_mean_log_likelihood(
    fitted_gmm: DiagCovarGaussianMixture,
) -> None:
    """``score`` returns a finite mean log-likelihood, higher for training data."""
    data = _blob_data()
    rng = np.random.default_rng(4)
    noise = rng.uniform(-100.0, 100.0, size=(100, 5))
    assert np.isfinite(fitted_gmm.score(data))
    assert fitted_gmm.score(data) > fitted_gmm.score(noise)


# Serialisation


def test_gmm_to_dict_from_dict_roundtrip(
    fitted_gmm: DiagCovarGaussianMixture,
) -> None:
    """A serialised model reloads with identical parameters and posteriors."""
    rng = np.random.default_rng(3)
    samples = rng.random((30, 5))
    restored = DiagCovarGaussianMixture.from_dict(fitted_gmm.to_dict())
    assert restored.n_clusters == fitted_gmm.n_clusters
    assert np.array_equal(restored.weights, fitted_gmm.weights)
    assert np.array_equal(restored.means, fitted_gmm.means)
    assert np.array_equal(restored.covariances, fitted_gmm.covariances)
    assert np.array_equal(
        restored.predict_proba(samples), fitted_gmm.predict_proba(samples)
    )


def test_gmm_to_dict_before_fit_raises() -> None:
    """Serialising an unfitted model raises ``NotFittedError``."""
    with pytest.raises(NotFittedError):
        DiagCovarGaussianMixture(4).to_dict()


def test_gmm_from_dict_legacy_sklearn_state() -> None:
    """States serialised from the old sklearn-backed model still load."""
    weights = np.full(3, 1.0 / 3.0)
    means = np.arange(15, dtype=np.float64).reshape(3, 5)
    covariances = np.ones((3, 5))

    def _array(array: np.ndarray) -> dict:
        return {
            "__ndarray__": True,
            "data": array.tolist(),
            "dtype": "float64",
            "shape": list(array.shape),
            "order": "C",
        }

    legacy = {
        "__class__": "GaussianMixture",
        "__module__": "sklearn.mixture._gaussian_mixture",
        "state": {
            "n_components": 3,
            "covariance_type": "diag",
            "tol": 0.001,
            "reg_covar": 1e-06,
            "max_iter": 100,
            "n_init": 1,
            "init_params": "kmeans",
            "random_state": 7,
            "n_features_in_": 5,
            "weights_": _array(weights),
            "means_": _array(means),
            "covariances_": _array(covariances),
        },
    }
    model = DiagCovarGaussianMixture.from_dict(legacy)
    assert model.is_fitted
    assert model.n_clusters == 3
    assert model.n_features_in == 5
    assert model._rng == 7  # legacy "random_state" maps to rng
    assert np.array_equal(model.weights, weights)
    assert np.array_equal(model.means, means)
    assert np.array_equal(model.covariances, covariances)


def test_gmm_adopted_model_matches_sklearn_predict_proba() -> None:
    """A model adopted from sklearn reproduces sklearn's posteriors."""
    rng = np.random.default_rng(0)
    data = rng.random((200, 5))
    sklearn_model = GaussianMixture(
        n_components=3, covariance_type="diag", random_state=0
    ).fit(data)
    model = DiagCovarGaussianMixture.from_sklearn(sklearn_model)
    assert np.allclose(
        model.predict_proba(data), sklearn_model.predict_proba(data), atol=1e-10
    )


def test_gmm_from_dict_non_diag_raises() -> None:
    """A legacy state with a non-diagonal covariance type is rejected."""
    with pytest.raises(ValueError, match="covariance_type='diag'"):
        DiagCovarGaussianMixture.from_dict(
            {
                "__class__": "GaussianMixture",
                "state": {"covariance_type": "full"},
            }
        )


def test_gmm_from_dict_wrong_class_raises() -> None:
    """``from_dict`` rejects dictionaries serialised by other model types."""
    with pytest.raises(ValueError, match="expects a serialised"):
        DiagCovarGaussianMixture.from_dict({"__class__": "KMeans", "state": {}})


def test_gmm_from_dict_missing_params_raises() -> None:
    """``from_dict`` rejects states without stored mixture parameters."""
    with pytest.raises(ValueError, match="weights_"):
        DiagCovarGaussianMixture.from_dict(
            {"__class__": "DiagCovarGaussianMixture", "state": {"n_components": 3}}
        )


# Conversion from scikit-learn


def test_gmm_from_sklearn_maps_params() -> None:
    """``from_sklearn`` translates the scikit-learn constructor arguments."""
    sklearn_model = GaussianMixture(
        n_components=8,
        covariance_type="diag",
        tol=1e-4,
        reg_covar=1e-5,
        max_iter=50,
        n_init=3,
        random_state=42,
    )
    model = DiagCovarGaussianMixture.from_sklearn(sklearn_model)
    assert model.n_clusters == 8
    assert model._n_init == 3
    assert model._max_iter == 50
    assert model._tol == pytest.approx(1e-4)
    assert model._reg_covar == pytest.approx(1e-5)
    assert model._rng == 42
    assert model.is_fitted is False


def test_gmm_from_sklearn_adopts_fitted_parameters() -> None:
    """``from_sklearn`` adopts the parameters of a fitted estimator."""
    rng = np.random.default_rng(0)
    data = rng.random((200, 5))
    sklearn_model = GaussianMixture(
        n_components=4, covariance_type="diag", random_state=0
    ).fit(data)
    model = DiagCovarGaussianMixture.from_sklearn(sklearn_model)
    assert model.is_fitted
    assert np.array_equal(model.weights, sklearn_model.weights_)
    assert np.array_equal(model.means, sklearn_model.means_)
    assert np.array_equal(model.covariances, sklearn_model.covariances_)


def test_gmm_from_sklearn_non_diag_raises() -> None:
    """``from_sklearn`` rejects a non-diagonal sklearn mixture."""
    with pytest.raises(ValueError, match="covariance_type='diag'"):
        DiagCovarGaussianMixture.from_sklearn(GaussianMixture(covariance_type="full"))


def test_gmm_from_sklearn_random_init_raises() -> None:
    """``init_params='random'`` is rejected; k-means++ is the only seeding."""
    sklearn_model = GaussianMixture(covariance_type="diag", init_params="random")
    with pytest.raises(ValueError, match="init_params"):
        DiagCovarGaussianMixture.from_sklearn(sklearn_model)


def test_gmm_from_sklearn_custom_init_arrays_raise() -> None:
    """Custom initial parameter arrays have no equivalent and are rejected."""
    sklearn_model = GaussianMixture(
        n_components=2, covariance_type="diag", weights_init=np.array([0.5, 0.5])
    )
    with pytest.raises(ValueError, match="weights_init"):
        DiagCovarGaussianMixture.from_sklearn(sklearn_model)


def test_gmm_from_sklearn_wrong_type_raises() -> None:
    """``from_sklearn`` rejects objects that are not the expected estimator."""
    with pytest.raises(TypeError, match="sklearn.mixture.GaussianMixture"):
        DiagCovarGaussianMixture.from_sklearn(object())


# Deprecated alias


def test_gmm_deprecated_alias_warns() -> None:
    """Importing ``GaussianMixtureModel`` warns and returns the new class."""
    import pyvisim.classic._clustering as clustering

    with pytest.warns(FutureWarning, match="renamed to DiagCovarGaussianMixture"):
        alias = clustering.GaussianMixtureModel
    assert alias is DiagCovarGaussianMixture
