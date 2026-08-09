"""Tests for :class:`pyvisim.encoders._clustering.PCA`."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import PCA as SklearnPCA

from pyvisim._errors import NotFittedError
from pyvisim.encoders._clustering import PCA


@pytest.fixture
def fitted_pca() -> PCA:
    """A ``PCA(4)`` fitted on deterministic ``(100, 10)`` data.

    :returns: a fitted :class:`pyvisim.encoders._clustering.PCA` instance.
    """
    rng = np.random.default_rng(0)
    model = PCA(4)
    model.fit(rng.random((100, 10)))
    return model


def test_pca_is_fitted_false_before_fit() -> None:
    """A freshly built PCA reports ``is_fitted is False``."""
    assert PCA(4).is_fitted is False


def test_pca_n_components_before_fit_raises() -> None:
    """Accessing ``n_components`` before fitting raises ``NotFittedError``."""
    with pytest.raises(NotFittedError):
        _ = PCA(4).n_components


def test_pca_n_features_in_before_fit_raises() -> None:
    """Accessing ``n_features_in`` before fitting raises ``NotFittedError``."""
    with pytest.raises(NotFittedError):
        _ = PCA(4).n_features_in


def test_pca_transform_before_fit_raises() -> None:
    """Calling ``transform`` before fitting raises ``NotFittedError``."""
    rng = np.random.default_rng(0)
    with pytest.raises(NotFittedError):
        PCA(4).transform(rng.random((20, 10)))


def test_pca_invalid_n_components_raises() -> None:
    """A non-positive ``n_components`` is rejected at construction."""
    with pytest.raises(ValueError, match="n_components"):
        PCA(0)


def test_pca_invalid_solver_raises() -> None:
    """An unknown ``svd_solver`` is rejected at construction."""
    with pytest.raises(ValueError, match="svd_solver"):
        PCA(4, svd_solver="randomized")


def test_pca_negative_tol_raises() -> None:
    """A negative ``tol`` is rejected at construction."""
    with pytest.raises(ValueError, match="tol"):
        PCA(4, tol=-1.0)


def test_pca_fit_too_few_samples_raises() -> None:
    """Fitting on a single sample raises (the sample variance is undefined)."""
    with pytest.raises(ValueError, match="at least 2 samples"):
        PCA(1).fit(np.ones((1, 5)))


def test_pca_fit_n_components_too_large_raises() -> None:
    """``n_components`` above ``min(n_samples, n_features)`` raises at fit."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="n_components"):
        PCA(11).fit(rng.random((100, 10)))


def test_pca_fit_non_2d_raises() -> None:
    """A non-2D feature matrix raises ``ValueError``."""
    with pytest.raises(ValueError, match="2D"):
        PCA(2).fit(np.ones(10))


def test_pca_arpack_full_rank_raises() -> None:
    """ARPACK needs ``n_components`` strictly below ``min(n_samples, n_features)``."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="strictly less"):
        PCA(10, svd_solver="arpack").fit(rng.random((100, 10)))


def test_pca_transform_wrong_feature_count_raises(fitted_pca: PCA) -> None:
    """Transforming features of the wrong width raises ``ValueError``."""
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="features per sample"):
        fitted_pca.transform(rng.random((20, 5)))


def test_pca_fit_sets_is_fitted(fitted_pca: PCA) -> None:
    """Fitting flips ``is_fitted`` to ``True``."""
    assert fitted_pca.is_fitted is True


def test_pca_n_components_after_fit(fitted_pca: PCA) -> None:
    """A fitted PCA exposes its requested ``n_components``."""
    assert fitted_pca.n_components == 4


def test_pca_n_features_in_after_fit(fitted_pca: PCA) -> None:
    """A fitted PCA reports the number of input features it saw."""
    assert fitted_pca.n_features_in == 10


def test_pca_transform_shape(fitted_pca: PCA) -> None:
    """``transform`` reduces the feature dimension to ``n_components``."""
    rng = np.random.default_rng(1)
    out = fitted_pca.transform(rng.random((20, 10)))
    assert out.shape == (20, 4)


def test_pca_transform_is_ndarray(fitted_pca: PCA) -> None:
    """``transform`` returns a NumPy array."""
    rng = np.random.default_rng(1)
    out = fitted_pca.transform(rng.random((20, 10)))
    assert isinstance(out, np.ndarray)


@pytest.mark.parametrize("solver", ["full", "covariance_eigh", "arpack"])
def test_pca_matches_sklearn(solver: str) -> None:
    """Every solver reproduces scikit-learn's components and projections."""
    rng = np.random.default_rng(0)
    train = rng.normal(size=(300, 12))
    test = rng.normal(size=(40, 12))
    model = PCA(5, svd_solver=solver, rng=0)
    model.fit(train)
    reference = SklearnPCA(n_components=5, svd_solver="full").fit(train)
    np.testing.assert_allclose(model.mean, reference.mean_, atol=1e-10)
    np.testing.assert_allclose(model.components, reference.components_, atol=1e-8)
    np.testing.assert_allclose(
        model.explained_variance, reference.explained_variance_, atol=1e-8
    )
    np.testing.assert_allclose(
        model.explained_variance_ratio, reference.explained_variance_ratio_, atol=1e-8
    )
    np.testing.assert_allclose(
        model.transform(test), reference.transform(test), atol=1e-8
    )


def test_pca_noise_variance_matches_sklearn() -> None:
    """The probabilistic-PCA noise estimate matches scikit-learn's."""
    rng = np.random.default_rng(3)
    data = rng.normal(size=(200, 8))
    model = PCA(3, svd_solver="full")
    model.fit(data)
    reference = SklearnPCA(n_components=3, svd_solver="full").fit(data)
    assert model.noise_variance == pytest.approx(reference.noise_variance_)


def test_pca_transform_uses_train_mean() -> None:
    """``transform`` centers with the training mean, not the batch mean."""
    rng = np.random.default_rng(2)
    train = rng.normal(size=(200, 6))
    model = PCA(3)
    model.fit(train)
    # A test batch with a very different mean: projecting it with the batch
    # mean would send the offset to ~zero; the training mean keeps it.
    shifted = rng.normal(size=(30, 6)) + 100.0
    expected = (shifted - train.mean(axis=0)) @ model.components.T
    np.testing.assert_allclose(model.transform(shifted), expected, atol=1e-8)


def test_pca_whiten_unit_variance() -> None:
    """Whitening scales the projected training data to ~unit variance."""
    rng = np.random.default_rng(0)
    data = rng.random((200, 10))
    model = PCA(4, whiten=True)
    model.fit(data)
    out = model.transform(data)
    assert np.allclose(out.var(axis=0, ddof=1), 1.0, atol=1e-8)


def test_pca_whiten_near_zero_variance_is_finite() -> None:
    """Whitening rank-deficient data stays finite (epsilon-floored scale)."""
    rng = np.random.default_rng(0)
    # Rank-2 data embedded in 6 dimensions: components 3..4 have ~zero variance.
    basis = rng.normal(size=(2, 6))
    data = rng.normal(size=(100, 2)) @ basis
    model = PCA(4, whiten=True, svd_solver="full")
    model.fit(data)
    out = model.transform(data)
    assert np.all(np.isfinite(out))


def test_pca_signs_are_deterministic_across_solvers() -> None:
    """All solvers agree on component signs (largest entry positive)."""
    rng = np.random.default_rng(4)
    data = rng.normal(size=(150, 9))
    full = PCA(3, svd_solver="full")
    full.fit(data)
    eigh = PCA(3, svd_solver="covariance_eigh")
    eigh.fit(data)
    np.testing.assert_allclose(full.components, eigh.components, atol=1e-8)


def test_pca_auto_solver_selection() -> None:
    """``svd_solver="auto"`` resolves with scikit-learn's policy."""
    rng = np.random.default_rng(0)
    tall_skinny = PCA(4)
    tall_skinny.fit(rng.random((1_000, 10)))  # n >= 10 * d, d <= 1000
    assert tall_skinny.fitted_svd_solver == "covariance_eigh"

    small = PCA(4)
    small.fit(rng.random((100, 50)))  # max dim <= 500
    assert small.fitted_svd_solver == "full"

    truncated = PCA(4, rng=0)
    truncated.fit(rng.random((600, 600)))  # k < 0.8 * min dim
    assert truncated.fitted_svd_solver == "arpack"

    near_full_rank = PCA(550)
    near_full_rank.fit(rng.random((600, 600)))  # k >= 0.8 * min dim
    assert near_full_rank.fitted_svd_solver == "full"


def test_pca_to_dict_before_fit_raises() -> None:
    """Serialising an unfitted model raises ``NotFittedError``."""
    with pytest.raises(NotFittedError):
        PCA(4).to_dict()


def test_pca_dict_roundtrip(fitted_pca: PCA) -> None:
    """``from_dict(to_dict())`` restores an identically-transforming model."""
    rng = np.random.default_rng(5)
    test = rng.random((20, 10))
    restored = PCA.from_dict(fitted_pca.to_dict())
    assert restored.is_fitted is True
    assert restored.n_components == fitted_pca.n_components
    assert restored.fitted_svd_solver == fitted_pca.fitted_svd_solver
    np.testing.assert_allclose(restored.transform(test), fitted_pca.transform(test))


def test_pca_from_dict_legacy_sklearn_state() -> None:
    """Legacy dictionaries holding a sklearn PCA state are still accepted."""
    rng = np.random.default_rng(0)
    data = rng.random((100, 10))
    reference = SklearnPCA(n_components=4, svd_solver="randomized", random_state=0)
    reference.fit(data)
    legacy = {
        "__class__": "PCA",
        "__module__": "sklearn.decomposition._pca",
        "state": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in vars(reference).items()
            if not key.startswith("_")
        },
    }
    restored = PCA.from_dict(legacy)
    assert restored.is_fitted is True
    assert restored.n_components == 4
    np.testing.assert_allclose(
        restored.transform(data), reference.transform(data), atol=1e-8
    )


def test_pca_from_dict_wrong_class_raises() -> None:
    """``from_dict`` rejects dictionaries of a different model type."""
    with pytest.raises(ValueError, match="expects a serialised"):
        PCA.from_dict({"__class__": "KMeans", "state": {}})


def test_pca_from_sklearn_adopts_fitted() -> None:
    """``from_sklearn`` adopts an already-fitted sklearn estimator."""
    rng = np.random.default_rng(0)
    data = rng.random((50, 10))
    sklearn_model = SklearnPCA(n_components=3).fit(data)
    adopted = PCA.from_sklearn(sklearn_model)
    assert adopted.is_fitted is True
    assert adopted.n_components == 3
    np.testing.assert_allclose(
        adopted.transform(data), sklearn_model.transform(data), atol=1e-8
    )


def test_pca_from_sklearn_wrong_type_raises() -> None:
    """``from_sklearn`` rejects objects that are not the expected estimator."""
    with pytest.raises(TypeError):
        PCA.from_sklearn(object())


def test_pca_from_sklearn_non_integer_n_components_raises() -> None:
    """``from_sklearn`` rejects estimators whose n_components is not an int."""
    with pytest.raises(TypeError, match="n_components"):
        PCA.from_sklearn(SklearnPCA(n_components=0.95))
