"""Tests for :class:`pyvisim.clustering.PCA`."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.exceptions import NotFittedError

from pyvisim.clustering import PCA


@pytest.fixture
def fitted_pca() -> PCA:
    """A ``PCA(4)`` fitted on deterministic ``(100, 10)`` data.

    :returns: a fitted :class:`pyvisim.clustering.PCA` instance.
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


def test_pca_forwards_params() -> None:
    """Extra params (``whiten``) are forwarded to the underlying sklearn PCA."""
    rng = np.random.default_rng(0)
    data = rng.random((200, 10))
    model = PCA(4, whiten=True, random_state=0)
    model.fit(data)
    out = model.transform(data)
    # Whitening makes each projected component have ~unit variance.
    assert np.allclose(out.var(axis=0), 1.0, atol=0.05)


def test_pca_from_sklearn_adopts_fitted() -> None:
    """``_from_sklearn`` adopts an already-fitted sklearn estimator."""
    rng = np.random.default_rng(0)
    sklearn_model = SklearnPCA(n_components=3).fit(rng.random((50, 10)))
    adopted = PCA._from_sklearn(sklearn_model)
    assert adopted.is_fitted is True
    assert adopted.n_components == 3


def test_pca_from_sklearn_wrong_type_raises() -> None:
    """``_from_sklearn`` rejects objects that are not the expected estimator."""
    with pytest.raises(TypeError):
        PCA._from_sklearn(object())
