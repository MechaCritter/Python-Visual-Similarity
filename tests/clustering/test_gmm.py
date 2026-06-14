"""Tests for :class:`pyvisim.clustering.GaussianMixtureModel`."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.mixture import GaussianMixture

from pyvisim.clustering import GaussianMixtureModel


@pytest.fixture
def fitted_gmm() -> GaussianMixtureModel:
    """A ``GaussianMixtureModel(4)`` fitted on deterministic ``(300, 5)`` data.

    :returns: a fitted :class:`pyvisim.clustering.GaussianMixtureModel`.
    """
    rng = np.random.default_rng(0)
    model = GaussianMixtureModel(4, random_state=0)
    model.fit(rng.random((300, 5)))
    return model


def test_gmm_default_covariance_is_diag() -> None:
    """The underlying sklearn model is forced to ``covariance_type='diag'``."""
    assert GaussianMixtureModel(4)._model.covariance_type == "diag"


def test_gmm_non_diag_covariance_raises() -> None:
    """Requesting a non-diagonal covariance type raises ``ValueError``."""
    with pytest.raises(ValueError, match=r"only supports covariance_type='diag'"):
        GaussianMixtureModel(4, covariance_type="full")


def test_gmm_n_clusters_equals_n_components() -> None:
    """``n_clusters`` mirrors ``n_components`` and works before fitting."""
    assert GaussianMixtureModel(6).n_clusters == 6


def test_gmm_weights_before_fit_raises() -> None:
    """Accessing ``weights`` before fitting raises ``NotFittedError``."""
    with pytest.raises(NotFittedError):
        _ = GaussianMixtureModel(4).weights


def test_gmm_means_before_fit_raises() -> None:
    """Accessing ``means`` before fitting raises ``NotFittedError``."""
    with pytest.raises(NotFittedError):
        _ = GaussianMixtureModel(4).means


def test_gmm_covariances_before_fit_raises() -> None:
    """Accessing ``covariances`` before fitting raises ``NotFittedError``."""
    with pytest.raises(NotFittedError):
        _ = GaussianMixtureModel(4).covariances


def test_gmm_predict_proba_before_fit_raises() -> None:
    """Calling ``predict_proba`` before fitting raises ``NotFittedError``."""
    rng = np.random.default_rng(0)
    with pytest.raises(NotFittedError):
        GaussianMixtureModel(4).predict_proba(rng.random((10, 5)))


def test_gmm_weights_shape_and_sum(fitted_gmm: GaussianMixtureModel) -> None:
    """Mixture weights have shape ``(k,)`` and sum to one."""
    assert fitted_gmm.weights.shape == (4,)
    assert fitted_gmm.weights.sum() == pytest.approx(1.0)


def test_gmm_means_shape(fitted_gmm: GaussianMixtureModel) -> None:
    """Component means have shape ``(k, D)``."""
    assert fitted_gmm.means.shape == (4, 5)


def test_gmm_covariances_shape(fitted_gmm: GaussianMixtureModel) -> None:
    """Diagonal covariances have shape ``(k, D)``."""
    assert fitted_gmm.covariances.shape == (4, 5)


def test_gmm_predict_proba_shape_and_rows_sum(
    fitted_gmm: GaussianMixtureModel,
) -> None:
    """``predict_proba`` returns ``(N, k)`` posteriors whose rows sum to one."""
    rng = np.random.default_rng(1)
    proba = fitted_gmm.predict_proba(rng.random((30, 5)))
    assert proba.shape == (30, 4)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_gmm_from_sklearn_non_diag_raises() -> None:
    """``_from_sklearn`` rejects a non-diagonal sklearn mixture."""
    with pytest.raises(ValueError):
        GaussianMixtureModel._from_sklearn(GaussianMixture(covariance_type="full"))


def test_gmm_from_sklearn_diag_ok() -> None:
    """``_from_sklearn`` adopts a fitted diagonal sklearn mixture."""
    rng = np.random.default_rng(0)
    sklearn_model = GaussianMixture(
        n_components=3, covariance_type="diag", random_state=0
    )
    sklearn_model.fit(rng.random((60, 5)))
    adopted = GaussianMixtureModel._from_sklearn(sklearn_model)
    assert adopted.n_clusters == 3


def test_gmm_from_sklearn_wrong_type_raises() -> None:
    """``_from_sklearn`` rejects objects that are not the expected estimator."""
    with pytest.raises(TypeError):
        GaussianMixtureModel._from_sklearn(object())
