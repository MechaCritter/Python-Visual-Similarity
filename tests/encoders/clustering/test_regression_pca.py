"""Regression tests for :class:`pyvisim.encoders._clustering.PCA` vs sklearn."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA as SklearnPCA

from pyvisim.encoders._clustering import PCA

#: Solvers whose output must agree with scikit-learn's exact ``full`` solver.
SOLVERS = ["full", "covariance_eigh", "arpack"]


def _make_data(
    n_samples: int,
    n_features: int,
    effective_rank: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Gaussian data with a decaying spectrum and non-zero feature means."""
    rng = np.random.default_rng(seed)
    rank = effective_rank or min(n_samples, n_features)
    spectrum = np.exp(-np.arange(rank) / (rank / 4))
    basis = rng.normal(size=(rank, n_features))
    data = (rng.normal(size=(n_samples, rank)) * spectrum) @ basis
    data += 0.01 * rng.normal(size=(n_samples, n_features))
    data += rng.normal(size=n_features)
    return data


def _reconstruction_error(
    data: np.ndarray,
    projected: np.ndarray,
    components: np.ndarray,
    mean: np.ndarray,
) -> float:
    """Relative Frobenius error of the rank-``k`` reconstruction.

    :param data: the original samples.
    :param projected: the samples projected onto the principal subspace.
    :param components: the principal axes, shape ``(k, n_features)``.
    :param mean: the training mean subtracted before projecting.
    :returns: ``||X - X_reconstructed|| / ||X - mean||``.
    """
    reconstructed = projected @ components + mean
    return float(np.linalg.norm(data - reconstructed) / np.linalg.norm(data - mean))


def _sign_alignment(components: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Per-component sign that aligns ``components`` with ``reference``.

    :param components: the components to align, shape ``(k, n_features)``.
    :param reference: the reference components of the same shape.
    :returns: a ``(k,)`` array of +/-1 signs.
    """
    signs = np.sign(np.sum(components * reference, axis=1))
    signs[signs == 0.0] = 1.0
    return signs


@pytest.mark.parametrize("solver", SOLVERS)
def test_pca_reconstruction_error_matches_sklearn(solver: str) -> None:
    """Each solver reconstructs the data as accurately as scikit-learn."""
    data = _make_data(2_000, 256, effective_rank=128, seed=3)
    n_components = 32
    reference = SklearnPCA(n_components=n_components, svd_solver="full").fit(data)
    reference_error = _reconstruction_error(
        data, reference.transform(data), reference.components_, reference.mean_
    )

    model = PCA(n_components, svd_solver=solver, rng=0)
    model.fit(data)
    model_error = _reconstruction_error(
        data, model.transform(data), model.components, model.mean
    )
    assert model_error == pytest.approx(reference_error, abs=1e-8)


@pytest.mark.parametrize("solver", SOLVERS)
def test_pca_subspace_matches_sklearn(solver: str) -> None:
    """Each solver learns the same subspace as scikit-learn's ``full``."""
    data = _make_data(2_000, 256, effective_rank=128, seed=3)
    n_components = 32
    reference = SklearnPCA(n_components=n_components, svd_solver="full").fit(data)
    model = PCA(n_components, svd_solver=solver, rng=0)
    model.fit(data)

    max_angle_deg = float(
        np.degrees(np.max(subspace_angles(model.components.T, reference.components_.T)))
    )
    assert max_angle_deg < 1e-4

    assert (
        np.max(
            np.abs(model.explained_variance_ratio - reference.explained_variance_ratio_)
        )
        < 1e-8
    )

    signs = _sign_alignment(model.components, reference.components_)
    aligned_components = model.components * signs[:, None]
    np.testing.assert_allclose(aligned_components, reference.components_, atol=1e-6)
    aligned_projection = model.transform(data) * signs
    np.testing.assert_allclose(aligned_projection, reference.transform(data), atol=1e-6)


def test_pca_whitening_unit_variance() -> None:
    """With ``whiten=True`` each projected component has ~unit variance."""
    data = _make_data(2_000, 256, effective_rank=128, seed=3)
    model = PCA(32, whiten=True, rng=0)
    model.fit(data)
    variances = model.transform(data).var(axis=0, ddof=1)
    assert np.max(np.abs(variances - 1.0)) < 1e-6


def test_pca_whitening_rank_deficient_is_finite() -> None:
    """Whitening rank-deficient data stays finite (epsilon-floored scale)."""
    rng = np.random.default_rng(0)
    low_rank = rng.normal(size=(500, 2)) @ rng.normal(size=(2, 8))
    model = PCA(4, whiten=True)
    model.fit(low_rank)
    assert np.all(np.isfinite(model.transform(low_rank)))


def test_pca_transform_uses_train_mean() -> None:
    """``transform`` centers a shifted test batch with the training mean."""
    rng = np.random.default_rng(0)
    train = rng.normal(size=(1_000, 16))
    shifted = rng.normal(size=(100, 16)) + 50.0
    model = PCA(4)
    model.fit(train)
    expected = (shifted - train.mean(axis=0)) @ model.components.T
    np.testing.assert_allclose(model.transform(shifted), expected, atol=1e-8)


@pytest.mark.slow
def test_pca_reconstruction_error_matches_sklearn_sift_scale() -> None:
    """On tall SIFT-shaped data the auto solver matches scikit-learn (slow)."""
    data = _make_data(50_000, 128, effective_rank=100, seed=3)
    n_components = 64
    reference = SklearnPCA(n_components=n_components, svd_solver="full").fit(data)
    reference_error = _reconstruction_error(
        data, reference.transform(data), reference.components_, reference.mean_
    )

    model = PCA(n_components, rng=0)
    model.fit(data)
    assert model.fitted_svd_solver == "covariance_eigh"
    model_error = _reconstruction_error(
        data, model.transform(data), model.components, model.mean
    )
    assert model_error == pytest.approx(reference_error, abs=1e-8)
