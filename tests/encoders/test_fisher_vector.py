"""Tests for :class:`pyvisim.encoders.FisherVectorEncoder`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from sklearn.exceptions import NotFittedError

from pyvisim.clustering import GaussianMixtureModel, KMeans
from pyvisim.encoders import FisherVectorEncoder, KMeansWeights

if TYPE_CHECKING:
    from tests.conftest import ImageObj

#: Fisher output width for ``k=8`` and RootSIFT (128-d): ``2 * 8 * 128 + 8``.
FISHER_DIM_NO_PCA = 2 * 8 * 128 + 8
#: Fisher output width with PCA to 32 dimensions: ``2 * 8 * 32 + 8``.
FISHER_DIM_PCA = 2 * 8 * 32 + 8


@pytest.fixture(scope="module")
def fisher_no_pca(category_train_images_flat: list[np.ndarray]) -> FisherVectorEncoder:
    """A learned ``FisherVectorEncoder(n_components=8)`` without PCA.

    :param category_train_images_flat: flattened training images to learn from.
    :returns: a fitted ``FisherVectorEncoder``.
    """
    encoder = FisherVectorEncoder(n_components=8, gmm_params={"random_state": 0})
    encoder.learn(category_train_images_flat)
    return encoder


@pytest.fixture(scope="module")
def fisher_pca(category_train_images_flat: list[np.ndarray]) -> FisherVectorEncoder:
    """A learned ``FisherVectorEncoder(n_components=8)`` with PCA to 32 dimensions.

    :param category_train_images_flat: flattened training images to learn from.
    :returns: a fitted ``FisherVectorEncoder``.
    """
    encoder = FisherVectorEncoder(
        n_components=8,
        gmm_params={"random_state": 0},
        pca_params={"n_components": 32, "random_state": 0},
    )
    encoder.learn(category_train_images_flat)
    return encoder


def test_fisher_clustering_model_type() -> None:
    """Fisher builds a Gaussian mixture clustering model."""
    model = FisherVectorEncoder(n_components=8).clustering_model
    assert isinstance(model, GaussianMixtureModel)


def test_fisher_n_components_kwargs_collision() -> None:
    """Passing ``n_components`` inside ``gmm_params`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="Pass 'n_components' directly"):
        FisherVectorEncoder(gmm_params={"n_components": 8})


def test_fisher_wrong_weights_class_raises() -> None:
    """Passing KMeans weights to Fisher raises ``ValueError``."""
    with pytest.raises(ValueError, match="only pass an instance of GMMWeights"):
        FisherVectorEncoder(weights=KMeansWeights.OXFORD102_K256_SIFT)


def test_fisher_clustering_setter_rejects_non_gmm() -> None:
    """Assigning a non-GMM clustering model raises ``ValueError``."""
    encoder = FisherVectorEncoder(n_components=8)
    with pytest.raises(ValueError, match="GaussianMixtureModel"):
        encoder.clustering_model = KMeans(8)


def test_fisher_encode_shape_no_pca(
    fisher_no_pca: FisherVectorEncoder, checkerboard_image: ImageObj
) -> None:
    """Without PCA, a single image encodes to ``(1, 2 * k * 128 + k)``."""
    out = fisher_no_pca.encode([checkerboard_image.array])
    assert out.shape == (1, FISHER_DIM_NO_PCA)


def test_fisher_encode_shape_with_pca(
    fisher_pca: FisherVectorEncoder, checkerboard_image: ImageObj
) -> None:
    """With PCA to 32 dims, a single image encodes to ``(1, 2 * k * 32 + k)``."""
    out = fisher_pca.encode([checkerboard_image.array])
    assert out.shape == (1, FISHER_DIM_PCA)


def test_fisher_encode_batch_shape(
    fisher_no_pca: FisherVectorEncoder, checkerboard_image: ImageObj
) -> None:
    """A batch of two images encodes to ``(2, 2 * k * 128 + k)``."""
    base = checkerboard_image.array
    batch = [base, np.roll(base, 8, axis=0)]
    assert fisher_no_pca.encode(batch).shape == (2, FISHER_DIM_NO_PCA)


def test_fisher_encode_rejects_tensor(fisher_no_pca: FisherVectorEncoder) -> None:
    """Encoding a torch tensor raises ``RuntimeError``."""
    with pytest.raises(RuntimeError, match="Torch images are not supported"):
        fisher_no_pca.encode(torch.zeros(3, 64, 64))


def test_fisher_encode_no_descriptors(
    fisher_no_pca: FisherVectorEncoder, solid_image: ImageObj
) -> None:
    """Encoding a featureless image raises (GMM cannot score zero descriptors)."""
    with pytest.raises(ValueError):
        fisher_no_pca.encode([solid_image.array])


def test_fisher_before_learn_raises(checkerboard_image: ImageObj) -> None:
    """Encoding before learning raises ``NotFittedError`` (GMM not fitted)."""
    with pytest.raises(NotFittedError):
        FisherVectorEncoder(n_components=8).encode([checkerboard_image.array])
