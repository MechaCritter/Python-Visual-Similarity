"""Tests for :class:`pyvisim.classic.FisherVectorEmbedder`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from pyvisim._errors import NotFittedError
from pyvisim.classic import FisherVectorEmbedder
from pyvisim.classic._clustering import DiagCovarGaussianMixture, KMeans

if TYPE_CHECKING:
    from tests.conftest import ImageObj

#: Fisher output width for ``k=8`` and RootSIFT (128-d): ``2 * 8 * 128 + 8``.
FISHER_DIM_NO_PCA = 2 * 8 * 128 + 8
#: Fisher output width with PCA to 32 dimensions: ``2 * 8 * 32 + 8``.
FISHER_DIM_PCA = 2 * 8 * 32 + 8


@pytest.fixture(scope="module")
def fisher_no_pca(category_train_images_flat: list[np.ndarray]) -> FisherVectorEmbedder:
    """A learned ``FisherVectorEmbedder(n_components=8)`` without PCA.

    :param category_train_images_flat: flattened training images to learn from.
    :returns: a fitted ``FisherVectorEmbedder``.
    """
    embedder = FisherVectorEmbedder(n_components=8, gmm_params={"rng": 0})
    embedder.learn(category_train_images_flat)
    return embedder


@pytest.fixture(scope="module")
def fisher_pca(category_train_images_flat: list[np.ndarray]) -> FisherVectorEmbedder:
    """A learned ``FisherVectorEmbedder(n_components=8)`` with PCA to 32 dimensions.

    :param category_train_images_flat: flattened training images to learn from.
    :returns: a fitted ``FisherVectorEmbedder``.
    """
    embedder = FisherVectorEmbedder(
        n_components=8,
        gmm_params={"rng": 0},
        pca_params={"n_components": 32, "rng": 0},
    )
    embedder.learn(category_train_images_flat)
    return embedder


def test_clustering_model_type() -> None:
    """Fisher builds a Gaussian mixture clustering model."""
    model = FisherVectorEmbedder(n_components=8).clustering_model
    assert isinstance(model, DiagCovarGaussianMixture)


def test_n_components_kwargs_collision() -> None:
    """Passing ``n_components`` inside ``gmm_params`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="Pass 'n_components' directly"):
        FisherVectorEmbedder(gmm_params={"n_components": 8})


def test_clustering_model_is_read_only() -> None:
    """``clustering_model`` is a read-only property; direct assignment raises ``AttributeError``."""
    embedder = FisherVectorEmbedder(n_components=8)
    with pytest.raises(AttributeError):
        embedder.clustering_model = KMeans(8)  # type: ignore[assignment]


def test_embed_shape_no_pca(
    fisher_no_pca: FisherVectorEmbedder, checkerboard_image: ImageObj
) -> None:
    """Without PCA, a single image embeds to ``(1, 2 * k * 128 + k)``."""
    out = fisher_no_pca.embed([checkerboard_image.array])
    assert out.shape == (1, FISHER_DIM_NO_PCA)


def test_embed_shape_with_pca(
    fisher_pca: FisherVectorEmbedder, checkerboard_image: ImageObj
) -> None:
    """With PCA to 32 dims, a single image embeds to ``(1, 2 * k * 32 + k)``."""
    out = fisher_pca.embed([checkerboard_image.array])
    assert out.shape == (1, FISHER_DIM_PCA)


def test_embed_batch_shape(
    fisher_no_pca: FisherVectorEmbedder, checkerboard_image: ImageObj
) -> None:
    """A batch of two images embeds to ``(2, 2 * k * 128 + k)``."""
    base = checkerboard_image.array
    batch = [base, np.roll(base, 8, axis=0)]
    assert fisher_no_pca.embed(batch).shape == (2, FISHER_DIM_NO_PCA)


def test_embed_accepts_tensor(
    fisher_no_pca: FisherVectorEmbedder, checkerboard_image: ImageObj
) -> None:
    """A grayscale torch tensor image is accepted and embeds like its array."""
    tensor = torch.from_numpy(checkerboard_image.array)
    assert fisher_no_pca.embed([tensor]).shape == (1, FISHER_DIM_NO_PCA)


def test_embed_no_descriptors(
    fisher_no_pca: FisherVectorEmbedder, solid_image: ImageObj
) -> None:
    """Embedding a featureless image raises (GMM cannot score zero descriptors)."""
    with pytest.raises(ValueError):
        fisher_no_pca.embed([solid_image.array])


def test_before_learn_raises(checkerboard_image: ImageObj) -> None:
    """Embedding before learning raises ``NotFittedError`` (GMM not fitted)."""
    with pytest.raises(NotFittedError):
        FisherVectorEmbedder(n_components=8).embed([checkerboard_image.array])
