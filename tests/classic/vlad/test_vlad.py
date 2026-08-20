"""Tests for :class:`pyvisim.classic.VLADEmbedder`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from pyvisim._errors import NotFittedError
from pyvisim.classic import VLADEmbedder
from pyvisim.classic._clustering import DiagCovarGaussianMixture, KMeans

if TYPE_CHECKING:
    from tests.conftest import ImageObj

#: VLAD output width for ``n_clusters=8`` and RootSIFT (128-d): ``8 * 128``.
VLAD_DIM_NO_PCA = 8 * 128
#: VLAD output width with PCA to 32 dimensions: ``8 * 32``.
VLAD_DIM_PCA = 8 * 32


@pytest.fixture(scope="module")
def vlad_no_pca(category_train_images_flat: list[np.ndarray]) -> VLADEmbedder:
    """A learned ``VLADEmbedder(n_clusters=8)`` without PCA.

    :param category_train_images_flat: flattened training images to learn from.
    :returns: a fitted ``VLADEmbedder``.
    """
    embedder = VLADEmbedder(n_clusters=8, kmeans_params={"rng": 0})
    embedder.learn(category_train_images_flat)
    return embedder


@pytest.fixture(scope="module")
def vlad_pca(category_train_images_flat: list[np.ndarray]) -> VLADEmbedder:
    """A learned ``VLADEmbedder(n_clusters=8)`` with PCA to 32 dimensions.

    :param category_train_images_flat: flattened training images to learn from.
    :returns: a fitted ``VLADEmbedder``.
    """
    embedder = VLADEmbedder(
        n_clusters=8,
        kmeans_params={"rng": 0},
        pca_params={"n_components": 32, "rng": 0},
    )
    embedder.learn(category_train_images_flat)
    return embedder


def test_clustering_model_type() -> None:
    """VLAD builds a KMeans clustering model."""
    assert isinstance(VLADEmbedder(n_clusters=8).clustering_model, KMeans)


def test_n_clusters_param_kwargs_collision() -> None:
    """Passing ``n_clusters`` inside ``kmeans_params`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="Pass 'n_clusters' directly"):
        VLADEmbedder(kmeans_params={"n_clusters": 8})


def test_clustering_model_is_read_only() -> None:
    """``clustering_model`` is a read-only property; direct assignment raises ``AttributeError``."""
    embedder = VLADEmbedder(n_clusters=8)
    with pytest.raises(AttributeError):
        embedder.clustering_model = DiagCovarGaussianMixture(8)  # type: ignore[assignment]


def test_embed_shape_no_pca(
    vlad_no_pca: VLADEmbedder, checkerboard_image: ImageObj
) -> None:
    """Without PCA, a single image embeds to ``(1, n_clusters * 128)``."""
    out = vlad_no_pca.embed([checkerboard_image.array])
    assert out.shape == (1, VLAD_DIM_NO_PCA)


def test_embed_batch_shape(
    vlad_no_pca: VLADEmbedder, checkerboard_image: ImageObj
) -> None:
    """A batch of three images embeds to ``(3, n_clusters * 128)``."""
    base = checkerboard_image.array
    batch = [base, np.roll(base, 8, axis=0), np.roll(base, 8, axis=1)]
    assert vlad_no_pca.embed(batch).shape == (3, VLAD_DIM_NO_PCA)


def test_embed_shape_with_pca(
    vlad_pca: VLADEmbedder, checkerboard_image: ImageObj
) -> None:
    """With PCA to 32 dims, a single image embeds to ``(1, n_clusters * 32)``."""
    out = vlad_pca.embed([checkerboard_image.array])
    assert out.shape == (1, VLAD_DIM_PCA)


def test_embed_rows_l2_normalized(
    vlad_no_pca: VLADEmbedder, checkerboard_image: ImageObj
) -> None:
    """Each non-zero cluster row of the embedding has unit L2 norm."""
    out = vlad_no_pca.embed([checkerboard_image.array]).reshape(8, 128)
    norms = np.linalg.norm(out, axis=1)
    non_zero = norms[norms > 1e-6]
    assert non_zero == pytest.approx(np.ones_like(non_zero), rel=1e-3)


def test_embed_single_rgb_image_ok(
    vlad_no_pca: VLADEmbedder, rgb_image: ImageObj
) -> None:
    """A bare 3-D RGB array is treated as a single image."""
    assert vlad_no_pca.embed(rgb_image.array).shape == (1, VLAD_DIM_NO_PCA)


def test_embed_single_2d_must_be_wrapped(
    vlad_no_pca: VLADEmbedder, checkerboard_image: ImageObj
) -> None:
    """A single 2-D image must be wrapped in a list; a bare 2-D array iterates rows."""
    gray = checkerboard_image.array
    assert vlad_no_pca.embed([gray]).shape == (1, VLAD_DIM_NO_PCA)
    with pytest.raises(ValueError):
        vlad_no_pca.embed(gray)


def test_embed_no_descriptors_raises(
    vlad_no_pca: VLADEmbedder, solid_image: ImageObj
) -> None:
    """Embedding a featureless image raises a clear ``ValueError``."""
    with pytest.raises(ValueError, match="No descriptors found in the image"):
        vlad_no_pca.embed([solid_image.array])


def test_embed_accepts_tensor(
    vlad_no_pca: VLADEmbedder, checkerboard_image: ImageObj
) -> None:
    """A grayscale torch tensor image is accepted and embeds like its array."""
    tensor = torch.from_numpy(checkerboard_image.array)
    assert vlad_no_pca.embed([tensor]).shape == (1, VLAD_DIM_NO_PCA)


def test_predict_before_learn_raises(checkerboard_image: ImageObj) -> None:
    """Embedding before learning raises ``NotFittedError`` (KMeans not fitted)."""
    with pytest.raises(NotFittedError):
        VLADEmbedder(n_clusters=8).embed([checkerboard_image.array])
