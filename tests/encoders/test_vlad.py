"""Tests for :class:`pyvisim.encoders.VLADEncoder`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from sklearn.exceptions import NotFittedError

from pyvisim.clustering import GaussianMixtureModel, KMeans
from pyvisim.encoders import GMMWeights, VLADEncoder

if TYPE_CHECKING:
    from tests.conftest import ImageObj

#: VLAD output width for ``n_clusters=8`` and RootSIFT (128-d): ``8 * 128``.
VLAD_DIM_NO_PCA = 8 * 128
#: VLAD output width with PCA to 32 dimensions: ``8 * 32``.
VLAD_DIM_PCA = 8 * 32


@pytest.fixture(scope="module")
def vlad_no_pca(category_train_images_flat: list[np.ndarray]) -> VLADEncoder:
    """A learned ``VLADEncoder(n_clusters=8)`` without PCA.

    :param category_train_images_flat: flattened training images to learn from.
    :returns: a fitted ``VLADEncoder``.
    """
    encoder = VLADEncoder(n_clusters=8, kmeans_params={"random_state": 0, "n_init": 3})
    encoder.learn(category_train_images_flat)
    return encoder


@pytest.fixture(scope="module")
def vlad_pca(category_train_images_flat: list[np.ndarray]) -> VLADEncoder:
    """A learned ``VLADEncoder(n_clusters=8)`` with PCA to 32 dimensions.

    :param category_train_images_flat: flattened training images to learn from.
    :returns: a fitted ``VLADEncoder``.
    """
    encoder = VLADEncoder(
        n_clusters=8,
        kmeans_params={"random_state": 0, "n_init": 3},
        pca_params={"n_components": 32, "random_state": 0},
    )
    encoder.learn(category_train_images_flat)
    return encoder


def test_vlad_clustering_model_type() -> None:
    """VLAD builds a KMeans clustering model."""
    assert isinstance(VLADEncoder(n_clusters=8).clustering_model, KMeans)


def test_vlad_n_clusters_param_kwargs_collision() -> None:
    """Passing ``n_clusters`` inside ``kmeans_params`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="Pass 'n_clusters' directly"):
        VLADEncoder(kmeans_params={"n_clusters": 8})


def test_vlad_wrong_weights_class_raises() -> None:
    """Passing GMM weights to VLAD raises ``ValueError``."""
    with pytest.raises(ValueError, match="only pass an instance of KMeansWeights"):
        VLADEncoder(weights=GMMWeights.OXFORD102_K256_SIFT)


def test_vlad_clustering_setter_rejects_non_kmeans() -> None:
    """Assigning a non-KMeans clustering model raises ``ValueError``."""
    encoder = VLADEncoder(n_clusters=8)
    with pytest.raises(
        ValueError, match="must be an instance of pyvisim.clustering.KMeans"
    ):
        encoder.clustering_model = GaussianMixtureModel(8)


def test_vlad_encode_shape_no_pca(
    vlad_no_pca: VLADEncoder, checkerboard_image: ImageObj
) -> None:
    """Without PCA, a single image encodes to ``(1, n_clusters * 128)``."""
    out = vlad_no_pca.encode([checkerboard_image.array])
    assert out.shape == (1, VLAD_DIM_NO_PCA)


def test_vlad_encode_batch_shape(
    vlad_no_pca: VLADEncoder, checkerboard_image: ImageObj
) -> None:
    """A batch of three images encodes to ``(3, n_clusters * 128)``."""
    base = checkerboard_image.array
    batch = [base, np.roll(base, 8, axis=0), np.roll(base, 8, axis=1)]
    assert vlad_no_pca.encode(batch).shape == (3, VLAD_DIM_NO_PCA)


def test_vlad_encode_shape_with_pca(
    vlad_pca: VLADEncoder, checkerboard_image: ImageObj
) -> None:
    """With PCA to 32 dims, a single image encodes to ``(1, n_clusters * 32)``."""
    out = vlad_pca.encode([checkerboard_image.array])
    assert out.shape == (1, VLAD_DIM_PCA)


def test_vlad_encode_rows_l2_normalized(
    vlad_no_pca: VLADEncoder, checkerboard_image: ImageObj
) -> None:
    """Each non-zero cluster row of the encoding has unit L2 norm."""
    out = vlad_no_pca.encode([checkerboard_image.array]).reshape(8, 128)
    norms = np.linalg.norm(out, axis=1)
    non_zero = norms[norms > 1e-6]
    assert non_zero == pytest.approx(np.ones_like(non_zero), rel=1e-3)


def test_vlad_encode_single_rgb_image_ok(
    vlad_no_pca: VLADEncoder, rgb_image: ImageObj
) -> None:
    """A bare 3-D RGB array is treated as a single image."""
    assert vlad_no_pca.encode(rgb_image.array).shape == (1, VLAD_DIM_NO_PCA)


def test_vlad_encode_single_2d_must_be_wrapped(
    vlad_no_pca: VLADEncoder, checkerboard_image: ImageObj
) -> None:
    """A single 2-D image must be wrapped in a list; a bare 2-D array iterates rows."""
    gray = checkerboard_image.array
    assert vlad_no_pca.encode([gray]).shape == (1, VLAD_DIM_NO_PCA)
    with pytest.raises(ValueError):
        vlad_no_pca.encode(gray)


def test_vlad_encode_no_descriptors_raises(
    vlad_no_pca: VLADEncoder, solid_image: ImageObj
) -> None:
    """Encoding a featureless image raises a clear ``ValueError``."""
    with pytest.raises(ValueError, match="No descriptors found in the image"):
        vlad_no_pca.encode([solid_image.array])


def test_vlad_encode_accepts_tensor(
    vlad_no_pca: VLADEncoder, checkerboard_image: ImageObj
) -> None:
    """A grayscale torch tensor image is accepted and encodes like its array."""
    tensor = torch.from_numpy(checkerboard_image.array)
    assert vlad_no_pca.encode([tensor]).shape == (1, VLAD_DIM_NO_PCA)


def test_vlad_predict_before_learn_raises(checkerboard_image: ImageObj) -> None:
    """Encoding before learning raises ``NotFittedError`` (KMeans not fitted)."""
    with pytest.raises(NotFittedError):
        VLADEncoder(n_clusters=8).encode([checkerboard_image.array])
