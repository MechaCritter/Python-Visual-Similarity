"""Tests for the encoder base class and its similarity-function helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from pyvisim.clustering import PCA
from pyvisim.encoders import FisherVectorEncoder, VLADEncoder
from pyvisim.encoders._base_encoder import ClusteringBasedEncoder
from pyvisim.features import Lambda, RootSIFT

if TYPE_CHECKING:
    from tests.conftest import ImageObj


# §3.1 similarity-function selection


def test_similarity_func_defaults_to_cosine() -> None:
    """Encoders default to the cosine similarity metric."""
    encoder = VLADEncoder(n_clusters=8)
    assert encoder.similarity_func_name == "cosine"


@pytest.mark.parametrize("name", ["cosine", "euclidean", "l1", "manhattan"])
def test_similarity_func_accepts_supported_metrics(name: str) -> None:
    """Each supported metric name resolves to a callable."""
    encoder = VLADEncoder(n_clusters=8, similarity_func=name)
    assert encoder.similarity_func_name == name
    assert callable(encoder.similarity_func)


def test_similarity_func_rejects_custom_function() -> None:
    """Passing a custom callable is no longer supported and raises ``ValueError``."""
    with pytest.raises(ValueError, match="Unsupported similarity function"):
        VLADEncoder(n_clusters=8, similarity_func=lambda a, b: a)  # type: ignore[arg-type]


def test_similarity_func_rejects_unknown_name() -> None:
    """An unknown metric name raises ``ValueError``."""
    with pytest.raises(ValueError, match="Unsupported similarity function"):
        VLADEncoder(n_clusters=8, similarity_func="dot")


# §3.2 ImageEncoderBase shared behaviour (exercised via VLADEncoder)


class _NoModelEncoder(ClusteringBasedEncoder):
    """Minimal concrete encoder used to test the "no clustering model" path."""

    def encode(self, images: Iterable[np.ndarray], flatten: bool = True) -> np.ndarray:
        """Unused stub; ``learn`` fails before encoding is ever reached."""
        raise NotImplementedError


@pytest.fixture(scope="module")
def learned_vlad(category_train_images_flat: list[np.ndarray]) -> VLADEncoder:
    """A ``VLADEncoder(n_clusters=8)`` learned once for this module.

    :param category_train_images_flat: flattened training images to learn from.
    :returns: a fitted ``VLADEncoder``.
    """
    encoder = VLADEncoder(n_clusters=8, kmeans_params={"random_state": 0, "n_init": 3})
    encoder.learn(category_train_images_flat)
    return encoder


def test_default_feature_extractor_is_rootsift() -> None:
    """Encoders default to the RootSIFT feature extractor."""
    assert isinstance(VLADEncoder(n_clusters=8).feature_extractor, RootSIFT)


def test_feature_extractor_setter_type_check() -> None:
    """Assigning a non-extractor to ``feature_extractor`` raises ``TypeError``."""
    encoder = VLADEncoder(n_clusters=8)
    with pytest.raises(TypeError):
        encoder.feature_extractor = "x"  # type: ignore[assignment]


def test_pca_is_read_only() -> None:
    """``pca`` is a read-only property; direct assignment raises ``AttributeError``."""
    encoder = VLADEncoder(n_clusters=8)
    with pytest.raises(AttributeError):
        encoder.pca = object()  # type: ignore[assignment]


def test_learn_without_clustering_model_raises(
    category_train_images_flat: list[np.ndarray],
) -> None:
    """Learning with no clustering model raises ``RuntimeError``."""
    encoder = _NoModelEncoder(clustering_model=None)
    with pytest.raises(RuntimeError, match="no clustering model"):
        encoder.learn(category_train_images_flat)


def test_learn_bad_dim_reduction_factor_zero(
    category_train_images_flat: list[np.ndarray],
) -> None:
    """A zero ``dim_reduction_factor`` raises ``ValueError``."""
    encoder = VLADEncoder(n_clusters=8, kmeans_params={"random_state": 0, "n_init": 3})
    with pytest.raises(ValueError, match="must be a positive integer"):
        encoder.learn(category_train_images_flat, dim_reduction_factor=0)


def test_learn_bad_dim_reduction_factor_negative(
    category_train_images_flat: list[np.ndarray],
) -> None:
    """A negative ``dim_reduction_factor`` raises ``ValueError``."""
    encoder = VLADEncoder(n_clusters=8, kmeans_params={"random_state": 0, "n_init": 3})
    with pytest.raises(ValueError, match="must be a positive integer"):
        encoder.learn(category_train_images_flat, dim_reduction_factor=-2)


def test_learn_with_dim_reduction_factor_builds_pca(
    category_train_images_flat: list[np.ndarray],
) -> None:
    """A ``dim_reduction_factor`` builds and fits a PCA halving the dimension."""
    encoder = VLADEncoder(n_clusters=8, kmeans_params={"random_state": 0, "n_init": 3})
    encoder.learn(category_train_images_flat, dim_reduction_factor=2)
    assert isinstance(encoder.pca, PCA)
    assert encoder.pca.is_fitted is True
    assert encoder.pca.n_components == 64  # RootSIFT dim 128 // 2


def test_feature_extractor_pca_dim_mismatch_raises(
    category_train_images_flat: list[np.ndarray],
) -> None:
    """Assigning an extractor whose output dim mismatches the fitted PCA raises."""
    encoder = VLADEncoder(n_clusters=8, kmeans_params={"random_state": 0, "n_init": 3})
    encoder.learn(category_train_images_flat, dim_reduction_factor=2)  # PCA in=128
    with pytest.raises(RuntimeError):
        encoder.feature_extractor = Lambda(
            lambda image: np.ones((5, 64), np.float32), output_dim=64
        )


def test_save_unfitted_raises(tmp_path: Path) -> None:
    """Saving an unfitted encoder raises ``NotFittedError``."""
    encoder = VLADEncoder(n_clusters=8)
    with pytest.raises(NotFittedError, match="not fitted"):
        encoder.save_to_disk(tmp_path / "m")


def test_save_appends_suffix(learned_vlad: VLADEncoder, tmp_path: Path) -> None:
    """Saving appends the ``.encoder`` suffix and writes the file."""
    path = learned_vlad.save_to_disk(tmp_path / "model")
    assert str(path).endswith(".encoder")
    assert path.exists()


def test_save_keeps_existing_suffix(learned_vlad: VLADEncoder, tmp_path: Path) -> None:
    """Saving with an existing ``.encoder`` suffix does not double it."""
    path = learned_vlad.save_to_disk(tmp_path / "m.encoder")
    assert path.name == "m.encoder"


def test_load_roundtrip_same_encoding(
    learned_vlad: VLADEncoder, tmp_path: Path, checkerboard_image: ImageObj
) -> None:
    """A saved-and-reloaded encoder produces the same encoding."""
    path = learned_vlad.save_to_disk(tmp_path / "model")
    loaded = VLADEncoder.load_from_disk(path)
    assert np.allclose(
        loaded.encode([checkerboard_image.array]),
        learned_vlad.encode([checkerboard_image.array]),
    )


def test_load_invalid_file_raises(tmp_path: Path) -> None:
    """Loading a file that is not a valid encoder raises ``ValueError``."""
    bad = tmp_path / "bad.encoder"
    bad.write_bytes(b"not a safetensors file")
    with pytest.raises(ValueError, match="not a valid .encoder file"):
        VLADEncoder.load_from_disk(bad)


def test_load_wrong_class_raises(learned_vlad: VLADEncoder, tmp_path: Path) -> None:
    """Loading a VLAD file with the Fisher loader raises ``ValueError``."""
    path = learned_vlad.save_to_disk(tmp_path / "model")
    with pytest.raises(ValueError, match="was saved by VLADEncoder"):
        FisherVectorEncoder.load_from_disk(path)


def test_repr_smoke(learned_vlad: VLADEncoder) -> None:
    """``repr`` names the encoder class and its RootSIFT feature extractor."""
    text = repr(learned_vlad)
    assert "VLADEncoder(" in text
    assert "feature_extractor=RootSIFT" in text
