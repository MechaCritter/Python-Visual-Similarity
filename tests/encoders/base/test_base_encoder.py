"""Tests for the encoder base class and its similarity-function helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pytest
from PIL import Image
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import cosine_similarity

from pyvisim.clustering import PCA
from pyvisim.encoders import FisherVectorEncoder, VLADEncoder
from pyvisim.encoders._base_encoder import (
    ImageEncoderBase,
    _make_fallback_func,
    check_desired_output,
)
from pyvisim.features import Lambda, RootSIFT
from pyvisim.image_store import ImageEncodingMap

if TYPE_CHECKING:
    from tests.conftest import ImageObj


# §3.1 check_desired_output / _make_fallback_func


def _list_returning_func(vecs1: np.ndarray, vecs2: np.ndarray) -> list[list[int]]:
    """A similarity function that wrongly returns a Python list."""
    return [[1, 2]]


def _wrong_shape_func(vecs1: np.ndarray, vecs2: np.ndarray) -> np.ndarray:
    """A similarity function that returns an array of the wrong shape."""
    return np.zeros((3, 3))


def _raising_func(vecs1: np.ndarray, vecs2: np.ndarray) -> np.ndarray:
    """A similarity function that always raises."""
    raise RuntimeError("boom")


def _scalar_cosine(vecs1: np.ndarray, vecs2: np.ndarray) -> float:
    """Cosine similarity of two single-row inputs reduced to a scalar.

    The row-wise fallback assigns the result to a scalar cell, so the wrapped
    function must return a scalar rather than a ``(1, 1)`` array.
    """
    return float(cosine_similarity(vecs1, vecs2)[0, 0])


def test_cdo_valid_returns_same_func() -> None:
    """A valid similarity function is returned unchanged."""
    rng = np.random.default_rng(0)
    vecs = rng.random((10, 10))
    assert check_desired_output(cosine_similarity, vecs, vecs) is cosine_similarity


def test_cdo_raising_func_falls_back() -> None:
    """A raising similarity function is replaced by a fallback, with a warning."""
    rng = np.random.default_rng(0)
    vecs = rng.random((10, 10))
    with pytest.warns(UserWarning, match="threw an error"):
        result = check_desired_output(_raising_func, vecs, vecs)
    assert result is not _raising_func


def test_cdo_non_ndarray_falls_back() -> None:
    """A function returning a non-array is replaced by a fallback, with a warning."""
    rng = np.random.default_rng(0)
    vecs = rng.random((10, 10))
    with pytest.warns(UserWarning, match="Expected a NumPy array"):
        result = check_desired_output(_list_returning_func, vecs, vecs)
    assert result is not _list_returning_func


def test_cdo_wrong_shape_falls_back() -> None:
    """A function returning the wrong shape is replaced by a fallback, with a warning."""
    rng = np.random.default_rng(0)
    vecs = rng.random((10, 10))
    with pytest.warns(UserWarning, match="not the expected"):
        result = check_desired_output(_wrong_shape_func, vecs, vecs)
    assert result is not _wrong_shape_func


def test_fallback_loops_rowwise() -> None:
    """The fallback wrapper builds an ``(N1, N2)`` float32 matrix row-wise."""
    fallback = _make_fallback_func(_scalar_cosine)
    rng = np.random.default_rng(0)
    out = fallback(rng.random((4, 5)), rng.random((3, 5)))
    assert out.shape == (4, 3)
    assert out.dtype == np.float32


# §3.2 ImageEncoderBase shared behaviour (exercised via VLADEncoder)


class _NoModelEncoder(ImageEncoderBase):
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


def test_pca_setter_type_check() -> None:
    """Assigning a non-PCA object to ``pca`` raises ``ValueError``."""
    encoder = VLADEncoder(n_clusters=8)
    with pytest.raises(
        ValueError, match="must be an instance of pyvisim.clustering.PCA"
    ):
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
    joblib.dump({"foo": 1}, bad)
    with pytest.raises(ValueError, match="not a valid .encoder file"):
        VLADEncoder.load_from_disk(bad)


def test_load_wrong_class_raises(learned_vlad: VLADEncoder, tmp_path: Path) -> None:
    """Loading a VLAD file with the Fisher loader raises ``ValueError``."""
    path = learned_vlad.save_to_disk(tmp_path / "model")
    with pytest.raises(ValueError, match="was saved by VLADEncoder"):
        FisherVectorEncoder.load_from_disk(path)


def test_generate_encoding_map_returns_image_encoding_map(
    learned_vlad: VLADEncoder,
    tmp_path: Path,
    category_train_images_flat: list[np.ndarray],
) -> None:
    """``generate_encoding_map`` returns an :class:`ImageEncodingMap`.

    The mapping exposes each registered image path and yields an equal-length
    1-D vector per path, so existing path-based access keeps working.
    """
    paths = []
    for index in (0, 1):
        gray = category_train_images_flat[index]
        rgb = np.stack([gray, gray, gray], axis=-1)
        path = str(tmp_path / f"img_{index}.png")
        Image.fromarray(rgb).save(path)
        paths.append(path)

    encoding_map = learned_vlad.generate_encoding_map(paths)
    assert isinstance(encoding_map, ImageEncodingMap)
    assert set(encoding_map) == set(paths)
    vectors = [np.asarray(vector) for vector in encoding_map.values()]
    assert all(vector.ndim == 1 for vector in vectors)
    assert len({vector.shape[0] for vector in vectors}) == 1


def test_repr_smoke(learned_vlad: VLADEncoder) -> None:
    """``repr`` names the encoder class and its RootSIFT feature extractor."""
    text = repr(learned_vlad)
    assert "VLADEncoder(" in text
    assert "feature_extractor=RootSIFT" in text
