"""Tests for the encoder base class and its similarity-function helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image
from sklearn.exceptions import NotFittedError

from pyvisim.clustering import PCA
from pyvisim.encoders import FisherVectorEncoder, VLADEncoder
from pyvisim.encoders._base_encoder import ImageEncoderBase
from pyvisim.features import Lambda, RootSIFT
from pyvisim.image_store import ImageEncodingMap

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


def _write_rgb_image(array: np.ndarray, path: Path) -> str:
    """Write a grayscale array out as an RGB PNG and return its path string.

    :param array: source grayscale image array.
    :param path: destination file path.
    :returns: the saved file path as a string.
    """
    rgb = np.stack([array, array, array], axis=-1)
    Image.fromarray(rgb).save(path)
    return str(path)


def test_generate_encoding_map_drops_duplicate_paths(
    learned_vlad: VLADEncoder,
    tmp_path: Path,
    category_train_images_flat: list[np.ndarray],
) -> None:
    """Duplicate input paths collapse to a single registered entry."""
    path = _write_rgb_image(category_train_images_flat[0], tmp_path / "img.png")
    encoding_map = learned_vlad.generate_encoding_map([path, path])
    assert list(encoding_map) == [path]


def test_generate_encoding_map_non_string_path_raises(
    learned_vlad: VLADEncoder,
) -> None:
    """A non-string path is rejected before any encoding happens."""
    with pytest.raises(TypeError, match="Image paths must be strings"):
        learned_vlad.generate_encoding_map([123])  # type: ignore[list-item]


def test_generate_encoding_map_missing_file_raises(
    learned_vlad: VLADEncoder, tmp_path: Path
) -> None:
    """A missing image file raises ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError):
        learned_vlad.generate_encoding_map([str(tmp_path / "gone.png")])


def test_generate_encoding_map_unreadable_file_raises(
    learned_vlad: VLADEncoder, tmp_path: Path
) -> None:
    """A file that is not a valid image raises ``ValueError``."""
    bogus = tmp_path / "bogus.png"
    bogus.write_text("this is not an image")
    with pytest.raises(ValueError, match="Could not read image"):
        learned_vlad.generate_encoding_map([str(bogus)])


def test_generate_encoding_map_skip_errors_warns_and_keeps_good_images(
    learned_vlad: VLADEncoder,
    tmp_path: Path,
    category_train_images_flat: list[np.ndarray],
) -> None:
    """``skip_errors`` warns about and omits images that fail to encode."""
    good = _write_rgb_image(category_train_images_flat[0], tmp_path / "good.png")
    missing = str(tmp_path / "missing.png")
    with pytest.warns(FutureWarning, match="Skipped 1 image"):
        encoding_map = learned_vlad.generate_encoding_map(
            [good, missing], skip_errors=True
        )
    assert set(encoding_map) == {good}


def test_repr_smoke(learned_vlad: VLADEncoder) -> None:
    """``repr`` names the encoder class and its RootSIFT feature extractor."""
    text = repr(learned_vlad)
    assert "VLADEncoder(" in text
    assert "feature_extractor=RootSIFT" in text
