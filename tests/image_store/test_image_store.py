"""Tests for :class:`pyvisim.image_store.ImageEncodingMap`."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from safetensors.numpy import save_file

from pyvisim.encoders import FisherVectorEncoder, VLADEncoder
from pyvisim.image_store import ImageEncodingMap

#: Number of images materialized on disk for the test store.
NUM_IMAGES = 4


@pytest.fixture(scope="module")
def encoder(category_train_images_flat: list[np.ndarray]) -> VLADEncoder:
    """A small ``VLADEncoder`` learned once for this module.

    :param category_train_images_flat: flattened training images to learn from.
    :returns: a fitted ``VLADEncoder`` with ``n_clusters=4``.
    """
    enc = VLADEncoder(n_clusters=4, kmeans_params={"random_state": 0, "n_init": 3})
    enc.learn(category_train_images_flat)
    return enc


@pytest.fixture(scope="module")
def image_paths(
    tmp_path_factory: pytest.TempPathFactory,
    category_train_images_flat: list[np.ndarray],
) -> list[str]:
    """Write :data:`NUM_IMAGES` RGB PNGs to disk and return their paths.

    :param tmp_path_factory: pytest's session-scoped temp directory factory.
    :param category_train_images_flat: source grayscale training images.
    :returns: a list of file-path strings, one per saved image.
    """
    directory = tmp_path_factory.mktemp("image_store_imgs")
    paths: list[str] = []
    for index in range(NUM_IMAGES):
        gray = category_train_images_flat[index]
        rgb = np.stack([gray, gray, gray], axis=-1)
        path = directory / f"img_{index}.png"
        Image.fromarray(rgb).save(path)
        paths.append(str(path))
    return paths


@pytest.fixture
def image_map(encoder: VLADEncoder, image_paths: list[str]) -> ImageEncodingMap:
    """An :class:`ImageEncodingMap` over the test images.

    :param encoder: the learned encoder fixture.
    :param image_paths: the on-disk image paths fixture.
    :returns: an ``ImageEncodingMap`` whose encodings are computed up front.
    """
    return ImageEncodingMap(encoder, image_paths)


# encoding


def test_access_returns_one_dimensional_vector(
    image_map: ImageEncodingMap, image_paths: list[str]
) -> None:
    """Indexing a path returns its flattened, 1-D encoding vector."""
    vector = image_map[image_paths[0]]
    assert vector.ndim == 1


def test_constructor_has_no_cache_size_parameter() -> None:
    """The removed buffer cap must not reappear on the constructor."""
    params = inspect.signature(ImageEncodingMap.__init__).parameters
    assert "max_cache_size" not in params


# Mapping protocol


def test_iter_preserves_insertion_order(
    image_map: ImageEncodingMap, image_paths: list[str]
) -> None:
    """Iteration yields the registered paths in insertion order."""
    assert list(image_map) == image_paths


def test_dict_conversion_round_trips_keys(
    image_map: ImageEncodingMap, image_paths: list[str]
) -> None:
    """``dict(map)`` materializes every encoding keyed by its path."""
    as_dict = dict(image_map)
    assert set(as_dict) == set(image_paths)
    assert all(vector.ndim == 1 for vector in as_dict.values())


def test_contains_reports_registered_paths(
    image_map: ImageEncodingMap, image_paths: list[str]
) -> None:
    """Membership tests report whether a path was registered."""
    assert image_paths[0] in image_map
    assert "not-a-registered-path" not in image_map


# Path registration and key validation


def test_duplicate_paths_are_dropped(
    encoder: VLADEncoder, image_paths: list[str]
) -> None:
    """Duplicate input paths collapse to a single registered entry."""
    duplicated = image_paths + image_paths
    store = ImageEncodingMap(encoder, duplicated)
    assert len(store) == NUM_IMAGES
    assert list(store) == image_paths


def test_non_string_path_raises_type_error(encoder: VLADEncoder) -> None:
    """A non-string path is rejected at registration time."""
    with pytest.raises(TypeError, match="Image paths must be strings"):
        ImageEncodingMap(encoder, [123])  # type: ignore[list-item]


def test_unknown_path_raises_key_error(image_map: ImageEncodingMap) -> None:
    """Indexing an unregistered path raises ``KeyError``."""
    with pytest.raises(KeyError, match="Unknown image path"):
        image_map["never-registered.png"]


def test_non_string_key_raises_key_error(image_map: ImageEncodingMap) -> None:
    """Indexing with a non-string key raises ``KeyError``."""
    with pytest.raises(KeyError, match="must be a path string"):
        image_map[42]  # type: ignore[index]


def test_empty_map_has_no_paths(encoder: VLADEncoder) -> None:
    """An ``ImageEncodingMap`` may be created with no paths at all."""
    store = ImageEncodingMap(encoder)
    assert len(store) == 0
    assert list(store) == []


# Encoding error propagation


def test_missing_file_raises_file_not_found(
    encoder: VLADEncoder, tmp_path: Path
) -> None:
    """A registered-but-missing file raises ``FileNotFoundError`` on construction."""
    missing = str(tmp_path / "gone.png")
    with pytest.raises(FileNotFoundError):
        ImageEncodingMap(encoder, [missing])


def test_unreadable_file_raises_value_error(
    encoder: VLADEncoder, tmp_path: Path
) -> None:
    """A registered file that is not an image raises ``ValueError`` on construction."""
    bogus = tmp_path / "bogus.png"
    bogus.write_text("this is not an image")
    with pytest.raises(ValueError, match="Could not read image"):
        ImageEncodingMap(encoder, [str(bogus)])


def test_skip_errors_warns_and_keeps_good_images(
    encoder: VLADEncoder, image_paths: list[str], tmp_path: Path
) -> None:
    """``skip_errors`` warns about and omits images that fail to encode."""
    missing = str(tmp_path / "missing.png")
    with pytest.warns(RuntimeWarning, match="Skipped 1 image"):
        store = ImageEncodingMap(encoder, [image_paths[0], missing], skip_errors=True)
    assert set(store) == {image_paths[0]}


# Persistence: save_to_disk / load_from_disk


def test_save_and_load_round_trip(
    image_map: ImageEncodingMap, encoder: VLADEncoder, tmp_path: Path
) -> None:
    """A saved store reloads with identical encodings."""
    expected = {path: image_map[path] for path in image_map}
    target = str(tmp_path / "store.safetensors")
    image_map.save_to_disk(target)

    loaded = ImageEncodingMap.load_from_disk(target, encoder)
    assert set(loaded) == set(image_map)
    assert len(loaded._encodings) == len(loaded)
    for path, vector in expected.items():
        assert np.allclose(loaded[path], vector)


def test_save_empty_store_raises(encoder: VLADEncoder, tmp_path: Path) -> None:
    """Saving a store with no images raises ``ValueError``."""
    store = ImageEncodingMap(encoder)
    with pytest.raises(ValueError, match="empty ImageStore"):
        store.save_to_disk(str(tmp_path / "empty.safetensors"))


def test_save_to_missing_directory_raises(image_map: ImageEncodingMap) -> None:
    """Saving into a non-existent directory raises ``OSError``."""
    with pytest.raises(OSError, match="directory does not exist"):
        image_map.save_to_disk("/no/such/directory/store.safetensors")


def test_load_missing_file_raises(encoder: VLADEncoder, tmp_path: Path) -> None:
    """Loading a path that does not exist raises ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError, match="No such safetensors file"):
        ImageEncodingMap.load_from_disk(str(tmp_path / "absent.safetensors"), encoder)


def test_load_file_missing_tensor_raises(encoder: VLADEncoder, tmp_path: Path) -> None:
    """Loading a safetensors file without the encodings tensor raises ``ValueError``."""
    target = tmp_path / "wrong.safetensors"
    save_file(
        {"unrelated": np.zeros(3, dtype=np.float32)},
        str(target),
        metadata={"paths": json.dumps([])},
    )
    with pytest.raises(ValueError, match="missing the 'encodings' tensor"):
        ImageEncodingMap.load_from_disk(str(target), encoder)


def test_load_file_missing_paths_metadata_raises(
    encoder: VLADEncoder, tmp_path: Path
) -> None:
    """Loading a safetensors file without the paths metadata raises ``ValueError``."""
    target = tmp_path / "no_paths.safetensors"
    save_file({"encodings": np.zeros((1, 3), dtype=np.float32)}, str(target))
    with pytest.raises(ValueError, match="missing the 'paths' metadata"):
        ImageEncodingMap.load_from_disk(str(target), encoder)


def test_load_with_mismatched_encoder_warns(
    image_map: ImageEncodingMap,
    encoder: VLADEncoder,
    category_train_images_flat: list[np.ndarray],
    tmp_path: Path,
) -> None:
    """Loading with a different encoder class warns about the mismatch."""
    target = str(tmp_path / "store.safetensors")
    image_map.save_to_disk(target)

    other = FisherVectorEncoder(n_components=4, gmm_params={"random_state": 0})
    other.learn(category_train_images_flat)
    with pytest.warns(RuntimeWarning, match="Encoder mismatch"):
        ImageEncodingMap.load_from_disk(target, other)
