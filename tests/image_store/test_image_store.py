"""Tests for :class:`pyvisim.image_store.ImageEncodingMap`."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from pyvisim.image_store import ImageEncodingMap

#: Number of encodings materialized for the test map.
NUM_IMAGES = 4
#: Dimensionality of each test encoding vector.
DIM = 6


@pytest.fixture
def encodings() -> dict[str, np.ndarray]:
    """A ``path -> vector`` mapping of precomputed encodings.

    :returns: an ordered dict of four equal-length random vectors.
    """
    rng = np.random.default_rng(0)
    return {
        f"img_{index}.png": rng.random(DIM).astype(np.float32)
        for index in range(NUM_IMAGES)
    }


@pytest.fixture
def image_map(encodings: dict[str, np.ndarray]) -> ImageEncodingMap:
    """An :class:`ImageEncodingMap` over the precomputed encodings.

    :param encodings: the ``path -> vector`` fixture.
    :returns: an ``ImageEncodingMap`` wrapping ``encodings``.
    """
    return ImageEncodingMap(encodings)


# Construction


def test_access_returns_the_stored_vector(
    image_map: ImageEncodingMap, encodings: dict[str, np.ndarray]
) -> None:
    """Indexing a path returns the vector it was constructed with."""
    path = next(iter(encodings))
    assert np.allclose(image_map[path], encodings[path])


def test_constructor_has_no_encoder_parameter() -> None:
    """The map is decoupled from any encoder, so it takes no encoder."""
    params = inspect.signature(ImageEncodingMap.__init__).parameters
    assert "encoder" not in params


def test_constructor_has_no_cache_size_parameter() -> None:
    """The removed buffer cap must not reappear on the constructor."""
    params = inspect.signature(ImageEncodingMap.__init__).parameters
    assert "max_cache_size" not in params


# Mapping protocol


def test_iter_preserves_insertion_order(
    image_map: ImageEncodingMap, encodings: dict[str, np.ndarray]
) -> None:
    """Iteration yields the registered paths in insertion order."""
    assert list(image_map) == list(encodings)


def test_dict_conversion_round_trips_keys(
    image_map: ImageEncodingMap, encodings: dict[str, np.ndarray]
) -> None:
    """``dict(map)`` materializes every encoding keyed by its path."""
    as_dict = dict(image_map)
    assert set(as_dict) == set(encodings)
    assert all(vector.ndim == 1 for vector in as_dict.values())


def test_contains_reports_registered_paths(
    image_map: ImageEncodingMap, encodings: dict[str, np.ndarray]
) -> None:
    """Membership tests report whether a path was registered."""
    assert next(iter(encodings)) in image_map
    assert "not-a-registered-path" not in image_map


def test_empty_map_has_no_paths() -> None:
    """An ``ImageEncodingMap`` may be created with no encodings at all."""
    store = ImageEncodingMap()
    assert len(store) == 0
    assert list(store) == []


# Key validation


def test_non_string_path_raises_type_error() -> None:
    """A non-string path is rejected at construction time."""
    with pytest.raises(TypeError, match="Image paths must be strings"):
        ImageEncodingMap({123: np.zeros(DIM)})  # type: ignore[dict-item]


def test_unknown_path_raises_key_error(image_map: ImageEncodingMap) -> None:
    """Indexing an unregistered path raises ``KeyError``."""
    with pytest.raises(KeyError, match="Unknown image path"):
        image_map["never-registered.png"]


def test_non_string_key_raises_key_error(image_map: ImageEncodingMap) -> None:
    """Indexing with a non-string key raises ``KeyError``."""
    with pytest.raises(KeyError, match="must be a path string"):
        image_map[42]  # type: ignore[index]


# Persistence: save_to_disk / load_from_disk


def test_save_and_load_round_trip(
    image_map: ImageEncodingMap,
    encodings: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    """A saved map reloads with identical encodings."""
    target = str(tmp_path / "store.safetensors")
    image_map.save_to_disk(target)

    loaded = ImageEncodingMap.load_from_disk(target)
    assert list(loaded) == list(image_map)
    for path, vector in encodings.items():
        assert np.allclose(loaded[path], vector)


def test_save_empty_store_raises(tmp_path: Path) -> None:
    """Saving a map with no encodings raises ``ValueError``."""
    store = ImageEncodingMap()
    with pytest.raises(ValueError, match="empty ImageStore"):
        store.save_to_disk(str(tmp_path / "empty.safetensors"))


def test_save_inconsistent_lengths_raises(tmp_path: Path) -> None:
    """Saving encodings of differing lengths raises ``ValueError``."""
    store = ImageEncodingMap({"a.png": np.zeros(3), "b.png": np.zeros(4)})
    with pytest.raises(ValueError, match="same length"):
        store.save_to_disk(str(tmp_path / "ragged.safetensors"))


def test_save_to_missing_directory_raises(image_map: ImageEncodingMap) -> None:
    """Saving into a non-existent directory raises ``OSError``."""
    with pytest.raises(OSError, match="directory does not exist"):
        image_map.save_to_disk("/no/such/directory/store.safetensors")


def test_load_missing_file_raises(tmp_path: Path) -> None:
    """Loading a path that does not exist raises ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError, match="No such safetensors file"):
        ImageEncodingMap.load_from_disk(str(tmp_path / "absent.safetensors"))


def test_load_file_missing_tensor_raises(tmp_path: Path) -> None:
    """Loading a safetensors file without the encodings tensor raises ``ValueError``."""
    target = tmp_path / "wrong.safetensors"
    save_file(
        {"unrelated": np.zeros(3, dtype=np.float32)},
        str(target),
        metadata={"paths": json.dumps([])},
    )
    with pytest.raises(ValueError, match="missing the 'encodings' tensor"):
        ImageEncodingMap.load_from_disk(str(target))


def test_load_file_missing_paths_metadata_raises(tmp_path: Path) -> None:
    """Loading a safetensors file without the paths metadata raises ``ValueError``."""
    target = tmp_path / "no_paths.safetensors"
    save_file({"encodings": np.zeros((1, 3), dtype=np.float32)}, str(target))
    with pytest.raises(ValueError, match="missing the 'paths' metadata"):
        ImageEncodingMap.load_from_disk(str(target))
