"""Tests for :func:`pyvisim.eval.top_k_map` and :func:`pyvisim.eval.top_k_accuracy`."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pyvisim.encoders import VLADEncoder
from pyvisim.eval import top_k_accuracy, top_k_map
from pyvisim.image_store import InMemoryImageEmbeddingStore


def _to_rgb(image: np.ndarray) -> np.ndarray:
    """Stack a grayscale image into a 3-channel RGB array.

    :param image: a ``(H, W)`` grayscale array.
    :returns: a ``(H, W, 3)`` RGB array.
    """
    return np.stack([image, image, image], axis=-1)


@pytest.fixture(scope="module")
def labelled_store(
    tmp_path_factory: pytest.TempPathFactory,
    category_train_images: dict[str, list[np.ndarray]],
    learned_vlad_encoder: VLADEncoder,
) -> tuple[InMemoryImageEmbeddingStore, dict[str, int]]:
    """A store over labelled gallery images and the path-to-label mapping.

    :param tmp_path_factory: pytest's session temp-directory factory.
    :param category_train_images: per-category training images.
    :param learned_vlad_encoder: a fitted VLAD encoder.
    :returns: a ``(store, path_labels)`` pair.
    """
    directory = tmp_path_factory.mktemp("eval_gallery")
    paths: list[str] = []
    path_labels: dict[str, int] = {}
    for label, images in enumerate(category_train_images.values()):
        for offset, image in enumerate(images):
            path = directory / f"c{label}_{offset}.png"
            Image.fromarray(_to_rgb(image)).save(path)
            paths.append(str(path))
            path_labels[str(path)] = label
    store = InMemoryImageEmbeddingStore(
        paths,
        learned_vlad_encoder,
        index_params={"nlist": 4, "nprobe": 4},
    )
    return store, path_labels


@pytest.fixture(scope="module")
def queries(
    category_query_images: dict[str, list[np.ndarray]],
) -> tuple[list[np.ndarray], list[int]]:
    """Held-out RGB query images and their labels.

    :param category_query_images: held-out per-category images.
    :returns: a ``(images, labels)`` pair of equal length.
    """
    images: list[np.ndarray] = []
    labels: list[int] = []
    for label, category_images in enumerate(category_query_images.values()):
        for image in category_images:
            images.append(_to_rgb(image))
            labels.append(label)
    return images, labels


def test_top_k_map_returns_value_in_unit_range(
    labelled_store: tuple[InMemoryImageEmbeddingStore, dict[str, int]],
    queries: tuple[list[np.ndarray], list[int]],
) -> None:
    """mAP is a float in ``[0, 1]``."""
    store, path_labels = labelled_store
    images, labels = queries
    score = top_k_map(images, labels, store, path_labels, k=5)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_top_k_accuracy_returns_value_in_unit_range(
    labelled_store: tuple[InMemoryImageEmbeddingStore, dict[str, int]],
    queries: tuple[list[np.ndarray], list[int]],
) -> None:
    """Top-k accuracy is a float in ``[0, 1]``."""
    store, path_labels = labelled_store
    images, labels = queries
    score = top_k_accuracy(images, labels, store, path_labels, k=5)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_top_k_accuracy_perfect_for_gallery_queries(
    labelled_store: tuple[InMemoryImageEmbeddingStore, dict[str, int]],
    category_train_images: dict[str, list[np.ndarray]],
) -> None:
    """Querying with the gallery images themselves recovers their own label."""
    store, path_labels = labelled_store
    images = [_to_rgb(next(iter(category_train_images.values()))[0])]
    score = top_k_accuracy(images, [0], store, path_labels, k=1)
    assert score == 1.0
