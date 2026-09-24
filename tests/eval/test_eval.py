"""Tests for :func:`pyvisim.eval.top_k_map` and :func:`pyvisim.eval.top_k_accuracy`."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pyvisim.classic import VLADEmbedder
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
    learned_vlad_embedder: VLADEmbedder,
) -> tuple[InMemoryImageEmbeddingStore, dict[str, int]]:
    """A store over labelled gallery images and the path-to-label mapping.

    :param tmp_path_factory: pytest's session temp-directory factory.
    :param category_train_images: per-category training images.
    :param learned_vlad_embedder: a fitted VLAD embedder.
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
    store = InMemoryImageEmbeddingStore(paths, learned_vlad_embedder, lazy_build=False)
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


class _FixedQueryEmbedder:
    """Embeds every query to the same vector along the first axis."""

    def embed(self, images: object) -> np.ndarray:
        """Return the fixed query embedding.

        :param images: the query images, ignored.
        :returns: a ``(1, 2)`` embedding.
        """
        return np.array([[1.0, 0.0]])


class _RankedStore:
    """A gallery the fixed query ranks as ``a, b, c, d, e``."""

    paths = ["a", "b", "c", "d", "e"]
    embeddings = np.array([[1.0, 0.0], [1.0, 0.1], [1.0, 0.2], [1.0, 0.3], [1.0, 0.4]])
    embedder = _FixedQueryEmbedder()


#: Labels of the ranked gallery: label 0 sits at ranks 2, 4 and 5.
RANKED_LABELS = {"a": 1, "b": 0, "c": 1, "d": 0, "e": 0}


@pytest.mark.parametrize(
    ("k", "expected"),
    [
        (None, (1 / 2 + 2 / 4 + 3 / 5) / 3),
        (2, (1 / 2) / 2),
        (5, (1 / 2 + 2 / 4 + 3 / 5) / 3),
    ],
)
def test_top_k_map_divides_by_the_relevant_gallery_images(
    k: int | None, expected: float
) -> None:
    """AP@k divides by ``min(R, k)``, with ``R`` counted over the whole gallery."""
    query = [np.zeros((4, 4), dtype=np.uint8)]
    score = top_k_map(query, [0], _RankedStore(), RANKED_LABELS, k=k)  # type: ignore[arg-type]
    assert score == pytest.approx(expected)
