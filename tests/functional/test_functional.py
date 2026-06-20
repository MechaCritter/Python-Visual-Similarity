"""Tests for :func:`pyvisim.functional.retrieve_top_k_similar`."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pyvisim.functional import Candidate, retrieve_top_k_similar
from pyvisim.image_store import InMemoryImageEmbeddingStore
from pyvisim.typing import FloatNumpyArray, ImageInput, UInt8NumpyArray

#: Number of gallery images materialized for the tests.
NUM_GALLERY = 12


class FlattenEncoder:
    """Deterministic encoder flattening each RGB image into its pixel vector.

    Identical pixels yield identical vectors, so an image is always its own
    nearest neighbour. This keeps the retrieval assertions exact without
    depending on a learned encoder.
    """

    def encode(
        self,
        images: ImageInput,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> FloatNumpyArray:
        """Flatten one image or a batch of images into ``(N, H*W*C)`` vectors."""
        if isinstance(images, np.ndarray) and images.ndim == 4:
            batch = list(images)
        elif isinstance(images, (list, tuple)):
            batch = list(images)
        else:
            batch = [images]
        return np.stack(
            [np.asarray(image, dtype=np.float32).reshape(-1) for image in batch]
        )


@pytest.fixture
def gallery(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[list[UInt8NumpyArray], list[str], InMemoryImageEmbeddingStore]:
    """Materialize random RGB images on disk and index them into a store.

    :param tmp_path_factory: pytest's temp-directory factory.
    :returns: a tuple ``(arrays, paths, store)``.
    """
    directory = tmp_path_factory.mktemp("functional_gallery")
    rng = np.random.default_rng(0)
    arrays: list[UInt8NumpyArray] = []
    paths: list[str] = []
    for index in range(NUM_GALLERY):
        array = rng.integers(0, 256, size=(4, 4, 3), dtype=np.uint8)
        path = directory / f"g{index}.png"
        Image.fromarray(array).save(path)
        arrays.append(array)
        paths.append(str(path))
    store = InMemoryImageEmbeddingStore(
        paths,
        FlattenEncoder(),
        "ivf-flat",
        quantizer="inner_product",
        index_params={"nlist": 2, "nprobe": 2},
    )
    return arrays, paths, store


def test_returns_one_ranked_list_per_query_in_order(
    gallery: tuple[list[UInt8NumpyArray], list[str], InMemoryImageEmbeddingStore],
) -> None:
    """Each query maps to its own ranked list, in input order."""
    arrays, paths, store = gallery
    results = retrieve_top_k_similar([arrays[2], arrays[5]], store, k=3)
    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0][0].path == paths[2]
    assert results[1][0].path == paths[5]


def test_single_image_returns_single_row(
    gallery: tuple[list[UInt8NumpyArray], list[str], InMemoryImageEmbeddingStore],
) -> None:
    """A single image (not wrapped in a list) yields a one-row result."""
    arrays, paths, store = gallery
    results = retrieve_top_k_similar(arrays[1], store, k=2)
    assert len(results) == 1
    assert results[0][0].path == paths[1]


def test_candidates_carry_path_and_score(
    gallery: tuple[list[UInt8NumpyArray], list[str], InMemoryImageEmbeddingStore],
) -> None:
    """Every result entry is a :class:`Candidate` with a path and a score."""
    arrays, _, store = gallery
    results = retrieve_top_k_similar(arrays[0], store, k=3)
    for candidate in results[0]:
        assert isinstance(candidate, Candidate)
        assert isinstance(candidate.path, str)
        assert isinstance(candidate.score, float)


def test_k_limits_number_of_candidates(
    gallery: tuple[list[UInt8NumpyArray], list[str], InMemoryImageEmbeddingStore],
) -> None:
    """No query returns more than ``k`` candidates."""
    arrays, _, store = gallery
    results = retrieve_top_k_similar(arrays[:3], store, k=4)
    assert all(len(row) <= 4 for row in results)


def test_inner_product_scores_are_descending(
    gallery: tuple[list[UInt8NumpyArray], list[str], InMemoryImageEmbeddingStore],
) -> None:
    """Inner-product results are ordered most-similar first."""
    arrays, _, store = gallery
    results = retrieve_top_k_similar(arrays[4], store, k=5)
    scores = [candidate.score for candidate in results[0]]
    assert scores == sorted(scores, reverse=True)


def test_store_method_matches_functional(
    gallery: tuple[list[UInt8NumpyArray], list[str], InMemoryImageEmbeddingStore],
) -> None:
    """The store's own method delegates to the functional entry point."""
    arrays, _, store = gallery
    via_function = retrieve_top_k_similar(arrays[7], store, k=3)
    via_method = store.retrieve_top_k_similar(arrays[7], k=3)
    assert [c.path for c in via_function[0]] == [c.path for c in via_method[0]]
