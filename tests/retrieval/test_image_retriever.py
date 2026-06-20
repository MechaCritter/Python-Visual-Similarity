"""Tests for :class:`pyvisim.retrieval.ImageRetriever`."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pyvisim.functional import Candidate
from pyvisim.image_store import InMemoryImageEmbeddingStore
from pyvisim.retrieval import ImageRetriever
from pyvisim.typing import FloatNumpyArray, ImageInput, UInt8NumpyArray

#: Number of gallery images materialized for the tests.
NUM_GALLERY = 12


class FlattenEncoder:
    """Deterministic encoder flattening each RGB image into its pixel vector."""

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
def retriever(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[list[UInt8NumpyArray], list[str], ImageRetriever]:
    """Build an :class:`ImageRetriever` over a random on-disk gallery.

    :param tmp_path_factory: pytest's temp-directory factory.
    :returns: a tuple ``(arrays, paths, retriever)``.
    """
    directory = tmp_path_factory.mktemp("retriever_gallery")
    rng = np.random.default_rng(1)
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
    return arrays, paths, ImageRetriever(store)


def test_store_property_returns_the_store(
    retriever: tuple[list[UInt8NumpyArray], list[str], ImageRetriever],
) -> None:
    """The retriever exposes the store it was built with."""
    _, _, machine = retriever
    assert isinstance(machine.store, InMemoryImageEmbeddingStore)


def test_retrieve_returns_ordered_lists_of_candidates(
    retriever: tuple[list[UInt8NumpyArray], list[str], ImageRetriever],
) -> None:
    """Retrieval returns one ranked candidate list per query, in input order."""
    arrays, paths, machine = retriever
    results = machine.retrieve_top_k_similar([arrays[3], arrays[9]], k=4)
    assert len(results) == 2
    assert all(isinstance(candidate, Candidate) for row in results for candidate in row)
    assert results[0][0].path == paths[3]
    assert results[1][0].path == paths[9]


def test_retrieve_single_image_returns_single_row(
    retriever: tuple[list[UInt8NumpyArray], list[str], ImageRetriever],
) -> None:
    """A single query image yields a single ranked list."""
    arrays, paths, machine = retriever
    results = machine.retrieve_top_k_similar(arrays[6], k=2)
    assert len(results) == 1
    assert results[0][0].path == paths[6]
