"""Tests for the batched embedding of :class:`pyvisim.classic.Pipeline`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyvisim.classic import FisherVectorEmbedder, Pipeline, VLADEmbedder

#: Batch sizes that split the sample images differently: one image at a time, an
#: exact divisor, a size leaving a shorter last batch, and the whole input.
BATCH_SIZES = [1, 2, 3, -1]


@pytest.fixture(scope="module")
def sample_images(category_train_images_flat: list[np.ndarray]) -> list[np.ndarray]:
    """Four images to embed, enough for every batch size to split them.

    :param category_train_images_flat: flattened training images.
    :returns: a short list of images.
    """
    return category_train_images_flat[:4]


@pytest.fixture(scope="module")
def pipeline(category_train_images_flat: list[np.ndarray]) -> Pipeline:
    """A pipeline over a learned VLAD and Fisher embedder.

    :param category_train_images_flat: flattened training images to learn from.
    :returns: a ``Pipeline`` over both fitted embedders.
    """
    vlad = VLADEmbedder(n_clusters=8, kmeans_params={"rng": 0})
    fisher = FisherVectorEmbedder(n_components=8, gmm_params={"rng": 0})
    vlad.learn(category_train_images_flat)
    fisher.learn(category_train_images_flat)
    return Pipeline([vlad, fisher])


def test_default_batch_size_hands_over_the_whole_input() -> None:
    """A pipeline keeps handing its embedders everything it is given."""
    assert Pipeline([]).batch_size == -1


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_batch_size_does_not_change_embeddings(
    pipeline: Pipeline, sample_images: list[np.ndarray], batch_size: int
) -> None:
    """Batching only regroups the work: the concatenated vectors stay the same."""
    pipeline.set_batch_size(-1)
    whole = pipeline.embed(sample_images)
    pipeline.set_batch_size(batch_size)
    batched = pipeline.embed(sample_images)
    pipeline.set_batch_size(-1)
    assert batched.shape == whole.shape == (len(sample_images), whole.shape[1])
    np.testing.assert_allclose(batched, whole, rtol=1e-5, atol=1e-6)


def test_the_embedders_keep_their_own_batch_size(
    pipeline: Pipeline, sample_images: list[np.ndarray]
) -> None:
    """A pipeline batch is split further by whatever its embedders are set to."""
    for embedder in pipeline.embedders:
        embedder.set_batch_size(2)
    pipeline.set_batch_size(3)
    try:
        batched = pipeline.embed(sample_images)
    finally:
        pipeline.set_batch_size(-1)
        for embedder in pipeline.embedders:
            embedder.set_batch_size(1)
    np.testing.assert_allclose(
        batched, pipeline.embed(sample_images), rtol=1e-5, atol=1e-6
    )


def test_the_flatten_flag_is_restored_after_each_batch(
    pipeline: Pipeline, sample_images: list[np.ndarray]
) -> None:
    """Forcing ``flatten`` on for the run must not leave the embedders changed."""
    vlad = pipeline.embedders[0]
    vlad.flatten = False  # type: ignore[attr-defined]
    pipeline.set_batch_size(2)
    try:
        pipeline.embed(sample_images)
        assert vlad.flatten is False  # type: ignore[attr-defined]
    finally:
        vlad.flatten = True  # type: ignore[attr-defined]
        pipeline.set_batch_size(-1)


def test_batch_size_survives_an_embedder_file_round_trip(
    pipeline: Pipeline, tmp_path: Path
) -> None:
    """A saved pipeline comes back with the batch size it was saved with."""
    pipeline.set_batch_size(4)
    try:
        path = pipeline.save_to_disk(tmp_path / "pipeline")
    finally:
        pipeline.set_batch_size(-1)
    assert Pipeline.load_from_disk(path).batch_size == 4


def test_invalid_batch_size_is_rejected_by_the_constructor() -> None:
    """The pipeline validates the batch size like every other metric."""
    with pytest.raises(ValueError, match="batch_size"):
        Pipeline([], batch_size=0)
