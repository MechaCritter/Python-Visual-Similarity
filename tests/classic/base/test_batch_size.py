"""Tests for the batched feature extraction and encoding of VLAD and Fisher."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pyvisim.classic import FisherVectorEmbedder, VLADEmbedder

if TYPE_CHECKING:
    from tests.conftest import ImageObj

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


def test_default_batch_size_is_one_image() -> None:
    """Clustering-based embedders walk their input as a stream by default."""
    assert VLADEmbedder(n_clusters=8).batch_size == 1
    assert FisherVectorEmbedder(n_components=8).batch_size == 1


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_vlad_batch_size_does_not_change_embeddings(
    learned_vlad_embedder: VLADEmbedder,
    sample_images: list[np.ndarray],
    batch_size: int,
) -> None:
    """Batching only regroups the work: the VLAD vectors stay the same."""
    learned_vlad_embedder.set_batch_size(1)
    one_at_a_time = learned_vlad_embedder.embed(sample_images)
    learned_vlad_embedder.set_batch_size(batch_size)
    batched = learned_vlad_embedder.embed(sample_images)
    learned_vlad_embedder.set_batch_size(1)
    assert batched.shape == one_at_a_time.shape
    np.testing.assert_allclose(batched, one_at_a_time, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_fisher_batch_size_does_not_change_embeddings(
    learned_fisher_embedder: FisherVectorEmbedder,
    sample_images: list[np.ndarray],
    batch_size: int,
) -> None:
    """Batching only regroups the work: the Fisher vectors stay the same."""
    learned_fisher_embedder.set_batch_size(1)
    one_at_a_time = learned_fisher_embedder.embed(sample_images)
    learned_fisher_embedder.set_batch_size(batch_size)
    batched = learned_fisher_embedder.embed(sample_images)
    learned_fisher_embedder.set_batch_size(1)
    assert batched.shape == one_at_a_time.shape
    np.testing.assert_allclose(batched, one_at_a_time, rtol=1e-5, atol=1e-6)


def test_vlad_unflattened_batching_keeps_the_layout(
    learned_vlad_embedder: VLADEmbedder, sample_images: list[np.ndarray]
) -> None:
    """With ``flatten`` off, batches stack per-cluster rows just as singles do."""
    learned_vlad_embedder.flatten = False
    try:
        learned_vlad_embedder.set_batch_size(1)
        one_at_a_time = learned_vlad_embedder.embed(sample_images)
        learned_vlad_embedder.set_batch_size(3)
        batched = learned_vlad_embedder.embed(sample_images)
    finally:
        learned_vlad_embedder.flatten = True
        learned_vlad_embedder.set_batch_size(1)
    n_clusters = learned_vlad_embedder.clustering_model.n_clusters
    assert one_at_a_time.shape[0] == len(sample_images) * n_clusters
    np.testing.assert_allclose(batched, one_at_a_time, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_learn_does_not_depend_on_the_batch_size(
    sample_images: list[np.ndarray], batch_size: int
) -> None:
    """The vocabulary is fitted on the same features however they are batched."""
    reference = VLADEmbedder(n_clusters=4, kmeans_params={"rng": 0})
    reference.learn(sample_images)
    embedder = VLADEmbedder(
        n_clusters=4, kmeans_params={"rng": 0}, batch_size=batch_size
    )
    embedder.learn(sample_images)
    np.testing.assert_allclose(
        embedder.clustering_model.cluster_centers,
        reference.clustering_model.cluster_centers,
        rtol=1e-5,
        atol=1e-6,
    )


def test_batch_size_applies_to_a_generator_of_images(
    learned_vlad_embedder: VLADEmbedder, sample_images: list[np.ndarray]
) -> None:
    """A generator is consumed batch by batch and embeds like a list."""
    learned_vlad_embedder.set_batch_size(2)
    try:
        from_generator = learned_vlad_embedder.embed(image for image in sample_images)
        from_list = learned_vlad_embedder.embed(sample_images)
    finally:
        learned_vlad_embedder.set_batch_size(1)
    np.testing.assert_array_equal(from_generator, from_list)


def test_vlad_reports_a_featureless_image_inside_a_batch(
    learned_vlad_embedder: VLADEmbedder,
    sample_images: list[np.ndarray],
    solid_image: ImageObj,
) -> None:
    """An image yielding no descriptor is still caught when it shares a batch."""
    learned_vlad_embedder.set_batch_size(-1)
    try:
        with pytest.raises(ValueError, match="No descriptors found in the image"):
            learned_vlad_embedder.embed([*sample_images, solid_image.array])
    finally:
        learned_vlad_embedder.set_batch_size(1)


def test_batch_size_survives_an_embedder_file_round_trip(
    learned_vlad_embedder: VLADEmbedder, tmp_path
) -> None:
    """A saved VLAD embedder comes back with the batch size it was saved with."""
    learned_vlad_embedder.set_batch_size(8)
    try:
        path = learned_vlad_embedder.save_to_disk(tmp_path / "vlad")
    finally:
        learned_vlad_embedder.set_batch_size(1)
    assert VLADEmbedder.load_from_disk(path).batch_size == 8


def test_invalid_batch_size_is_rejected_by_the_constructor() -> None:
    """The embedders validate the batch size like every other metric."""
    with pytest.raises(ValueError, match="batch_size"):
        VLADEmbedder(n_clusters=8, batch_size=0)
    with pytest.raises(ValueError, match="batch_size"):
        FisherVectorEmbedder(n_components=8, batch_size=-2)
