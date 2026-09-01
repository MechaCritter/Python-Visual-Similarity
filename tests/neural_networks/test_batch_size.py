"""Tests for the batched forward passes of the neural embedders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyvisim.neural_networks import (
    BCESiameseNetwork,
    ContrastiveSiameseNetwork,
    TripletNeuralNetwork,
)

from ._stubs import (
    MeanBackbone,
    build_stub_bce_model,
    build_stub_model,
    build_stub_triplet_model,
    make_random_rgb_image,
)

#: Batch sizes that split the sample images differently: one image per pass, an
#: exact divisor, a size leaving a shorter last pass, and the whole input.
BATCH_SIZES = [1, 2, 3, -1]

#: Builders of the three networks that embed images through a backbone.
NETWORK_BUILDERS = [build_stub_model, build_stub_triplet_model]


@pytest.fixture()
def images() -> list[np.ndarray]:
    """Four RGB images, enough for every batch size to split them."""
    return [make_random_rgb_image(seed) for seed in range(4)]


def test_networks_embed_the_whole_input_in_one_pass_by_default() -> None:
    """The neural embedders keep their single-forward-pass default."""
    assert ContrastiveSiameseNetwork(pretrained_backbone=False).batch_size == -1
    assert TripletNeuralNetwork(pretrained_backbone=False).batch_size == -1
    assert BCESiameseNetwork(pretrained_backbone=False).batch_size == -1


@pytest.mark.parametrize("build", NETWORK_BUILDERS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_batch_size_does_not_change_embeddings(
    build: object, batch_size: int, images: list[np.ndarray]
) -> None:
    """Splitting the forward passes returns the same embeddings."""
    model = build(MeanBackbone(), embedding_dim=4, pretrained_backbone=False)  # type: ignore[operator]
    model.set_batch_size(-1)
    one_pass = model.embed(images)
    model.set_batch_size(batch_size)
    batched = model.embed(images)
    assert batched.shape == one_pass.shape == (len(images), 4)
    np.testing.assert_allclose(batched, one_pass, rtol=1e-6, atol=1e-6)


def test_batch_size_reaches_the_network_from_the_constructor() -> None:
    """Every network forwards ``batch_size`` to the shared metric base."""
    model = build_stub_model(MeanBackbone(), embedding_dim=4, batch_size=2)
    assert model.batch_size == 2
    scorer = build_stub_bce_model(MeanBackbone(), embedding_dim=4, batch_size=3)
    assert scorer.batch_size == 3


def test_embedding_nothing_still_raises(images: list[np.ndarray]) -> None:
    """An input holding no image is an error, not an empty embedding matrix."""
    model = build_stub_model(MeanBackbone(), embedding_dim=4, batch_size=2)
    with pytest.raises(ValueError, match="at least one image"):
        model.embed([])


def test_similarity_score_is_batched_too(images: list[np.ndarray]) -> None:
    """``similarity_score`` runs on top of ``embed``, so it batches as well."""
    model = build_stub_model(MeanBackbone(), embedding_dim=4)
    model.set_batch_size(-1)
    one_pass = model.similarity_score(images, images)
    model.set_batch_size(2)
    batched = model.similarity_score(images, images)
    np.testing.assert_allclose(batched, one_pass, rtol=1e-6, atol=1e-6)


def test_batch_size_survives_a_checkpoint_round_trip(tmp_path: Path) -> None:
    """A saved network comes back with the batch size it was saved with."""
    network = ContrastiveSiameseNetwork(
        embedding_dim=4, pretrained_backbone=False, batch_size=3
    )
    path = network.save_to_disk(tmp_path / "network")
    assert ContrastiveSiameseNetwork.load_from_disk(path).batch_size == 3


def test_invalid_batch_size_is_rejected_by_the_constructor() -> None:
    """The networks validate the batch size like every other metric."""
    with pytest.raises(ValueError, match="batch_size"):
        build_stub_model(MeanBackbone(), embedding_dim=4, batch_size=0)
