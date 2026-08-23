"""Slow serialization tests of the Siamese networks on a full ResNet-18.

The fast suite in ``tests/neural_networks/test_serialization.py`` keeps
itself offline: it builds the networks with an untrained backbone (or a stub
one) and scores synthetic images. These tests take the opposite route and
serialise the model a user actually builds: the ImageNet-pretrained
ResNet-18 backbone, run on real flower images, so the round trip is checked
over real weights instead of random ones.

The module is marked ``slow`` because constructing the networks downloads the
pretrained weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pyvisim.datasets import OxfordFlowerDataset
from pyvisim.neural_networks import BCESiameseNetwork, ContrastiveSiameseNetwork
from pyvisim.typing import UInt8NumpyArray

pytestmark = pytest.mark.slow

#: Indices of the flower images every test is run on.
_IMAGE_INDICES = (7, 10, 15, 1000)

#: Embedding size of the networks, the constructor default.
_EMBEDDING_DIM = 128


@pytest.fixture(scope="module")
def flower_images(flower_subset: OxfordFlowerDataset) -> list[UInt8NumpyArray]:
    """Real RGB flower images of the dataset, at their native resolutions.

    :param flower_subset: The reduced Oxford Flowers training split.
    :return: One ``(H, W, 3)`` ``uint8`` array per index in ``_IMAGE_INDICES``.
    """
    return [flower_subset[index][0] for index in _IMAGE_INDICES]


@pytest.fixture(scope="module")
def contrastive_network() -> ContrastiveSiameseNetwork:
    """A contrastive network on the ImageNet-pretrained ResNet-18 backbone.

    :return: An untrained but fully pretrained-backbone network in eval mode.
    """
    torch.manual_seed(0)
    return ContrastiveSiameseNetwork(backbone="resnet18", pretrained_backbone=True)


@pytest.fixture(scope="module")
def bce_network() -> BCESiameseNetwork:
    """A BCE network on the ImageNet-pretrained ResNet-18 backbone.

    :return: An untrained but fully pretrained-backbone network in eval mode.
    """
    torch.manual_seed(0)
    return BCESiameseNetwork(backbone="resnet18", pretrained_backbone=True)


# §1 embedding the same image twice


def test_contrastive_embedding_is_deterministic(
    contrastive_network: ContrastiveSiameseNetwork,
    flower_images: list[UInt8NumpyArray],
) -> None:
    """The full network embeds one image bit-identically on every call."""
    embeddings = contrastive_network.embed(flower_images)
    assert embeddings.shape == (len(flower_images), _EMBEDDING_DIM)
    assert np.array_equal(contrastive_network.embed(flower_images), embeddings)


def test_bce_score_is_deterministic(
    bce_network: BCESiameseNetwork, flower_images: list[UInt8NumpyArray]
) -> None:
    """The full BCE network scores one pair bit-identically on every call.

    It has no embedding space, so its pair probability is what has to be
    reproducible.
    """
    image1, image2 = flower_images[0], flower_images[1]
    scores = bce_network.similarity_score(image1, image2)
    assert scores.shape == (1, 1)
    assert np.array_equal(bce_network.similarity_score(image1, image2), scores)


# §2 the round trip through a .embedder file


def test_contrastive_embeddings_are_identical_after_serialization(
    contrastive_network: ContrastiveSiameseNetwork,
    flower_images: list[UInt8NumpyArray],
    tmp_path: Path,
) -> None:
    """A reloaded full network embeds the images bit-identically."""
    expected = contrastive_network.embed(flower_images)

    path = contrastive_network.save_to_disk(tmp_path / "contrastive_resnet18")
    reloaded = ContrastiveSiameseNetwork.load_from_disk(path)

    assert np.array_equal(reloaded.embed(flower_images), expected)


def test_bce_scores_are_identical_after_serialization(
    bce_network: BCESiameseNetwork,
    flower_images: list[UInt8NumpyArray],
    tmp_path: Path,
) -> None:
    """A reloaded full BCE network scores the pairs bit-identically."""
    image1, image2 = flower_images[0], flower_images[1]
    expected = bce_network.similarity_score(image1, image2)

    path = bce_network.save_to_disk(tmp_path / "bce_resnet18")
    reloaded = BCESiameseNetwork.load_from_disk(path)

    assert np.array_equal(reloaded.similarity_score(image1, image2), expected)


# §3 the reloaded network is deterministic in turn


def test_reloaded_contrastive_embedding_is_deterministic(
    contrastive_network: ContrastiveSiameseNetwork,
    flower_images: list[UInt8NumpyArray],
    tmp_path: Path,
) -> None:
    """Embedding the same image twice stays exact after a round trip."""
    path = contrastive_network.save_to_disk(tmp_path / "contrastive_resnet18")
    reloaded = ContrastiveSiameseNetwork.load_from_disk(path)

    embeddings = reloaded.embed(flower_images)
    assert np.array_equal(reloaded.embed(flower_images), embeddings)
