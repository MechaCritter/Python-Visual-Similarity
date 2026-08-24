"""Slow serialization tests of the triplet network on a full ResNet-18.

The fast suite in ``tests/neural_networks/test_serialization.py`` keeps
itself offline: it builds the network with an untrained backbone (or a stub
one) and embeds synthetic images. These tests take the opposite route and
serialise the model a user actually builds: the ImageNet-pretrained
ResNet-18 backbone, run on real flower images, so the round trip is checked
over real weights instead of random ones.

The module is marked ``slow`` because constructing the network downloads the
pretrained weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pyvisim.datasets import OxfordFlowerDataset
from pyvisim.neural_networks import TripletNeuralNetwork
from pyvisim.typing import UInt8NumpyArray

pytestmark = pytest.mark.slow

#: Indices of the flower images every test is run on.
_IMAGE_INDICES = (7, 10, 15, 1000)

#: Embedding size of the network, the constructor default.
_EMBEDDING_DIM = 128


@pytest.fixture(scope="module")
def flower_images(flower_subset: OxfordFlowerDataset) -> list[UInt8NumpyArray]:
    """Real RGB flower images of the dataset, at their native resolutions.

    :param flower_subset: The reduced Oxford Flowers training split.
    :return: One ``(H, W, 3)`` ``uint8`` array per index in ``_IMAGE_INDICES``.
    """
    return [flower_subset[index][0] for index in _IMAGE_INDICES]


@pytest.fixture(scope="module")
def triplet_network() -> TripletNeuralNetwork:
    """A triplet network on the ImageNet-pretrained ResNet-18 backbone.

    :return: An untrained but fully pretrained-backbone network in eval mode.
    """
    torch.manual_seed(0)
    return TripletNeuralNetwork(backbone="resnet18", pretrained_backbone=True)


# §1 embedding the same image twice


def test_embedding_is_deterministic(
    triplet_network: TripletNeuralNetwork, flower_images: list[UInt8NumpyArray]
) -> None:
    """The full network embeds one image bit-identically on every call."""
    embeddings = triplet_network.embed(flower_images)
    assert embeddings.shape == (len(flower_images), _EMBEDDING_DIM)
    assert np.array_equal(triplet_network.embed(flower_images), embeddings)


# §2 the round trip through a .embedder file


def test_embeddings_are_identical_after_serialization(
    triplet_network: TripletNeuralNetwork,
    flower_images: list[UInt8NumpyArray],
    tmp_path: Path,
) -> None:
    """A reloaded full network embeds the images bit-identically."""
    expected = triplet_network.embed(flower_images)

    path = triplet_network.save_to_disk(tmp_path / "triplet_resnet18")
    reloaded = TripletNeuralNetwork.load_from_disk(path)

    assert np.array_equal(reloaded.embed(flower_images), expected)


# §3 the reloaded network is deterministic in turn


def test_reloaded_embedding_is_deterministic(
    triplet_network: TripletNeuralNetwork,
    flower_images: list[UInt8NumpyArray],
    tmp_path: Path,
) -> None:
    """Embedding the same image twice stays exact after a round trip."""
    path = triplet_network.save_to_disk(tmp_path / "triplet_resnet18")
    reloaded = TripletNeuralNetwork.load_from_disk(path)

    embeddings = reloaded.embed(flower_images)
    assert np.array_equal(reloaded.embed(flower_images), embeddings)
