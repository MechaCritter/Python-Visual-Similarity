"""Serialization tests for the neural image embedders.

Every neural embedder is a
:class:`~pyvisim._base_classes.SerializableImageEmbedder`, so it round-trips
through a ``.embedder`` safetensors file. The central promise checked here is
exactness: a reloaded embedder must produce bit-identical embeddings (and
scores) without downloading any pretrained weights.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from torchvision import transforms

from pyvisim._base_classes import SerializableImageEmbedder
from pyvisim.neural_networks import BCESiameseNetwork, ContrastiveSiameseNetwork
from pyvisim.neural_networks.siamese import backbones as backbones_module
from pyvisim.neural_networks.siamese._base_siamese import SiameseNetworkBase
from pyvisim.serialization import embedder_from_dict, embedder_to_dict

from ._stubs import make_random_rgb_image

#: Embedding size used by every network built here; small but not degenerate.
EMBEDDING_DIM = 8


def _contrastive_network() -> ContrastiveSiameseNetwork:
    """Build an untrained contrastive network with reproducible weights."""
    torch.manual_seed(0)
    return ContrastiveSiameseNetwork(
        embedding_dim=EMBEDDING_DIM, pretrained_backbone=False
    )


def _bce_network() -> BCESiameseNetwork:
    """Build an untrained BCE network with reproducible weights."""
    torch.manual_seed(0)
    return BCESiameseNetwork(embedding_dim=EMBEDDING_DIM, pretrained_backbone=False)


# §1 the serialization contract


@pytest.mark.parametrize(
    "network", [_contrastive_network, _bce_network], ids=["contrastive", "bce"]
)
def test_neural_embedders_are_serializable_embedders(
    network: Callable[[], SiameseNetworkBase],
) -> None:
    """Every neural embedder inherits the ``.embedder`` file contract."""
    assert isinstance(network(), SerializableImageEmbedder)


def test_save_to_disk_appends_the_embedder_suffix(tmp_path: Path) -> None:
    """A path without a suffix gets the ``.embedder`` one appended."""
    path = _contrastive_network().save_to_disk(tmp_path / "siamese")
    assert path == tmp_path / "siamese.embedder"
    assert path.is_file()


def test_load_from_disk_rejects_a_file_of_another_embedder(tmp_path: Path) -> None:
    """Loading a file with the wrong embedder class raises ``ValueError``."""
    path = _contrastive_network().save_to_disk(tmp_path / "siamese")
    with pytest.raises(ValueError, match="ContrastiveSiameseNetwork"):
        BCESiameseNetwork.load_from_disk(path)


# §2 exactness of the round trip


def test_contrastive_embeddings_are_identical_after_serialization(
    tmp_path: Path,
) -> None:
    """A reloaded contrastive network embeds images bit-identically."""
    network = _contrastive_network()
    images = [make_random_rgb_image(seed=1), make_random_rgb_image(seed=2)]
    expected = network.embed(images)

    path = network.save_to_disk(tmp_path / "siamese")
    reloaded = ContrastiveSiameseNetwork.load_from_disk(path)

    assert np.array_equal(reloaded.embed(images), expected)


def test_contrastive_scores_are_identical_after_serialization(
    tmp_path: Path,
) -> None:
    """A reloaded contrastive network scores image pairs bit-identically."""
    network = _contrastive_network()
    image1, image2 = make_random_rgb_image(seed=1), make_random_rgb_image(seed=2)
    expected = network.similarity_score(image1, image2)

    path = network.save_to_disk(tmp_path / "siamese")
    reloaded = ContrastiveSiameseNetwork.load_from_disk(path)

    assert np.array_equal(reloaded.similarity_score(image1, image2), expected)


def test_bce_scores_are_identical_after_serialization(tmp_path: Path) -> None:
    """A reloaded BCE network scores image pairs bit-identically.

    The BCE network has no embedding space of its own, so its learned
    scoring layer is what has to survive the round trip.
    """
    network = _bce_network()
    image1, image2 = make_random_rgb_image(seed=1), make_random_rgb_image(seed=2)
    expected = network.similarity_score(image1, image2)

    path = network.save_to_disk(tmp_path / "siamese")
    reloaded = BCESiameseNetwork.load_from_disk(path)

    assert np.array_equal(reloaded.similarity_score(image1, image2), expected)


def test_trained_weights_survive_the_round_trip(tmp_path: Path) -> None:
    """Weights changed after construction are the ones that are restored."""
    network = _contrastive_network()
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.add_(0.1)
    images = [make_random_rgb_image(seed=3)]
    expected = network.embed(images)

    path = network.save_to_disk(tmp_path / "siamese")
    reloaded = ContrastiveSiameseNetwork.load_from_disk(path)

    assert np.array_equal(reloaded.embed(images), expected)
    for restored, original in zip(
        reloaded.parameters(), network.parameters(), strict=True
    ):
        assert torch.equal(restored, original)


# §3 what the reloaded embedder looks like


def test_configuration_survives_the_round_trip(tmp_path: Path) -> None:
    """The architecture and the similarity metric are restored."""
    torch.manual_seed(0)
    network = ContrastiveSiameseNetwork(
        embedding_dim=EMBEDDING_DIM,
        similarity_func="euclidean",
        pretrained_backbone=False,
    )
    path = network.save_to_disk(tmp_path / "siamese")
    reloaded = ContrastiveSiameseNetwork.load_from_disk(path)

    assert reloaded.similarity_func_name == "euclidean"
    assert isinstance(reloaded.head, torch.nn.Linear)
    assert reloaded.head.out_features == EMBEDDING_DIM
    assert str(reloaded.device) == "cpu"


def test_reloaded_network_is_in_evaluation_mode(tmp_path: Path) -> None:
    """A reloaded network is ready for inference, not for training."""
    network = _contrastive_network()
    network.train()
    path = network.save_to_disk(tmp_path / "siamese")
    assert ContrastiveSiameseNetwork.load_from_disk(path).training is False


def test_loading_does_not_download_pretrained_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rebuilding the backbone skips the ImageNet weights it will overwrite."""
    path = _contrastive_network().save_to_disk(tmp_path / "siamese")
    requested: list[object] = []
    build_resnet18 = backbones_module.models.resnet18

    def spy(weights: object = None) -> torch.nn.Module:
        requested.append(weights)
        return build_resnet18(weights=None)

    monkeypatch.setattr(backbones_module.models, "resnet18", spy)
    ContrastiveSiameseNetwork.load_from_disk(path)

    assert requested == [None]


# §4 a custom transform cannot be serialised


def test_custom_transform_warns_when_serialised(tmp_path: Path) -> None:
    """Serialising a network built with a custom transform warns loudly."""
    torch.manual_seed(0)
    network = ContrastiveSiameseNetwork(
        embedding_dim=EMBEDDING_DIM,
        transform=transforms.Compose([transforms.Resize(64), transforms.ToTensor()]),
        pretrained_backbone=False,
    )
    with pytest.warns(FutureWarning, match="custom transform"):
        network.save_to_disk(tmp_path / "siamese")


def test_default_transform_does_not_warn(tmp_path: Path) -> None:
    """The default preprocessing is fully described by the backbone name."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _contrastive_network().save_to_disk(tmp_path / "siamese")


# §5 reconstruction through the class-name registry


def test_registry_round_trip_restores_the_embeddings() -> None:
    """The registry used by the image store rebuilds neural embedders too."""
    network = _contrastive_network()
    images = [make_random_rgb_image(seed=4)]
    expected = network.embed(images)

    reloaded = embedder_from_dict(embedder_to_dict(network))

    assert isinstance(reloaded, ContrastiveSiameseNetwork)
    assert np.array_equal(reloaded.embed(images), expected)
