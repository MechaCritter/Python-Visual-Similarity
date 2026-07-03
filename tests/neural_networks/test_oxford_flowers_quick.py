"""Quick integration test on a small Oxford Flowers subset.

Takes only 20 images from each flower class of the training split and runs
the real pipeline on it: pair mining with :class:`OxfordSiamesePairs`, a few
short training epochs of the actual :class:`SiameseNeuralNetwork` (pretrained
ResNet-18 backbone) via :func:`run_epoch`, and encoding/similarity scoring.

The whole module skips itself unless both the Oxford Flowers data and the
ResNet-18 weights are already cached locally, so the fast suite never
downloads anything (the full-download path belongs to the ``slow`` suite).
"""

from __future__ import annotations

import random
from collections import Counter
from unittest import mock

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset

import pyvisim.neural_networks.scripts.train_siamese_neural_network as ts
from pyvisim.datasets import OxfordFlowerDataset
from pyvisim.neural_networks import SiameseNeuralNetwork
from pyvisim.neural_networks.losses import ContrastiveLoss

from ._stubs import oxford_flowers_cached, resnet18_weights_cached

pytestmark = pytest.mark.skipif(
    not (oxford_flowers_cached() and resnet18_weights_cached()),
    reason="requires locally cached Oxford Flowers data and ResNet-18 weights",
)

#: Number of images taken from every flower class.
IMAGES_PER_CLASS = 20
#: Number of pairs used for the short training run.
TRAIN_PAIRS = 128
#: Embedding size kept small for speed.
EMBEDDING_DIM = 32


@pytest.fixture(scope="module")
def flower_subset() -> OxfordFlowerDataset:
    """The training split reduced to the first 20 images of every class.

    :return: An ``OxfordFlowerDataset`` whose paths/labels are the subset.
    """
    dataset = OxfordFlowerDataset(purpose="train")
    selected: dict[int, list[int]] = {}
    for index, label in enumerate(dataset.labels):
        indices = selected.setdefault(label, [])
        if len(indices) < IMAGES_PER_CLASS:
            indices.append(index)
    keep = sorted(index for indices in selected.values() for index in indices)
    dataset.image_paths = [dataset.image_paths[index] for index in keep]
    dataset.labels = [dataset.labels[index] for index in keep]
    return dataset


@pytest.fixture(scope="module")
def pair_dataset(flower_subset: OxfordFlowerDataset) -> ts.OxfordSiamesePairs:
    """Contrastive pairs mined from the 20-images-per-class subset.

    :param flower_subset: The reduced flower dataset.
    :return: An ``OxfordSiamesePairs`` over the subset.
    """
    with mock.patch.object(ts, "OxfordFlowerDataset", return_value=flower_subset):
        return ts.OxfordSiamesePairs("train", transform=ts.VAL_TF)


@pytest.fixture(scope="module")
def model() -> SiameseNeuralNetwork:
    """The real Siamese network with its pretrained ResNet-18 backbone.

    :return: Model on the script's device with a small embedding space.
    """
    torch.manual_seed(0)
    return SiameseNeuralNetwork(embedding_dim=EMBEDDING_DIM, device=ts.device)


def test_subset_has_20_images_per_class(flower_subset: OxfordFlowerDataset) -> None:
    """Every class contributes exactly 20 images to the subset."""
    counts = Counter(flower_subset.labels)
    assert len(counts) >= 2
    assert all(count == IMAGES_PER_CLASS for count in counts.values())
    assert len(flower_subset) == IMAGES_PER_CLASS * len(counts)


def test_subset_items_are_valid_images(flower_subset: OxfordFlowerDataset) -> None:
    """Subset items load as HWC uint8 images with their class label."""
    image, label, path = flower_subset[0]
    assert image.ndim == 3
    assert image.shape[2] == 3
    assert image.dtype == np.uint8
    assert label in set(flower_subset.labels)
    assert path == flower_subset.image_paths[0]


def test_pair_dataset_covers_subset(
    pair_dataset: ts.OxfordSiamesePairs, flower_subset: OxfordFlowerDataset
) -> None:
    """The pair dataset yields one well-formed pair per subset image."""
    assert len(pair_dataset) == len(flower_subset)
    img_a, img_b, label = pair_dataset[0]
    assert img_a.shape == (3, 224, 224)
    assert img_b.shape == (3, 224, 224)
    assert label.item() in (0.0, 1.0)


def test_short_training_on_subset_reduces_loss(
    pair_dataset: ts.OxfordSiamesePairs, model: SiameseNeuralNetwork
) -> None:
    """Three short epochs on repeated subset pairs drive the loss down."""
    loader = DataLoader(
        Subset(pair_dataset, range(TRAIN_PAIRS)), batch_size=16, shuffle=False
    )
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    losses = []
    for epoch in range(1, 4):
        # Re-seed so every epoch samples the identical pairs.
        random.seed(123)
        losses.append(
            ts.run_epoch(
                model, loader, criterion, optimizer, is_train=True, epoch=epoch
            )
        )

    assert all(np.isfinite(loss) for loss in losses)
    assert losses[-1] < losses[0]


def test_encode_flower_images(
    model: SiameseNeuralNetwork, flower_subset: OxfordFlowerDataset
) -> None:
    """Real flower images encode to unit-norm embedding rows."""
    indices = (0, 1, len(flower_subset) // 2, len(flower_subset) - 1)
    images = [flower_subset[index][0] for index in indices]
    embeddings = model.encode(images)
    assert embeddings.shape == (4, EMBEDDING_DIM)
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_identical_flower_similarity_is_one(
    model: SiameseNeuralNetwork, flower_subset: OxfordFlowerDataset
) -> None:
    """A flower image compared with itself scores cosine similarity 1."""
    image = flower_subset[0][0]
    score = model.similarity_score(image, image.copy())
    assert score.shape == (1, 1)
    assert score[0, 0] == pytest.approx(1.0, abs=1e-5)


def test_similarity_matrix_on_flowers(
    model: SiameseNeuralNetwork, flower_subset: OxfordFlowerDataset
) -> None:
    """Batches of flowers produce a bounded ``(N, M)`` similarity matrix."""
    batch_a = [flower_subset[index][0] for index in (0, 1)]
    batch_b = [flower_subset[index][0] for index in (2, 3, 4)]
    score = model.similarity_score(batch_a, batch_b)
    assert score.shape == (2, 3)
    assert np.all(score >= -1.0 - 1e-6)
    assert np.all(score <= 1.0 + 1e-6)
