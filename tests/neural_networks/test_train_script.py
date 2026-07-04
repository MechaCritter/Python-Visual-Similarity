"""Unit tests for the Siamese training script.

Covers :class:`OxfordSiamesePairs`, :func:`run_epoch`, :func:`train` and
:func:`demo` from ``pyvisim.neural_networks.scripts.train_siamese_neural_network``.
The Oxford Flowers dataset is replaced by an in-memory fake and the backbone
by deterministic stubs, so everything here runs without network access; the
real-data integration lives in the Oxford Flowers test modules.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

import pyvisim.neural_networks.scripts.train_siamese_neural_network as ts
from pyvisim.neural_networks import SiameseNeuralNetwork
from pyvisim.neural_networks.losses import ContrastiveLoss

from ._stubs import (
    FakeFlowerDataset,
    FlattenBackbone,
    MeanBackbone,
    build_stub_model,
    install_head,
    make_identity_head,
    make_solid_rgb_image,
)

#: Four distinguishable images: two of class 1, two of class 2.
_IMAGES = [make_solid_rgb_image(value, size=48) for value in (10, 60, 110, 160)]
_LABELS = [1, 1, 2, 2]


@pytest.fixture
def fake_flowers(monkeypatch: pytest.MonkeyPatch) -> FakeFlowerDataset:
    """Patch the script's ``OxfordFlowerDataset`` with a 4-image fake.

    :param monkeypatch: pytest's monkeypatch fixture.
    :return: The fake dataset instance the script will receive.
    """
    fake = FakeFlowerDataset(_IMAGES, _LABELS)
    monkeypatch.setattr(
        ts, "OxfordFlowerDataset", lambda transform=None, purpose="train": fake
    )
    return fake


# §1 OxfordSiamesePairs


def test_pairs_length_matches_base_dataset(fake_flowers: FakeFlowerDataset) -> None:
    """The pair dataset has one entry per base image."""
    pairs = ts.OxfordSiamesePairs("train", transform=ts.VAL_TF)
    assert len(pairs) == len(fake_flowers)


def test_pairs_label_index_map(fake_flowers: FakeFlowerDataset) -> None:
    """The label-to-indices map groups base indices by class."""
    pairs = ts.OxfordSiamesePairs("train", transform=ts.VAL_TF)
    assert pairs._label_to_indices == {1: [0, 1], 2: [2, 3]}


def test_pairs_item_structure(fake_flowers: FakeFlowerDataset) -> None:
    """Each item is two transformed image tensors plus a float binary label."""
    pairs = ts.OxfordSiamesePairs("train", transform=ts.VAL_TF)
    img_a, img_b, label = pairs[0]
    assert img_a.shape == (3, 224, 224)
    assert img_b.shape == (3, 224, 224)
    assert label.dtype == torch.float32
    assert label.item() in (0.0, 1.0)


def test_positive_pairs_come_from_same_class(
    fake_flowers: FakeFlowerDataset,
) -> None:
    """With ``pos_fraction=1`` every pair is the same-class partner, label 1.

    Each class has exactly two members, so the partner of every index is
    uniquely determined.
    """
    pairs = ts.OxfordSiamesePairs("train", transform=ts.VAL_TF, pos_fraction=1.0)
    partner = {0: 1, 1: 0, 2: 3, 3: 2}
    for idx in range(4):
        _, img_b, label = pairs[idx]
        assert label.item() == 1.0
        assert torch.equal(img_b, ts.VAL_TF(_IMAGES[partner[idx]]))


def test_negative_pairs_come_from_other_class(
    fake_flowers: FakeFlowerDataset,
) -> None:
    """With ``pos_fraction=0`` every pair crosses classes, label 0."""
    random.seed(0)
    pairs = ts.OxfordSiamesePairs("train", transform=ts.VAL_TF, pos_fraction=0.0)
    other_class = {0: (2, 3), 1: (2, 3), 2: (0, 1), 3: (0, 1)}
    for idx in range(4):
        _, img_b, label = pairs[idx]
        assert label.item() == 0.0
        candidates = [ts.VAL_TF(_IMAGES[j]) for j in other_class[idx]]
        assert any(torch.equal(img_b, candidate) for candidate in candidates)


def test_single_member_class_pairs_with_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A positive pair for a singleton class falls back to the image itself."""
    fake = FakeFlowerDataset(_IMAGES[:3], [1, 2, 2])
    monkeypatch.setattr(
        ts, "OxfordFlowerDataset", lambda transform=None, purpose="train": fake
    )
    pairs = ts.OxfordSiamesePairs("train", transform=ts.VAL_TF, pos_fraction=1.0)
    _, img_b, label = pairs[0]
    assert label.item() == 1.0
    assert torch.equal(img_b, ts.VAL_TF(_IMAGES[0]))


def test_pairs_require_at_least_two_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A base dataset with a single class is rejected."""
    fake = FakeFlowerDataset(_IMAGES[:2], [1, 1])
    monkeypatch.setattr(
        ts, "OxfordFlowerDataset", lambda transform=None, purpose="train": fake
    )
    with pytest.raises(ValueError, match=">= 2 classes"):
        ts.OxfordSiamesePairs("train", transform=ts.VAL_TF)


# §2 run_epoch


def _unit_vector_loader() -> DataLoader[tuple[torch.Tensor, ...]]:
    """Loader with four hand-crafted unit-vector pairs in two batches.

    Batch 1 pairs (margin 1): orthogonal/similar -> loss 1.0 and
    orthogonal/dissimilar -> loss 0.0, mean 0.5. Batch 2: identical/similar
    -> 0.0 and identical/dissimilar -> 0.5, mean 0.25. Epoch average: 0.375.

    :return: A deterministic, unshuffled two-batch loader.
    """
    emb_a = torch.tensor([[1.0, 0.0]] * 4)
    emb_b = torch.tensor([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    dataset = TensorDataset(emb_a, emb_b, labels)
    return DataLoader(dataset, batch_size=2, shuffle=False)


def _normalizing_model() -> SiameseNeuralNetwork:
    """Model whose forward pass is exactly L2 normalization of 2-vectors.

    :return: Stub model on the script's device with an identity head.
    """
    model = build_stub_model(FlattenBackbone(2), embedding_dim=2, device=ts.device)
    install_head(model, make_identity_head(2))
    return model


def test_run_epoch_returns_hand_computed_average_loss() -> None:
    """An eval epoch over known pairs returns the exact average batch loss."""
    model = _normalizing_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    loss = ts.run_epoch(
        model,
        _unit_vector_loader(),
        ContrastiveLoss(margin=1.0),
        optimizer,
        is_train=False,
        epoch=1,
    )
    assert isinstance(loss, float)
    assert loss == pytest.approx(0.375, abs=1e-3)


def test_eval_epoch_does_not_update_parameters() -> None:
    """Without training, model parameters stay bitwise identical."""
    model = _normalizing_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    ts.run_epoch(
        model,
        _unit_vector_loader(),
        ContrastiveLoss(margin=1.0),
        optimizer,
        is_train=False,
        epoch=1,
    )
    for parameter, snapshot in zip(model.parameters(), before, strict=True):
        assert torch.equal(parameter.detach(), snapshot)


def test_run_epoch_sets_model_mode() -> None:
    """The epoch runner switches the model to train or eval mode."""
    model = _normalizing_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    criterion = ContrastiveLoss(margin=1.0)
    ts.run_epoch(
        model, _unit_vector_loader(), criterion, optimizer, is_train=True, epoch=1
    )
    assert model.training is True
    ts.run_epoch(
        model, _unit_vector_loader(), criterion, optimizer, is_train=False, epoch=1
    )
    assert model.training is False


def _cluster_pair_loader(seed: int = 0) -> DataLoader[tuple[torch.Tensor, ...]]:
    """Loader of trivially separable contrastive pairs from two 2-D clusters.

    Positive pairs come from within one cluster, negative pairs across the
    two clusters, so a linear head can drive the loss towards zero.

    :param seed: Seed for the cluster noise.
    :return: A 32-pair loader with batch size 8.
    """
    generator = torch.Generator().manual_seed(seed)

    def sample(center: torch.Tensor, count: int) -> torch.Tensor:
        return center + 0.1 * torch.randn(count, 2, generator=generator)

    center_a = torch.tensor([2.0, 0.0])
    center_b = torch.tensor([0.0, 2.0])
    emb_a = torch.cat([sample(center_a, 8), sample(center_b, 8), sample(center_a, 16)])
    emb_b = torch.cat([sample(center_a, 8), sample(center_b, 8), sample(center_b, 16)])
    labels = torch.cat([torch.ones(16), torch.zeros(16)])
    dataset = TensorDataset(emb_a, emb_b, labels)
    return DataLoader(dataset, batch_size=8, shuffle=False)


def test_training_epochs_reduce_loss_on_separable_pairs() -> None:
    """Training on trivially separable pairs drives the loss down."""
    torch.manual_seed(0)
    model = build_stub_model(FlattenBackbone(2), embedding_dim=2, device=ts.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = ContrastiveLoss(margin=1.0)
    loader = _cluster_pair_loader()

    losses = [
        ts.run_epoch(model, loader, criterion, optimizer, is_train=True, epoch=epoch)
        for epoch in range(1, 16)
    ]
    assert all(np.isfinite(loss) for loss in losses)
    assert losses[-1] < losses[0]


def test_training_epoch_updates_parameters() -> None:
    """A training epoch with a violating pair changes the head parameters."""
    model = _normalizing_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    ts.run_epoch(
        model,
        _unit_vector_loader(),
        ContrastiveLoss(margin=1.0),
        optimizer,
        is_train=True,
        epoch=1,
    )
    assert any(
        not torch.equal(parameter.detach(), snapshot)
        for parameter, snapshot in zip(model.parameters(), before, strict=True)
    )


# §3 train


def test_train_writes_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_flowers: FakeFlowerDataset
) -> None:
    """One mocked training epoch writes ``latest.pt`` and ``best.pt``.

    The dataset and backbone are stubbed, so this exercises the script's
    orchestration: pair loaders, epoch loop, and checkpointing.
    """
    monkeypatch.setattr(
        SiameseNeuralNetwork,
        "_get_backbone",
        staticmethod(lambda backbone, pretrained: MeanBackbone()),
    )
    checkpoint_dir = tmp_path / "checkpoints"
    for key, value in {
        "num_epochs": 1,
        "batch_size": 2,
        "num_workers": 0,
        "checkpoint_dir": str(checkpoint_dir),
        "log_every_n": 1000,
    }.items():
        monkeypatch.setitem(ts.CONFIG, key, value)

    ts.train()

    latest = checkpoint_dir / "latest.pt"
    best = checkpoint_dir / "best.pt"
    assert latest.is_file()
    assert best.is_file()
    checkpoint = torch.load(latest, map_location="cpu")
    assert checkpoint["epoch"] == 1
    assert np.isfinite(checkpoint["val_loss"])
    assert set(checkpoint) >= {"model", "optimizer", "scheduler", "cfg"}


# §4 demo


def test_demo_prints_similarity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``demo`` loads a checkpoint and prints a similarity in ``[-1, 1]``."""
    monkeypatch.setattr(
        SiameseNeuralNetwork,
        "_get_backbone",
        staticmethod(lambda backbone, pretrained: MeanBackbone()),
    )
    source = build_stub_model(
        MeanBackbone(), embedding_dim=ts.CONFIG["embedding_dim"], device=ts.device
    )
    checkpoint_path = tmp_path / "best.pt"
    torch.save({"model": source.state_dict(), "epoch": 3}, checkpoint_path)

    image_paths = []
    for name, value in (("a.jpg", 40), ("b.jpg", 200)):
        path = tmp_path / name
        Image.fromarray(make_solid_rgb_image(value, size=48)).save(path)
        image_paths.append(str(path))

    ts.demo(str(checkpoint_path), image_paths[0], image_paths[1])

    output = capsys.readouterr().out
    assert "Loaded checkpoint from epoch 3" in output
    similarity_line = next(
        line for line in output.splitlines() if line.startswith("Similarity:")
    )
    score = float(similarity_line.split()[1])
    assert -1.0 <= score <= 1.0
