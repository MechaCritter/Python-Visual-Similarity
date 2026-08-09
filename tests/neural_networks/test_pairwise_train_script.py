"""Unit tests for the pairwise Siamese training script.

Covers :func:`run_epoch`, :func:`train` and :func:`demo` from
``pyvisim.neural_networks.scripts.train_pairwise_siamese_network``. The pair
mining itself lives in the shared ``_pairs`` module and is covered by the
contrastive script's tests; here the Oxford Flowers dataset is replaced by an
in-memory fake and the backbone by deterministic stubs, so everything runs
without network access.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

import pyvisim.neural_networks.scripts._pairs as pair_data
import pyvisim.neural_networks.scripts.train_pairwise_siamese_network as tsp
from pyvisim.neural_networks import PairwiseSiameseNetwork

from ._stubs import (
    FakeFlowerDataset,
    FlattenBackbone,
    MeanBackbone,
    build_stub_pairwise_model,
    install_head,
    install_scorer,
    make_identity_head,
    make_solid_rgb_image,
)

#: Four distinguishable images: two of class 1, two of class 2.
_IMAGES = [make_solid_rgb_image(value, size=48) for value in (10, 60, 110, 160)]
_LABELS = [1, 1, 2, 2]


# §1 run_epoch


def _identical_pair_loader() -> DataLoader[tuple[torch.Tensor, ...]]:
    """Loader with four identical input pairs in two batches.

    Every pair has zero L1 feature distance, so with a zero-bias scorer each
    logit is exactly 0 and the binary cross-entropy of every pair is ``ln 2``
    regardless of its label. The epoch average is therefore exactly ``ln 2``.
    Labels are grouped per batch (first batch all-positive, second batch
    all-negative) so that the bias gradients ``sigmoid(z) - y`` within a batch
    do not cancel and a training step must move the parameters.

    :return: A deterministic, unshuffled two-batch loader.
    """
    x_a = torch.zeros(4, 2)
    x_b = torch.zeros(4, 2)
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    dataset = TensorDataset(x_a, x_b, labels)
    return DataLoader(dataset, batch_size=2, shuffle=False)


def _zero_bias_model() -> PairwiseSiameseNetwork:
    """Model whose pair logit is exactly the weighted L1 feature distance.

    :return: Stub model on the script's device with an identity head and a
        scoring layer of unit weights and zero bias.
    """
    model = build_stub_pairwise_model(
        FlattenBackbone(2), embedding_dim=2, device=tsp.device
    )
    install_head(model, make_identity_head(2))
    install_scorer(model, weights=[1.0, 1.0], bias=0.0)
    return model


def test_run_epoch_returns_hand_computed_average_loss() -> None:
    """An eval epoch over identical pairs returns exactly ``ln 2``."""
    model = _zero_bias_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    loss = tsp.run_epoch(
        model,
        _identical_pair_loader(),
        torch.nn.BCEWithLogitsLoss(),
        optimizer,
        is_train=False,
        epoch=1,
    )
    assert isinstance(loss, float)
    assert loss == pytest.approx(math.log(2.0), abs=1e-6)


def test_eval_epoch_does_not_update_parameters() -> None:
    """Without training, model parameters stay bitwise identical."""
    model = _zero_bias_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    tsp.run_epoch(
        model,
        _identical_pair_loader(),
        torch.nn.BCEWithLogitsLoss(),
        optimizer,
        is_train=False,
        epoch=1,
    )
    for parameter, snapshot in zip(model.parameters(), before, strict=True):
        assert torch.equal(parameter.detach(), snapshot)


def test_run_epoch_sets_model_mode() -> None:
    """The epoch runner switches the model to train or eval mode."""
    model = _zero_bias_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    criterion = torch.nn.BCEWithLogitsLoss()
    tsp.run_epoch(
        model, _identical_pair_loader(), criterion, optimizer, is_train=True, epoch=1
    )
    assert model.training is True
    tsp.run_epoch(
        model, _identical_pair_loader(), criterion, optimizer, is_train=False, epoch=1
    )
    assert model.training is False


def test_training_epoch_updates_parameters() -> None:
    """A training epoch changes the parameters.

    The first batch is all-positive, so its mean bias gradient is
    ``sigmoid(0) - 1 = -0.5`` and the scoring layer's bias must move.
    """
    model = _zero_bias_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    tsp.run_epoch(
        model,
        _identical_pair_loader(),
        torch.nn.BCEWithLogitsLoss(),
        optimizer,
        is_train=True,
        epoch=1,
    )
    assert any(
        not torch.equal(parameter.detach(), snapshot)
        for parameter, snapshot in zip(model.parameters(), before, strict=True)
    )


def _separable_pair_loader() -> DataLoader[tuple[torch.Tensor, ...]]:
    """Loader of trivially separable pairs.

    Positive pairs are identical inputs (zero L1 feature distance), negative
    pairs are far apart, so the scoring layer alone can separate them.

    :return: A 32-pair loader with batch size 8.
    """
    x_a = torch.zeros(32, 2)
    x_b = torch.cat([torch.zeros(16, 2), 4.0 * torch.ones(16, 2)])
    labels = torch.cat([torch.ones(16), torch.zeros(16)])
    dataset = TensorDataset(x_a, x_b, labels)
    return DataLoader(dataset, batch_size=8, shuffle=False)


def test_training_epochs_reduce_loss_on_separable_pairs() -> None:
    """Training on trivially separable pairs drives the BCE loss down."""
    torch.manual_seed(0)
    model = build_stub_pairwise_model(
        FlattenBackbone(2), embedding_dim=2, device=tsp.device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = torch.nn.BCEWithLogitsLoss()
    loader = _separable_pair_loader()

    losses = [
        tsp.run_epoch(model, loader, criterion, optimizer, is_train=True, epoch=epoch)
        for epoch in range(1, 16)
    ]
    assert all(np.isfinite(loss) for loss in losses)
    assert losses[-1] < losses[0]


# §2 train


def test_train_writes_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One mocked training epoch writes ``latest.pt`` and ``best.pt``.

    The dataset and backbone are stubbed, so this exercises the script's
    orchestration: pair loaders, epoch loop, and checkpointing.
    """
    fake = FakeFlowerDataset(_IMAGES, _LABELS)
    monkeypatch.setattr(
        pair_data, "OxfordFlowerDataset", lambda transform=None, purpose="train": fake
    )
    monkeypatch.setattr(
        PairwiseSiameseNetwork,
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
        monkeypatch.setitem(tsp.CONFIG, key, value)

    tsp.train()

    latest = checkpoint_dir / "latest.pt"
    best = checkpoint_dir / "best.pt"
    assert latest.is_file()
    assert best.is_file()
    checkpoint = torch.load(latest, map_location="cpu")
    assert checkpoint["epoch"] == 1
    assert np.isfinite(checkpoint["val_loss"])
    assert set(checkpoint) >= {"model", "optimizer", "scheduler", "cfg"}


# §3 demo


def test_demo_prints_same_class_probability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``demo`` loads a checkpoint and prints a probability in ``[0, 1]``."""
    monkeypatch.setattr(
        PairwiseSiameseNetwork,
        "_get_backbone",
        staticmethod(lambda backbone, pretrained: MeanBackbone()),
    )
    source = build_stub_pairwise_model(
        MeanBackbone(), embedding_dim=tsp.CONFIG["embedding_dim"], device=tsp.device
    )
    checkpoint_path = tmp_path / "best.pt"
    torch.save({"model": source.state_dict(), "epoch": 3}, checkpoint_path)

    image_paths = []
    for name, value in (("a.jpg", 40), ("b.jpg", 200)):
        path = tmp_path / name
        Image.fromarray(make_solid_rgb_image(value, size=48)).save(path)
        image_paths.append(str(path))

    tsp.demo(str(checkpoint_path), image_paths[0], image_paths[1])

    output = capsys.readouterr().out
    assert "Loaded checkpoint from epoch 3" in output
    probability_line = next(
        line for line in output.splitlines() if "Same-class probability:" in line
    )
    probability = float(probability_line.split(":")[1].split()[0])
    assert 0.0 <= probability <= 1.0
