"""Full-dataset end-to-end test of the Siamese training script.

Runs the complete :func:`train` entry point on the full Oxford Flowers
dataset with the script's own configuration (20 epochs), then exercises
:func:`demo` on the resulting best checkpoint and verifies that the learned
metric ranks same-class flower pairs above cross-class pairs.

Everything here downloads data/weights when missing and trains for real, so
the module is marked ``slow`` and only runs via ``make test-slow``.
"""

from __future__ import annotations

from pathlib import Path
from statistics import mean

import numpy as np
import pytest
import torch
from PIL import Image

import pyvisim.neural_networks.scripts.train_siamese_neural_network as ts
from pyvisim.datasets import OxfordFlowerDataset
from pyvisim.neural_networks import ContrastiveSiameseNetwork
from pyvisim.typing import UInt8NumpyArray

pytestmark = pytest.mark.slow

#: Number of classes sampled for the learned-metric ranking check.
RANKING_CLASSES = 8


def _load_rgb(path: str) -> UInt8NumpyArray:
    """Load an image file as an RGB ``uint8`` array.

    :param path: Path to the image file.
    :return: The image as an ``(H, W, 3)`` array.
    """
    return np.asarray(Image.open(path).convert("RGB"))


def _paths_by_label(dataset: OxfordFlowerDataset) -> dict[int, list[str]]:
    """Group a dataset's image paths by class label.

    :param dataset: The dataset to index.
    :return: Mapping from label to the image paths of that class.
    """
    by_label: dict[int, list[str]] = {}
    for path, label in zip(dataset.image_paths, dataset.labels, strict=True):
        by_label.setdefault(label, []).append(path)
    return by_label


def test_full_training_and_demo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The whole train script runs on the full dataset and ``demo`` works.

    Checks, in order: training completes and checkpoints both ``latest.pt``
    and ``best.pt``; the checkpoint payload is well-formed; ``demo`` loads the
    best checkpoint and prints a similarity in ``[-1, 1]`` for two same-class
    validation images; and the trained embedding scores same-class pairs
    higher than cross-class pairs on average.
    """
    checkpoint_dir = tmp_path / "checkpoints"
    monkeypatch.setitem(ts.CONFIG, "checkpoint_dir", str(checkpoint_dir))

    ts.train()

    latest = checkpoint_dir / "latest.pt"
    best = checkpoint_dir / "best.pt"
    assert latest.is_file()
    assert best.is_file()

    checkpoint = torch.load(best, map_location="cpu")
    assert 1 <= checkpoint["epoch"] <= ts.CONFIG["num_epochs"]
    assert np.isfinite(checkpoint["val_loss"])
    assert set(checkpoint) >= {"model", "optimizer", "scheduler", "cfg"}

    # §2 demo on two same-class validation images
    validation = OxfordFlowerDataset(purpose="validation")
    by_label = _paths_by_label(validation)
    same_class_paths = next(paths for paths in by_label.values() if len(paths) >= 2)
    capsys.readouterr()
    ts.demo(str(best), same_class_paths[0], same_class_paths[1])
    output = capsys.readouterr().out
    assert "Loaded checkpoint from epoch" in output
    similarity_line = next(
        line for line in output.splitlines() if line.startswith("Similarity:")
    )
    score = float(similarity_line.split()[1])
    assert -1.0 <= score <= 1.0

    # §3 the learned metric ranks same-class pairs above cross-class pairs
    model = ContrastiveSiameseNetwork(
        embedding_dim=ts.CONFIG["embedding_dim"], device=ts.device
    )
    model.load_state_dict(torch.load(best, map_location=ts.device)["model"])
    chosen = sorted(label for label, paths in by_label.items() if len(paths) >= 2)[
        :RANKING_CLASSES
    ]
    same_scores = [
        float(
            model.similarity_score(
                _load_rgb(by_label[label][0]), _load_rgb(by_label[label][1])
            ).item()
        )
        for label in chosen
    ]
    cross_scores = [
        float(
            model.similarity_score(
                _load_rgb(by_label[label_a][0]), _load_rgb(by_label[label_b][0])
            ).item()
        )
        for label_a, label_b in zip(chosen, chosen[1:] + chosen[:1], strict=True)
    ]
    assert mean(same_scores) > mean(cross_scores)
