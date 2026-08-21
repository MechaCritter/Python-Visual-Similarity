"""Serialization tests for :class:`pyvisim.neural_networks.ClipEmbedder`.

A CLIP embedder saves into the same ``.embedder`` safetensors file as every
other pyvisim embedder, weights included: a reloaded embedder must produce
bit-identical embeddings without contacting the Hugging Face Hub. All tests
run against the tiny variant registered in ``conftest.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors import safe_open

from pyvisim._base_classes import SerializableImageEmbedder
from pyvisim.neural_networks import ClipEmbedder
from pyvisim.neural_networks.clip import clip_embedder as clip_embedder_module
from pyvisim.neural_networks.clip._model import build_vision_model
from pyvisim.serialization import embedder_from_dict, embedder_to_dict

from ._tiny_clip import CONFIG, TAG


def _random_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)


def _forbid_checkpoint_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any checkpoint download fail the test."""
    monkeypatch.setattr(
        clip_embedder_module,
        "fetch_checkpoint",
        lambda variant, pretrained, cache_dir=None: pytest.fail("must not fetch"),
    )


def test_clip_embedder_is_a_serializable_embedder(embedder: ClipEmbedder) -> None:
    """The CLIP embedder inherits the ``.embedder`` file contract."""
    assert isinstance(embedder, SerializableImageEmbedder)


def test_save_to_disk_writes_a_safetensors_file(
    embedder: ClipEmbedder, tmp_path: Path
) -> None:
    """The written file is a safetensors file holding the model's weights."""
    path = embedder.save_to_disk(tmp_path / "clip")
    assert path == tmp_path / "clip.embedder"

    image_tower = build_vision_model(CONFIG, quick_gelu=True)
    with safe_open(str(path), framework="numpy") as handle:
        assert len(handle.keys()) == len(image_tower.state_dict())


def test_embeddings_are_identical_after_serialization(
    embedder: ClipEmbedder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reloaded embedder embeds images bit-identically, without a download."""
    images = [_random_image(0), _random_image(1)]
    expected = embedder.embed(images)

    path = embedder.save_to_disk(tmp_path / "clip")
    _forbid_checkpoint_fetch(monkeypatch)
    reloaded = ClipEmbedder.load_from_disk(path)

    assert np.array_equal(reloaded.embed(images), expected)


def test_scores_are_identical_after_serialization(
    embedder: ClipEmbedder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reloaded embedder scores image pairs bit-identically."""
    image1, image2 = _random_image(0), _random_image(1)
    expected = embedder.similarity_score(image1, image2)

    path = embedder.save_to_disk(tmp_path / "clip")
    _forbid_checkpoint_fetch(monkeypatch)
    reloaded = ClipEmbedder.load_from_disk(path)

    assert np.array_equal(reloaded.similarity_score(image1, image2), expected)


def test_configuration_survives_the_round_trip(
    tiny_variant: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Variant, pretrained tag, normalization and metric are all restored."""
    embedder = ClipEmbedder(
        tiny_variant,
        TAG,
        device="cpu",
        normalize=False,
        similarity_func="euclidean",
    )
    path = embedder.save_to_disk(tmp_path / "clip")
    _forbid_checkpoint_fetch(monkeypatch)
    reloaded = ClipEmbedder.load_from_disk(path)

    assert reloaded.variant == tiny_variant
    assert reloaded.pretrained == TAG
    assert reloaded.normalize is False
    assert reloaded.similarity_func_name == "euclidean"
    assert reloaded.embedding_dim == embedder.embedding_dim
    assert reloaded.device == "cpu"


def test_registry_round_trip_restores_the_embeddings(embedder: ClipEmbedder) -> None:
    """The registry used by the image store rebuilds CLIP embedders too."""
    images = [_random_image(2)]
    expected = embedder.embed(images)

    reloaded = embedder_from_dict(embedder_to_dict(embedder))

    assert isinstance(reloaded, ClipEmbedder)
    assert np.array_equal(reloaded.embed(images), expected)
