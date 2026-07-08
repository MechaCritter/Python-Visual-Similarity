"""Fast unit tests for :class:`pyvisim.neural_networks.ClipEmbedder`.

The real checkpoints weigh hundreds of megabytes, so these tests stub the
checkpoint fetch and :func:`torch.jit.load` with a tiny in-memory model that
mimics the TorchScript archive's interface. Everything around the model --
preprocessing, batching, normalization, similarity scoring, configuration
validation -- is exercised for real. Tests against the real weights live in
``test_clip_embedder.py`` and run in the slow suite.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pyvisim.neural_networks import ClipEmbedder
from pyvisim.neural_networks.clip import clip_embedder as clip_embedder_module

#: Embedding width of the stub model below (3 pooled channels, duplicated).
_STUB_DIM = 6


class _StubJitModel(torch.nn.Module):
    """Stands in for the loaded TorchScript archive.

    Exposes the same interface the embedder relies on (``encode_image`` and
    ``encode_text``) and maps a preprocessed batch to a deterministic,
    deliberately non-unit-norm embedding.
    """

    def encode_image(self, batch: torch.Tensor) -> torch.Tensor:
        pooled = batch.mean(dim=(2, 3))
        return torch.cat([pooled, pooled * 2.0], dim=1) + 3.0

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("The embedder never touches the text tower.")


@pytest.fixture()
def stub_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Route checkpoint fetching and jit-loading through the stub model."""
    checkpoint = tmp_path / "ViT-B-32.pt"
    checkpoint.write_bytes(b"stub archive")
    monkeypatch.setattr(
        clip_embedder_module,
        "fetch_checkpoint",
        lambda variant, cache_dir=None: checkpoint,
    )
    monkeypatch.setattr(
        torch.jit, "load", lambda path, map_location=None: _StubJitModel()
    )
    return checkpoint


@pytest.fixture()
def embedder(stub_checkpoint: Path) -> ClipEmbedder:
    """A ``ClipEmbedder`` built on the stub model."""
    return ClipEmbedder(device="cpu")


def _random_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)


# §1 device resolution


def test_resolve_device_falls_back_to_cpu_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without CUDA, both auto-select and an explicit request yield ``cpu``."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert clip_embedder_module._resolve_device(None) == "cpu"
    assert clip_embedder_module._resolve_device("cuda") == "cpu"
    assert clip_embedder_module._resolve_device("cpu") == "cpu"


def test_resolve_device_prefers_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With CUDA available, auto-select and an explicit request yield ``cuda``."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert clip_embedder_module._resolve_device(None) == "cuda"
    assert clip_embedder_module._resolve_device("cuda") == "cuda"


# §2 embed behaviour


def test_embed_returns_one_row_per_image(embedder: ClipEmbedder) -> None:
    """Embedding a batch yields one float32 embedding per input image."""
    embeddings = embedder.embed([_random_image(0), _random_image(1)])
    assert embeddings.shape == (2, _STUB_DIM)
    assert embeddings.dtype == np.float32


def test_embed_accepts_single_image(embedder: ClipEmbedder) -> None:
    """A single (un-batched) image embeds to a ``(1, D)`` array."""
    assert embedder.embed(_random_image()).shape == (1, _STUB_DIM)


def test_embed_accepts_grayscale_images(embedder: ClipEmbedder) -> None:
    """A single-channel ``(H, W)`` image is converted to RGB and accepted."""
    gray = _random_image()[:, :, 0]
    assert embedder.embed(gray, dims="HW").shape == (1, _STUB_DIM)


def test_embed_normalizes_by_default(embedder: ClipEmbedder) -> None:
    """With ``normalize=True`` every embedding has unit L2 norm."""
    embeddings = embedder.embed([_random_image(0), _random_image(1)])
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)


def test_embed_without_normalization_keeps_raw_magnitude(
    stub_checkpoint: Path,
) -> None:
    """With ``normalize=False`` embeddings are not forced to unit norm."""
    embedder = ClipEmbedder(device="cpu", normalize=False)
    embeddings = embedder.embed(_random_image())
    assert not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-3)


def test_embed_no_images_raises(embedder: ClipEmbedder) -> None:
    """An empty iterable of images raises a ``ValueError``."""
    with pytest.raises(ValueError, match="at least one image"):
        embedder.embed([])


# §3 similarity


def test_similarity_score_identical_images_is_one(embedder: ClipEmbedder) -> None:
    """An image is maximally cosine-similar to itself."""
    image = _random_image()
    score = embedder.similarity_score(image, image)
    assert score.shape == (1, 1)
    assert np.isclose(score[0, 0], 1.0, atol=1e-5)


# §4 configuration, properties and repr


def test_properties_reflect_constructor_arguments(stub_checkpoint: Path) -> None:
    """The read-only properties expose the configured values."""
    embedder = ClipEmbedder(
        "ViT-L/14", device="cpu", normalize=False, similarity_func="euclidean"
    )
    assert embedder.variant == "ViT-L/14"
    assert embedder.device == "cpu"
    assert embedder.normalize is False
    assert embedder.embedding_dim == 768
    assert embedder.image_size == 224


def test_unknown_variant_raises_before_any_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unsupported variant fails fast, without touching the network."""
    monkeypatch.setattr(
        clip_embedder_module,
        "fetch_checkpoint",
        lambda variant, cache_dir=None: pytest.fail("must not fetch"),
    )
    with pytest.raises(ValueError, match="Unsupported CLIP variant"):
        ClipEmbedder("ViT-B/64")


def test_unknown_similarity_func_raises_before_any_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unsupported similarity metric fails fast, before the fetch."""
    monkeypatch.setattr(
        clip_embedder_module,
        "fetch_checkpoint",
        lambda variant, cache_dir=None: pytest.fail("must not fetch"),
    )
    with pytest.raises(ValueError, match="similarity function"):
        ClipEmbedder(similarity_func="dot")


def test_repr_names_class_and_variant(embedder: ClipEmbedder) -> None:
    """``repr`` names the embedder class and its CLIP variant."""
    text = repr(embedder)
    assert "ClipEmbedder(" in text
    assert "variant=ViT-B/32" in text
