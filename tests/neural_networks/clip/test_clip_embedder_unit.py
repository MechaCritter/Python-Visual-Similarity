"""Fast unit tests for :class:`pyvisim.neural_networks.ClipEmbedder`.

The real checkpoints weigh hundreds of megabytes, so a tiny CLIP variant is
registered on top of the real registry and its checkpoint is written to disk
as a real safetensors file. Everything else -- model building, weight
loading, preprocessing, batching, normalization, similarity scoring and
configuration validation -- runs through the production code path. Tests
against the real published weights live in ``test_clip_embedder.py`` and run
in the slow suite.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from pyvisim.neural_networks import ClipEmbedder
from pyvisim.neural_networks.clip import CheckpointSpec, VisionConfig
from pyvisim.neural_networks.clip import _registry as registry_module
from pyvisim.neural_networks.clip import clip_embedder as clip_embedder_module
from pyvisim.neural_networks.clip._model import build_vision_model

#: Name, tag and architecture of the tiny test-only variant.
_VARIANT = "tiny-ViT"
_TAG = "test"
_CONFIG = VisionConfig(
    embed_dim=8, image_size=32, width=16, layers=2, patch_size=16, head_width=8
)


@pytest.fixture()
def tiny_variant(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    """Register a tiny CLIP variant whose checkpoint lives in ``tmp_path``."""
    checkpoint = tmp_path / "open_clip_model.safetensors"
    torch.manual_seed(0)
    model = build_vision_model(_CONFIG, quick_gelu=True)
    save_file(
        {f"visual.{key}": value for key, value in model.state_dict().items()},
        str(checkpoint),
    )
    spec = CheckpointSpec(repo_id="pyvisim-tests/tiny-clip", quick_gelu=True)
    monkeypatch.setitem(registry_module.MODEL_CONFIGS, _VARIANT, _CONFIG)
    monkeypatch.setitem(registry_module.PRETRAINED, _VARIANT, {_TAG: spec})
    monkeypatch.setattr(
        clip_embedder_module,
        "fetch_checkpoint",
        lambda variant, pretrained, cache_dir=None: checkpoint,
    )
    return _VARIANT


@pytest.fixture()
def embedder(tiny_variant: str) -> ClipEmbedder:
    """A ``ClipEmbedder`` built on the tiny variant."""
    return ClipEmbedder(tiny_variant, _TAG, device="cpu")


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
    assert embeddings.shape == (2, _CONFIG.embed_dim)
    assert embeddings.dtype == np.float32


def test_embed_accepts_single_image(embedder: ClipEmbedder) -> None:
    """A single (un-batched) image embeds to a ``(1, D)`` array."""
    assert embedder.embed(_random_image()).shape == (1, _CONFIG.embed_dim)


def test_embed_accepts_grayscale_images(embedder: ClipEmbedder) -> None:
    """A single-channel ``(H, W)`` image is converted to RGB and accepted."""
    gray = _random_image()[:, :, 0]
    assert embedder.embed(gray, dims="HW").shape == (1, _CONFIG.embed_dim)


def test_embed_normalizes_by_default(embedder: ClipEmbedder) -> None:
    """With ``normalize=True`` every embedding has unit L2 norm."""
    embeddings = embedder.embed([_random_image(0), _random_image(1)])
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)


def test_embed_without_normalization_keeps_raw_magnitude(
    tiny_variant: str,
) -> None:
    """With ``normalize=False`` embeddings are not forced to unit norm."""
    embedder = ClipEmbedder(tiny_variant, _TAG, device="cpu", normalize=False)
    embeddings = embedder.embed(_random_image())
    assert not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-3)


def test_embed_is_deterministic(embedder: ClipEmbedder) -> None:
    """Embedding the same image twice yields identical embeddings."""
    image = _random_image()
    assert np.array_equal(embedder.embed(image), embedder.embed(image))


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


def test_properties_reflect_constructor_arguments(tiny_variant: str) -> None:
    """The read-only properties expose the configured values."""
    embedder = ClipEmbedder(
        tiny_variant, _TAG, device="cpu", normalize=False, similarity_func="euclidean"
    )
    assert embedder.variant == tiny_variant
    assert embedder.pretrained == _TAG
    assert embedder.device == "cpu"
    assert embedder.normalize is False
    assert embedder.embedding_dim == _CONFIG.embed_dim
    assert embedder.image_size == _CONFIG.image_size


def test_unknown_variant_raises_before_any_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unsupported variant fails fast, without touching the network."""
    monkeypatch.setattr(
        clip_embedder_module,
        "fetch_checkpoint",
        lambda variant, pretrained, cache_dir=None: pytest.fail("must not fetch"),
    )
    with pytest.raises(ValueError, match="Unsupported CLIP variant"):
        ClipEmbedder("ViT-B/64")


def test_unknown_pretrained_tag_raises_before_any_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unsupported pretrained tag fails fast, listing the valid tags."""
    monkeypatch.setattr(
        clip_embedder_module,
        "fetch_checkpoint",
        lambda variant, pretrained, cache_dir=None: pytest.fail("must not fetch"),
    )
    with pytest.raises(ValueError, match="Unsupported pretrained tag"):
        ClipEmbedder("ViT-H-14")  # ViT-H-14 has no "openai" checkpoint


def test_unknown_similarity_func_raises_before_any_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unsupported similarity metric fails fast, before the fetch."""
    monkeypatch.setattr(
        clip_embedder_module,
        "fetch_checkpoint",
        lambda variant, pretrained, cache_dir=None: pytest.fail("must not fetch"),
    )
    with pytest.raises(ValueError, match="similarity function"):
        ClipEmbedder(similarity_func="dot")


def test_repr_names_class_variant_and_pretrained_tag(
    embedder: ClipEmbedder,
) -> None:
    """``repr`` names the embedder class, its variant and its weights."""
    text = repr(embedder)
    assert "ClipEmbedder(" in text
    assert f"variant={_VARIANT}" in text
    assert f"pretrained={_TAG}" in text
