"""Fast unit tests for :class:`pyvisim.neural_networks.ClipEmbedder`.

The tests run against the tiny CLIP variant registered by
:mod:`._tiny_clip` (see the ``tiny_variant`` and ``embedder`` fixtures in
``conftest.py``), so model building, weight loading, preprocessing, batching,
normalization, similarity scoring and configuration validation all run
through the production code path without downloading anything. Tests against
the real published weights live in ``test_clip_embedder.py`` and run in the
slow suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyvisim.neural_networks import ClipEmbedder
from pyvisim.neural_networks.clip import clip_embedder as clip_embedder_module

from ._tiny_clip import CONFIG as _CONFIG
from ._tiny_clip import TAG as _TAG
from ._tiny_clip import VARIANT as _VARIANT


def _random_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)


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
