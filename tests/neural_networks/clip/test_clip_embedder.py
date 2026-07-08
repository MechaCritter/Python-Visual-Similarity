"""Behavioural tests for :class:`pyvisim.neural_networks.ClipEmbedder`.

These tests jit-load the official OpenAI ``ViT-B/32`` checkpoint, so
embeddings are reproducible across runs and instances. The checkpoint
download weighs ~338 MB on first run and is cached (and SHA-256-verified)
afterwards, so the whole module is marked ``slow`` and is deselected by
``make test-unit`` (``-m "not slow"``).

Weight-free logic (registry, cache, preprocessing, configuration) lives in
``test_checkpoints.py`` and ``test_clip_embedder_unit.py`` in the fast suite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pyvisim.neural_networks import ClipEmbedder
from pyvisim.neural_networks.clip import fetch_checkpoint, get_checkpoint_spec
from pyvisim.neural_networks.clip._checkpoints import _sha256_digest

if TYPE_CHECKING:
    from tests.conftest import ImageObj

#: Every test in this module downloads/loads the real CLIP weights.
pytestmark = pytest.mark.slow

#: The lightest official checkpoint, fixed for reproducible embeddings.
VARIANT = "ViT-B/32"
#: Joint embedding dimension of the ``ViT-B/32`` architecture.
EMBED_DIM = 512


@pytest.fixture(scope="module")
def clip_embedder() -> ClipEmbedder:
    """A ``ClipEmbedder`` with the official ``ViT-B/32`` weights.

    Module-scoped so the weights are loaded only once for the whole module.

    :returns: a ready-to-use ``ClipEmbedder``.
    """
    return ClipEmbedder(VARIANT)


# §1 checkpoint download and verification


def test_fetched_checkpoint_passes_its_published_digest() -> None:
    """The cached checkpoint hashes to the digest OpenAI publishes for it."""
    path = fetch_checkpoint(VARIANT)
    assert path.is_file()
    assert _sha256_digest(path) == get_checkpoint_spec(VARIANT).sha256


# §2 embed behaviour


def test_embed_returns_one_row_per_image(
    clip_embedder: ClipEmbedder, rgb_image: ImageObj, checkerboard_image: ImageObj
) -> None:
    """Embedding a batch yields one ``EMBED_DIM`` embedding per input image."""
    embeddings = clip_embedder.embed([rgb_image.array, checkerboard_image.array])
    assert embeddings.shape == (2, EMBED_DIM)
    assert embeddings.dtype == np.float32


def test_embed_embeddings_are_normalized_by_default(
    clip_embedder: ClipEmbedder, rgb_image: ImageObj, grayscale_image: ImageObj
) -> None:
    """With ``normalize=True`` every embedding has unit L2 norm."""
    embeddings = clip_embedder.embed([rgb_image.array, grayscale_image.array])
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_embed_is_reproducible(
    clip_embedder: ClipEmbedder, rgb_image: ImageObj
) -> None:
    """Embedding the same image twice yields identical embeddings."""
    first = clip_embedder.embed(rgb_image.array)
    second = clip_embedder.embed(rgb_image.array)
    assert np.array_equal(first, second)


def test_embed_grayscale_and_rgb_share_embedding_dim(
    clip_embedder: ClipEmbedder, grayscale_image: ImageObj, rgb_image: ImageObj
) -> None:
    """Grayscale and RGB inputs both yield ``EMBED_DIM`` embeddings.

    The preprocessing converts any input to RGB, so a single-channel
    ``(H, W)`` image (passed with ``dims="HW"``) is accepted just like an
    ``(H, W, C)`` one.
    """
    gray = clip_embedder.embed(grayscale_image.array, dims="HW")
    rgb = clip_embedder.embed(rgb_image.array)
    assert gray.shape == rgb.shape == (1, EMBED_DIM)


# §3 similarity


def test_similarity_score_identical_images_is_one(
    clip_embedder: ClipEmbedder, rgb_image: ImageObj
) -> None:
    """An image is maximally cosine-similar to itself."""
    score = clip_embedder.similarity_score(rgb_image.array, rgb_image.array)
    assert score.shape == (1, 1)
    assert np.isclose(score[0, 0], 1.0, atol=1e-5)


def test_shifted_image_more_similar_than_completely_different(
    clip_embedder: ClipEmbedder,
    checkerboard_image: ImageObj,
    horizontal_gradient_image: ImageObj,
) -> None:
    """A one-pixel shift scores higher than a completely different image.

    A checkerboard and its one-pixel-shifted copy share almost all of their
    content, so they must score more similar than the checkerboard against a
    structurally different gradient image. All three are single-channel
    ``(H, W)`` images, so ``dims="HW"`` is passed.
    """
    shifted = np.roll(checkerboard_image.array, shift=1, axis=1)
    shifted_score = clip_embedder.similarity_score(
        checkerboard_image.array, shifted, dims="HW"
    )
    different_score = clip_embedder.similarity_score(
        checkerboard_image.array, horizontal_gradient_image.array, dims="HW"
    )
    assert shifted_score[0, 0] > different_score[0, 0]
