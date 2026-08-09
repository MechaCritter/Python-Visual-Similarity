"""Behavioural tests for :class:`pyvisim.neural_networks.ClipEmbedder`.

These tests load real published CLIP weights (the original OpenAI
``ViT-B-32`` checkpoint and its LAION-2B retraining), so embeddings are
reproducible across runs and instances. Each checkpoint weighs a few hundred
megabytes on first download and is cached by huggingface_hub afterwards, so
the whole module is marked ``slow`` and is deselected by ``make test-unit``
(``-m "not slow"``).

Weight-free logic (registry, model architecture, preprocessing,
configuration) lives in ``test_registry.py``, ``test_model.py`` and
``test_clip_embedder_unit.py`` in the fast suite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pyvisim.neural_networks import ClipEmbedder
from pyvisim.neural_networks.clip import fetch_checkpoint

if TYPE_CHECKING:
    from tests.conftest import ImageObj

#: Every test in this module downloads/loads real CLIP weights.
pytestmark = pytest.mark.slow

#: The lightest variant, fixed for reproducible embeddings.
VARIANT = "ViT-B-32"
#: The original OpenAI weights of the variant.
PRETRAINED = "openai"
#: Joint embedding dimension of the ``ViT-B-32`` architecture.
EMBED_DIM = 512


@pytest.fixture(scope="module")
def clip_embedder() -> ClipEmbedder:
    """A ``ClipEmbedder`` with the official OpenAI ``ViT-B-32`` weights.

    Module-scoped so the weights are loaded only once for the whole module.

    :returns: a ready-to-use ``ClipEmbedder``.
    """
    return ClipEmbedder(VARIANT, PRETRAINED)


# §1 checkpoint download


def test_fetched_checkpoint_is_a_local_safetensors_file() -> None:
    """The checkpoint resolves to a safetensors file in the local cache."""
    path = fetch_checkpoint(VARIANT, PRETRAINED)
    assert path.is_file()
    assert path.suffix == ".safetensors"


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


# §4 pretrained tags select different weights


def test_laion2b_tag_loads_and_differs_from_openai(
    clip_embedder: ClipEmbedder, rgb_image: ImageObj
) -> None:
    """The LAION-2B tag loads the GELU tower and yields its own embeddings."""
    laion_embedder = ClipEmbedder(VARIANT, "laion2b_s34b_b79k")
    embeddings = laion_embedder.embed(rgb_image.array)
    assert embeddings.shape == (1, EMBED_DIM)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)
    assert not np.allclose(embeddings, clip_embedder.embed(rgb_image.array))
