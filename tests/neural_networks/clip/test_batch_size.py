"""Tests for the batched forward passes of :class:`ClipEmbedder`.

They run against the tiny CLIP variant registered by :mod:`._tiny_clip` (see
the ``tiny_variant`` and ``embedder`` fixtures in ``conftest.py``), so nothing
is downloaded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyvisim.neural_networks import ClipEmbedder

from ._tiny_clip import TAG as _TAG

#: Batch sizes that split the sample images differently: one image per pass, an
#: exact divisor, a size leaving a shorter last pass, and the whole input.
BATCH_SIZES = [1, 2, 3, -1]


def _images(count: int = 4) -> list[np.ndarray]:
    """Build ``count`` distinguishable RGB images.

    :param count: how many images to build.
    :returns: a list of ``(48, 64, 3)`` ``uint8`` images.
    """
    rng = np.random.default_rng(0)
    return [
        rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8) for _ in range(count)
    ]


def test_the_whole_input_is_embedded_in_one_pass_by_default(
    embedder: ClipEmbedder,
) -> None:
    """CLIP keeps embedding everything it is given in a single forward pass."""
    assert embedder.batch_size == -1


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_batch_size_does_not_change_embeddings(
    embedder: ClipEmbedder, batch_size: int
) -> None:
    """Splitting the forward passes returns the same embeddings."""
    images = _images()
    one_pass = embedder.embed(images)
    embedder.set_batch_size(batch_size)
    batched = embedder.embed(images)
    assert batched.shape == one_pass.shape
    np.testing.assert_allclose(batched, one_pass, rtol=1e-6, atol=1e-6)


def test_batch_size_reaches_the_embedder_from_the_constructor(
    tiny_variant: str,
) -> None:
    """The constructor argument reaches the exposed attribute."""
    assert ClipEmbedder(tiny_variant, _TAG, device="cpu", batch_size=2).batch_size == 2


def test_embedding_nothing_still_raises(embedder: ClipEmbedder) -> None:
    """An input holding no image is an error, not an empty embedding matrix."""
    embedder.set_batch_size(2)
    with pytest.raises(ValueError, match="at least one image"):
        embedder.embed([])


def test_batch_size_survives_a_checkpoint_round_trip(
    tiny_variant: str, tmp_path: Path
) -> None:
    """A saved embedder comes back with the batch size it was saved with."""
    embedder = ClipEmbedder(tiny_variant, _TAG, device="cpu", batch_size=3)
    path = embedder.save_to_disk(tmp_path / "clip")
    assert ClipEmbedder.load_from_disk(path).batch_size == 3


def test_invalid_batch_size_is_rejected_by_the_constructor(tiny_variant: str) -> None:
    """The embedder validates the batch size like every other metric."""
    with pytest.raises(ValueError, match="batch_size"):
        ClipEmbedder(tiny_variant, _TAG, device="cpu", batch_size=0)
