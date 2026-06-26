"""Behavioural and serialization tests for :class:`pyvisim.encoders.CLIPEncoder`.

These tests build the encoder with the fixed pretrained ``openai`` ViT-B-32
weights, so encodings are reproducible across runs and instances (which lets the
save/load test assert identical encodings). Loading the weights downloads
~338 MB on first run and is cached afterwards, so the whole module is marked
``slow`` and is deselected by ``make test-unit`` (``-m "not slow"``).

Weight-free logic (device resolution and the lazy import guard) lives in
``test_clip_unit.py`` and runs in the fast suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pyvisim.encoders import CLIPEncoder, VLADEncoder

if TYPE_CHECKING:
    from tests.conftest import ImageObj

#: Every test in this module downloads/loads real CLIP weights.
pytestmark = pytest.mark.slow

#: Fixed pretrained weight tag used for reproducible encodings.
#: `openai` weights are a little lighter compared to the default
#: laion2b_s34b_b79k weights, so this is used instead to make the
#: test faster
PRETRAINED = "openai"
#: Joint embedding dimension of the ``ViT-B-32`` architecture.
EMBED_DIM = 512


@pytest.fixture(scope="module")
def clip_encoder() -> CLIPEncoder:
    """A ``CLIPEncoder`` with the fixed pretrained ``openai`` ViT-B-32 weights.

    Module-scoped so the weights are loaded only once for the whole module.

    :returns: a ready-to-use ``CLIPEncoder``.
    """
    return CLIPEncoder(pretrained=PRETRAINED)


# §1 encode behaviour


def test_encode_returns_one_row_per_image(
    clip_encoder: CLIPEncoder, rgb_image: ImageObj, checkerboard_image: ImageObj
) -> None:
    """Encoding a batch yields one ``EMBED_DIM`` embedding per input image."""
    embeddings = clip_encoder.encode([rgb_image.array, checkerboard_image.array])
    assert embeddings.shape == (2, EMBED_DIM)
    assert embeddings.dtype == np.float32


def test_encode_accepts_single_image(
    clip_encoder: CLIPEncoder, rgb_image: ImageObj
) -> None:
    """A single (un-batched) image encodes to a ``(1, EMBED_DIM)`` array."""
    embeddings = clip_encoder.encode(rgb_image.array)
    assert embeddings.shape == (1, EMBED_DIM)


def test_encode_embeddings_are_normalized_by_default(
    clip_encoder: CLIPEncoder, rgb_image: ImageObj, grayscale_image: ImageObj
) -> None:
    """With ``normalize=True`` every embedding has unit L2 norm."""
    embeddings = clip_encoder.encode([rgb_image.array, grayscale_image.array])
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_encode_without_normalization_keeps_raw_magnitude(
    rgb_image: ImageObj,
) -> None:
    """With ``normalize=False`` embeddings are not forced to unit norm."""
    encoder = CLIPEncoder(pretrained=PRETRAINED, normalize=False)
    embeddings = encoder.encode(rgb_image.array)
    norms = np.linalg.norm(embeddings, axis=1)
    assert not np.allclose(norms, 1.0, atol=1e-3)


def test_encode_is_reproducible(clip_encoder: CLIPEncoder, rgb_image: ImageObj) -> None:
    """Encoding the same image twice yields identical embeddings."""
    first = clip_encoder.encode(rgb_image.array)
    second = clip_encoder.encode(rgb_image.array)
    assert np.array_equal(first, second)


def test_encode_grayscale_and_rgb_share_embedding_dim(
    clip_encoder: CLIPEncoder, grayscale_image: ImageObj, rgb_image: ImageObj
) -> None:
    """Grayscale and RGB inputs both yield ``EMBED_DIM`` embeddings.

    The open_clip preprocessing transform converts any input to RGB, so a
    single-channel ``(H, W)`` image (passed with ``dims="HW"``) is accepted just
    like an ``(H, W, C)`` one.
    """
    gray = clip_encoder.encode(grayscale_image.array, dims="HW")
    rgb = clip_encoder.encode(rgb_image.array)
    assert gray.shape == rgb.shape == (1, EMBED_DIM)


# §2 similarity


def test_default_similarity_func_is_cosine(clip_encoder: CLIPEncoder) -> None:
    """The encoder defaults to the cosine similarity metric."""
    assert clip_encoder.similarity_func_name == "cosine"


def test_similarity_score_identical_images_is_one(
    clip_encoder: CLIPEncoder, rgb_image: ImageObj
) -> None:
    """An image is maximally cosine-similar to itself."""
    score = clip_encoder.similarity_score(rgb_image.array, rgb_image.array)
    assert score.shape == (1, 1)
    assert np.isclose(score[0, 0], 1.0, atol=1e-5)


def test_shifted_image_more_similar_than_completely_different(
    clip_encoder: CLIPEncoder,
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
    shifted_score = clip_encoder.similarity_score(
        checkerboard_image.array, shifted, dims="HW"
    )
    different_score = clip_encoder.similarity_score(
        checkerboard_image.array, horizontal_gradient_image.array, dims="HW"
    )
    assert shifted_score[0, 0] > different_score[0, 0]


# §3 configuration, properties and repr


def test_properties_reflect_constructor_arguments() -> None:
    """The read-only properties expose the configured values."""
    encoder = CLIPEncoder(
        pretrained=PRETRAINED, normalize=False, similarity_func="euclidean"
    )
    assert encoder.model_name == "ViT-B-32"
    assert encoder.pretrained == PRETRAINED
    assert encoder.normalize is False
    assert encoder.similarity_func_name == "euclidean"
    assert encoder.device in {"cpu", "cuda"}


def test_repr_names_class_and_model(clip_encoder: CLIPEncoder) -> None:
    """``repr`` names the encoder class and its model architecture."""
    text = repr(clip_encoder)
    assert "CLIPEncoder(" in text
    assert "model_name=ViT-B-32" in text


# §4 serialization


def test_to_dict_contains_required_state_keys(clip_encoder: CLIPEncoder) -> None:
    """``to_dict`` emits every key required to validate and rebuild the file."""
    state = clip_encoder.to_dict()
    assert clip_encoder._STATE_KEYS.issubset(state)
    assert state["encoder_class"] == "CLIPEncoder"


def test_from_dict_roundtrips_configuration(clip_encoder: CLIPEncoder) -> None:
    """``from_dict(to_dict())`` restores the encoder configuration."""
    restored = CLIPEncoder.from_dict(clip_encoder.to_dict())
    assert restored.model_name == clip_encoder.model_name
    assert restored.pretrained == clip_encoder.pretrained
    assert restored.normalize == clip_encoder.normalize
    assert restored.similarity_func_name == clip_encoder.similarity_func_name


def test_save_appends_suffix(clip_encoder: CLIPEncoder, tmp_path: Path) -> None:
    """Saving appends the ``.encoder`` suffix and writes the file."""
    path = clip_encoder.save_to_disk(tmp_path / "clip")
    assert str(path).endswith(".encoder")
    assert path.exists()


def test_save_load_produces_identical_encodings(
    clip_encoder: CLIPEncoder, tmp_path: Path, rgb_image: ImageObj
) -> None:
    """A saved-and-reloaded encoder produces the same encodings.

    With fixed pretrained weights the reconstructed model is identical, so the
    encodings must match exactly (up to float tolerance).
    """
    path = clip_encoder.save_to_disk(tmp_path / "clip")
    loaded = CLIPEncoder.load_from_disk(path)
    assert np.allclose(
        loaded.encode(rgb_image.array),
        clip_encoder.encode(rgb_image.array),
        atol=1e-5,
    )


def test_load_invalid_file_raises(tmp_path: Path) -> None:
    """Loading a file that is not a valid encoder raises ``ValueError``."""
    bad = tmp_path / "bad.encoder"
    bad.write_bytes(b"not a safetensors file")
    with pytest.raises(ValueError, match="not a valid .encoder file"):
        CLIPEncoder.load_from_disk(bad)


def test_load_with_incompatible_encoder_raises(
    clip_encoder: CLIPEncoder, tmp_path: Path
) -> None:
    """A CLIP file is rejected by an encoder expecting different state keys.

    A ``VLADEncoder`` requires clustering-specific keys that a CLIP file does
    not carry, so the file fails validation before any class-name check.
    """
    path = clip_encoder.save_to_disk(tmp_path / "clip")
    with pytest.raises(ValueError, match="not a valid .encoder file"):
        VLADEncoder.load_from_disk(path)
