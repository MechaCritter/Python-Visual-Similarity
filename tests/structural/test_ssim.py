"""Tests for :class:`pyvisim.structural.SSIM`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pyvisim.structural import SSIM

if TYPE_CHECKING:
    from tests.conftest import ImageObj


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------


def test_identical_images_score_one(
    identical_image_pair: tuple[np.ndarray, np.ndarray],
) -> None:
    """Two pixel-identical images reach the maximum score of 1."""
    image_a, image_b = identical_image_pair
    scores = SSIM().similarity_score(image_a, image_b)
    assert scores.shape == (1, 1)
    assert scores[0, 0] == pytest.approx(1.0, abs=1e-12)


def test_symmetry(
    checkerboard_image: ImageObj, noisy_checkerboard_image: ImageObj
) -> None:
    """SSIM is symmetric in its two arguments."""
    metric = SSIM()
    forward = metric.similarity_score(
        checkerboard_image.array, noisy_checkerboard_image.array
    )
    backward = metric.similarity_score(
        noisy_checkerboard_image.array, checkerboard_image.array
    )
    assert forward[0, 0] == pytest.approx(backward[0, 0], abs=1e-12)


def test_more_noise_lowers_score(
    checkerboard_image: ImageObj,
    noisy_image_pair: tuple[ImageObj, ImageObj],
    very_noisy_image_pair: tuple[ImageObj, ImageObj],
) -> None:
    """Scores decrease monotonically with the noise level of the distortion."""
    metric = SSIM()
    clean = checkerboard_image.array
    mild = metric.similarity_score(clean, noisy_image_pair[0].array)[0, 0]
    heavy = metric.similarity_score(clean, very_noisy_image_pair[0].array)[0, 0]
    assert 0.0 < heavy < mild < 1.0


def test_black_vs_white_scores_near_zero(
    black_image: ImageObj, white_image: ImageObj
) -> None:
    """Images sharing no luminance at all score close to 0."""
    score = SSIM().similarity_score(black_image.array, white_image.array)[0, 0]
    assert abs(score) < 0.01


def test_replicated_rgb_matches_grayscale(
    checkerboard_image: ImageObj, noisy_checkerboard_image: ImageObj
) -> None:
    """An RGB image with three identical channels scores like its grayscale."""
    gray_score = SSIM().similarity_score(
        checkerboard_image.array, noisy_checkerboard_image.array
    )
    rgb_a = np.stack([checkerboard_image.array] * 3, axis=-1)
    rgb_b = np.stack([noisy_checkerboard_image.array] * 3, axis=-1)
    rgb_score = SSIM().similarity_score(rgb_a, rgb_b)
    assert rgb_score[0, 0] == pytest.approx(gray_score[0, 0], abs=1e-12)


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def test_score_matrix_shape(
    checkerboard_image: ImageObj,
    noisy_checkerboard_image: ImageObj,
    stripes_image: ImageObj,
) -> None:
    """Batches of N and M images yield an (N, M) score matrix."""
    images1 = [
        checkerboard_image.array,
        noisy_checkerboard_image.array,
        stripes_image.array,
    ]
    images2 = [checkerboard_image.array, stripes_image.array]
    scores = SSIM().similarity_score(images1, images2)
    assert scores.shape == (3, 2)
    assert scores[0, 0] == pytest.approx(1.0, abs=1e-12)
    assert scores[2, 1] == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_batch_size_does_not_change_scores(
    checkerboard_image: ImageObj,
    noisy_checkerboard_image: ImageObj,
    stripes_image: ImageObj,
    batch_size: int,
) -> None:
    """Chunked scoring returns exactly the same matrix as one big batch."""
    images1 = [
        checkerboard_image.array,
        noisy_checkerboard_image.array,
        stripes_image.array,
    ]
    images2 = [checkerboard_image.array, noisy_checkerboard_image.array]
    whole = SSIM(batch_size=-1).similarity_score(images1, images2)
    chunked = SSIM(batch_size=batch_size).similarity_score(images1, images2)
    np.testing.assert_array_equal(whole, chunked)


def test_batched_array_input(
    checkerboard_image: ImageObj, noisy_checkerboard_image: ImageObj
) -> None:
    """Stacked arrays with a batch axis are split via the ``dims`` labels."""
    stacked1 = np.stack([checkerboard_image.array, noisy_checkerboard_image.array])
    stacked2 = np.stack([checkerboard_image.array])
    scores = SSIM().similarity_score(stacked1, stacked2, dims="BHW")
    assert scores.shape == (2, 1)
    assert scores[0, 0] == pytest.approx(1.0, abs=1e-12)


def test_value_range_rescaling(checkerboard_image: ImageObj) -> None:
    """Float images in [0, 1] score like their uint8 counterparts."""
    as_float = checkerboard_image.array.astype(np.float64) / 255.0
    from_float = SSIM().similarity_score(as_float, as_float, value_range=(0.0, 1.0))
    assert from_float[0, 0] == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_image_smaller_than_window_raises(tiny_image: ImageObj) -> None:
    """An image smaller than the Gaussian window is rejected."""
    with pytest.raises(ValueError, match="smaller than the 11x11 SSIM window"):
        SSIM().similarity_score(tiny_image.array, tiny_image.array)


def test_mismatched_shapes_between_batches_raise(
    checkerboard_image: ImageObj, non_square_image: ImageObj
) -> None:
    """Comparing images of different sizes is rejected."""
    with pytest.raises(ValueError, match="same shape"):
        SSIM().similarity_score(checkerboard_image.array, non_square_image.array)


def test_mixed_shapes_within_batch_raise(
    checkerboard_image: ImageObj, non_square_image: ImageObj
) -> None:
    """A batch mixing image sizes is rejected."""
    with pytest.raises(ValueError, match="same shape"):
        SSIM().similarity_score(
            [checkerboard_image.array, non_square_image.array],
            checkerboard_image.array,
        )


def test_empty_input_raises(checkerboard_image: ImageObj) -> None:
    """An empty image collection is rejected."""
    with pytest.raises(ValueError, match="at least one image"):
        SSIM().similarity_score([], checkerboard_image.array)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"window_size": 4}, "odd integer"),
        ({"window_size": 1}, "odd integer"),
        ({"sigma": 0.0}, "sigma"),
        ({"k1": 0.0}, "k1 and k2"),
        ({"k2": -0.03}, "k1 and k2"),
        ({"batch_size": 0}, "batch_size"),
        ({"batch_size": -2}, "batch_size"),
    ],
)
def test_invalid_parameters_raise(kwargs: dict[str, float], match: str) -> None:
    """Out-of-range constructor parameters are rejected."""
    with pytest.raises(ValueError, match=match):
        SSIM(**kwargs)  # type: ignore[arg-type]


def test_batch_size_setter_validates() -> None:
    """Reassigning ``batch_size`` goes through the same validation."""
    metric = SSIM()
    metric.batch_size = 4
    assert metric.batch_size == 4
    with pytest.raises(ValueError, match="batch_size"):
        metric.batch_size = 0


def test_repr() -> None:
    """``repr`` reports the metric configuration."""
    assert repr(SSIM()) == (
        "SSIM(window_size=11, sigma=1.5, k1=0.01, k2=0.03, batch_size=2)"
    )
