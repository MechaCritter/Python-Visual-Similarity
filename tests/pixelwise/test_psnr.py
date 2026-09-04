"""Tests for :class:`pyvisim.pixelwise.PSNR`."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pyvisim.pixelwise import PSNR
from pyvisim.pixelwise import psnr as psnr_module

if TYPE_CHECKING:
    from tests.conftest import ImageObj

#: Peak value of the canonical ``uint8`` range, the ``MAX`` of the PSNR formula.
PEAK_SIGNAL = 255.0

BATCH_SIZES = [2, 3, 4, 6, 16]


@pytest.fixture
def gallery(
    checkerboard_image: ImageObj,
    noisy_checkerboard_image: ImageObj,
    stripes_image: ImageObj,
    black_image: ImageObj,
    white_image: ImageObj,
) -> list[np.ndarray]:
    """Five distinct images of identical shape, to be batched in every way."""
    return [
        checkerboard_image.array,
        noisy_checkerboard_image.array,
        stripes_image.array,
        black_image.array,
        white_image.array,
    ]


# Scores


def test_identical_images_score_infinity(
    identical_image_pair: tuple[np.ndarray, np.ndarray],
) -> None:
    """Two pixel-identical images have zero error and therefore infinite PSNR."""
    image_a, image_b = identical_image_pair
    scores = PSNR().similarity_score(image_a, image_b)
    assert scores.shape == (1, 1)
    assert scores[0, 0] == math.inf


def test_symmetry(
    checkerboard_image: ImageObj, noisy_checkerboard_image: ImageObj
) -> None:
    """PSNR is symmetric in its two arguments."""
    metric = PSNR()
    forward = metric.similarity_score(
        checkerboard_image.array, noisy_checkerboard_image.array
    )
    backward = metric.similarity_score(
        noisy_checkerboard_image.array, checkerboard_image.array
    )
    assert forward[0, 0] == backward[0, 0]


def test_more_noise_lowers_score(
    checkerboard_image: ImageObj,
    noisy_image_pair: tuple[ImageObj, ImageObj],
    very_noisy_image_pair: tuple[ImageObj, ImageObj],
) -> None:
    """Scores decrease monotonically with the noise level of the distortion."""
    metric = PSNR()
    clean = checkerboard_image.array
    mild = metric.similarity_score(clean, noisy_image_pair[0].array)[0, 0]
    heavy = metric.similarity_score(clean, very_noisy_image_pair[0].array)[0, 0]
    assert 0.0 < heavy < mild < math.inf


def test_black_vs_white_scores_zero_decibels(
    black_image: ImageObj, white_image: ImageObj
) -> None:
    """The largest possible error, 255 per pixel, is exactly 0 dB.

    The mean squared error equals the squared peak signal, so the ratio the
    logarithm is taken of is 1.
    """
    score = PSNR().similarity_score(black_image.array, white_image.array)[0, 0]
    assert score == pytest.approx(0.0, abs=1e-12)


def test_score_of_a_known_mean_squared_error() -> None:
    """A constant offset of 51 gives an MSE of 51^2 and 10*log10(25) dB."""
    dark = np.zeros((64, 64), dtype=np.uint8)
    lighter = np.full((64, 64), 51, dtype=np.uint8)
    score = PSNR().similarity_score(dark, lighter)[0, 0]
    expected = 10.0 * math.log10(PEAK_SIGNAL**2 / 51.0**2)
    assert score == pytest.approx(expected, abs=1e-12)
    assert score == pytest.approx(13.979400, abs=1e-6)


def test_replicated_rgb_matches_grayscale(
    checkerboard_image: ImageObj, noisy_checkerboard_image: ImageObj
) -> None:
    """An RGB image with three identical channels scores like its grayscale.

    Every channel contributes the same squared errors, so the mean over three
    times as many pixels is unchanged.
    """
    gray_score = PSNR().similarity_score(
        checkerboard_image.array, noisy_checkerboard_image.array
    )
    rgb_a = np.stack([checkerboard_image.array] * 3, axis=-1)
    rgb_b = np.stack([noisy_checkerboard_image.array] * 3, axis=-1)
    rgb_score = PSNR().similarity_score(rgb_a, rgb_b)
    assert rgb_score[0, 0] == pytest.approx(gray_score[0, 0], abs=1e-12)


# Batching


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
    scores = PSNR().similarity_score(images1, images2)
    assert scores.shape == (3, 2)
    assert scores[0, 0] == math.inf
    assert scores[2, 1] == math.inf


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_batch_size_does_not_change_scores(
    gallery: list[np.ndarray], batch_size: int
) -> None:
    """Chunked scoring returns exactly the matrix of one image per batch."""
    images1, images2 = gallery[:4], gallery[:2]
    one_by_one = PSNR(batch_size=1).similarity_score(images1, images2)
    chunked = PSNR(batch_size=batch_size).similarity_score(images1, images2)
    np.testing.assert_array_equal(chunked, one_by_one)


def test_the_whole_input_as_one_batch_does_not_change_scores(
    gallery: list[np.ndarray],
) -> None:
    """``-1`` scores every pair in one go, for the same matrix as ``1`` does."""
    images1, images2 = gallery[:4], gallery[:2]
    one_by_one = PSNR(batch_size=1).similarity_score(images1, images2)
    whole = PSNR(batch_size=-1).similarity_score(images1, images2)
    np.testing.assert_array_equal(whole, one_by_one)


@pytest.mark.parametrize(("n_images", "batch_size"), [(5, 2), (3, 4)])
def test_the_last_batch_holds_the_remaining_images(
    gallery: list[np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
    n_images: int,
    batch_size: int,
) -> None:
    """The last chunk holds ``N % batch_size`` images, however small ``N`` is."""
    chunk_sizes: list[int] = []
    score_chunk = psnr_module._peak_signal_noise_ratio

    def record(batch1: np.ndarray, batch2: np.ndarray) -> np.ndarray:
        chunk_sizes.append(len(batch1))
        return score_chunk(batch1, batch2)

    monkeypatch.setattr(psnr_module, "_peak_signal_noise_ratio", record)
    PSNR(batch_size=batch_size).similarity_score(gallery[:n_images], gallery[:1])
    assert chunk_sizes[-1] == n_images % batch_size
    assert sum(chunk_sizes) == n_images


def test_score_matrix_shape_for_a_single_image(gallery: list[np.ndarray]) -> None:
    """A single image keeps its own row of the matrix."""
    assert PSNR().similarity_score(gallery[:1], gallery[:2]).shape == (1, 2)


def test_batched_array_input(
    checkerboard_image: ImageObj, noisy_checkerboard_image: ImageObj
) -> None:
    """Stacked arrays with a batch axis are split via the ``dims`` labels."""
    stacked1 = np.stack([checkerboard_image.array, noisy_checkerboard_image.array])
    stacked2 = np.stack([checkerboard_image.array])
    scores = PSNR().similarity_score(stacked1, stacked2, dims="BHW")
    assert scores.shape == (2, 1)
    assert scores[0, 0] == math.inf


def test_value_range_rescaling(
    checkerboard_image: ImageObj, noisy_checkerboard_image: ImageObj
) -> None:
    """Float images in [0, 1] score like their uint8 counterparts."""
    metric = PSNR()
    from_uint8 = metric.similarity_score(
        checkerboard_image.array, noisy_checkerboard_image.array
    )
    from_float = metric.similarity_score(
        checkerboard_image.array.astype(np.float64) / 255.0,
        noisy_checkerboard_image.array.astype(np.float64) / 255.0,
        value_range=(0.0, 1.0),
    )
    assert from_float[0, 0] == pytest.approx(from_uint8[0, 0], abs=1e-12)


# Input validation


def test_mismatched_shapes_between_batches_raise(
    checkerboard_image: ImageObj, small_image: ImageObj
) -> None:
    """Comparing images of different shapes raises ``ValueError``."""
    with pytest.raises(ValueError, match="identical shape"):
        PSNR().similarity_score(checkerboard_image.array, small_image.array)


def test_mixed_shapes_within_batch_raise(
    checkerboard_image: ImageObj, small_image: ImageObj
) -> None:
    """A batch of differently shaped images raises ``ValueError``."""
    batch = [checkerboard_image.array, small_image.array]
    with pytest.raises(ValueError, match="identical shape"):
        PSNR().similarity_score(batch, checkerboard_image.array)


def test_grayscale_against_rgb_raises(checkerboard_image: ImageObj) -> None:
    """A grayscale image has no counterpart in an RGB one."""
    rgb = np.stack([checkerboard_image.array] * 3, axis=-1)
    with pytest.raises(ValueError, match="identical shape"):
        PSNR().similarity_score(checkerboard_image.array, rgb)


def test_empty_input_raises(checkerboard_image: ImageObj) -> None:
    """An empty image collection is rejected."""
    with pytest.raises(ValueError, match="at least one image"):
        PSNR().similarity_score([], checkerboard_image.array)


def test_set_batch_size_changes_chunking(
    checkerboard_image: ImageObj, noisy_checkerboard_image: ImageObj
) -> None:
    """``set_batch_size`` rechunks the metric without changing its scores."""
    images = [checkerboard_image.array, noisy_checkerboard_image.array]
    metric = PSNR()
    whole = metric.similarity_score(images, images)
    metric.set_batch_size(1)
    assert metric.batch_size == 1
    np.testing.assert_array_equal(whole, metric.similarity_score(images, images))


def test_repr() -> None:
    """The repr states the configured batch size."""
    assert repr(PSNR()) == "PSNR(batch_size=16)"
    assert repr(PSNR(batch_size=4)) == "PSNR(batch_size=4)"
