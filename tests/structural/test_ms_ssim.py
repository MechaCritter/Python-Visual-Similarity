"""Tests for :class:`pyvisim.structural.MSSSIM`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pyvisim.structural import MSSSIM, SSIM

if TYPE_CHECKING:
    from tests.conftest import ImageObj

BATCH_SIZES = [2, 4, 6, 12, 16]


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


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------


def test_identical_images_score_one(
    identical_image_pair: tuple[np.ndarray, np.ndarray],
) -> None:
    """Two pixel-identical images reach the maximum score of 1."""
    image_a, image_b = identical_image_pair
    scores = MSSSIM().similarity_score(image_a, image_b)
    assert scores.shape == (1, 1)
    assert scores[0, 0] == pytest.approx(1.0, abs=1e-12)


def test_single_scale_equals_ssim(
    checkerboard_image: ImageObj, noisy_checkerboard_image: ImageObj
) -> None:
    """With one scale and weight 1, MS-SSIM reduces exactly to SSIM."""
    single_scale = MSSSIM(weights=(1.0,)).similarity_score(
        checkerboard_image.array, noisy_checkerboard_image.array
    )
    plain = SSIM().similarity_score(
        checkerboard_image.array, noisy_checkerboard_image.array
    )
    assert single_scale[0, 0] == pytest.approx(plain[0, 0], abs=1e-12)


def test_symmetry(
    checkerboard_image: ImageObj, noisy_checkerboard_image: ImageObj
) -> None:
    """MS-SSIM is symmetric in its two arguments."""
    metric = MSSSIM()
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
    metric = MSSSIM()
    clean = checkerboard_image.array
    mild = metric.similarity_score(clean, noisy_image_pair[0].array)[0, 0]
    heavy = metric.similarity_score(clean, very_noisy_image_pair[0].array)[0, 0]
    assert 0.0 < heavy < mild < 1.0


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def test_score_matrix_shape(
    checkerboard_image: ImageObj,
    noisy_checkerboard_image: ImageObj,
    stripes_image: ImageObj,
) -> None:
    """Batches of N and M images yield an (N, M) score matrix."""
    images1 = [checkerboard_image.array, noisy_checkerboard_image.array]
    images2 = [
        checkerboard_image.array,
        noisy_checkerboard_image.array,
        stripes_image.array,
    ]
    scores = MSSSIM().similarity_score(images1, images2)
    assert scores.shape == (2, 3)
    assert scores[0, 0] == pytest.approx(1.0, abs=1e-12)
    assert scores[1, 1] == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_batch_size_does_not_change_scores(
    gallery: list[np.ndarray], batch_size: int
) -> None:
    """Chunked scoring returns exactly the matrix of one pair per batch."""
    images1, images2 = gallery[:3], gallery[:2]
    one_by_one = MSSSIM(batch_size=1).similarity_score(images1, images2)
    chunked = MSSSIM(batch_size=batch_size).similarity_score(images1, images2)
    np.testing.assert_array_equal(chunked, one_by_one)


def test_the_whole_input_as_one_batch_does_not_change_scores(
    gallery: list[np.ndarray],
) -> None:
    """``-1`` scores every pair in one go, for the same matrix as ``1`` does."""
    images1, images2 = gallery[:3], gallery[:2]
    one_by_one = MSSSIM(batch_size=1).similarity_score(images1, images2)
    whole = MSSSIM(batch_size=-1).similarity_score(images1, images2)
    np.testing.assert_array_equal(whole, one_by_one)


@pytest.mark.parametrize(("n_pairs", "batch_size"), [(5, 2), (3, 4)])
def test_the_last_batch_holds_the_remaining_pairs(
    gallery: list[np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
    n_pairs: int,
    batch_size: int,
) -> None:
    """The last batch holds ``N % batch_size`` pairs, however small ``N`` is."""
    metric = MSSSIM(batch_size=batch_size)
    batch_lengths: list[int] = []
    score_pairs = metric._score_pairs

    def record(images1: np.ndarray, images2: np.ndarray) -> np.ndarray:
        batch_lengths.append(len(images1))
        return score_pairs(images1, images2)

    monkeypatch.setattr(metric, "_score_pairs", record)
    metric.similarity_score(gallery[:n_pairs], gallery[:1])
    assert batch_lengths[-1] == n_pairs % batch_size
    assert sum(batch_lengths) == n_pairs


def test_score_matrix_shape_for_a_single_image(gallery: list[np.ndarray]) -> None:
    """A single image keeps its own row of the matrix."""
    assert MSSSIM().similarity_score(gallery[:1], gallery[:2]).shape == (1, 2)


def test_batched_array_input(
    checkerboard_image: ImageObj, noisy_checkerboard_image: ImageObj
) -> None:
    """Stacked arrays with a batch axis are split via the ``dims`` labels."""
    stacked1 = np.stack([checkerboard_image.array, noisy_checkerboard_image.array])
    stacked2 = np.stack([checkerboard_image.array])
    scores = MSSSIM().similarity_score(stacked1, stacked2, dims="BHW")
    assert scores.shape == (2, 1)
    assert scores[0, 0] == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Scales and weights
# ---------------------------------------------------------------------------


def test_default_weights_expose_five_scales() -> None:
    """The default configuration uses the five weights from the paper."""
    metric = MSSSIM()
    assert metric.n_scales == 5
    assert metric.weights == (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def test_fewer_scales_allow_smaller_images(non_square_image: ImageObj) -> None:
    """A 128x256 image is valid once the pyramid is shallow enough."""
    metric = MSSSIM(weights=(0.5, 0.3, 0.2))
    scores = metric.similarity_score(non_square_image.array, non_square_image.array)
    assert scores[0, 0] == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_too_small_image_raises(non_square_image: ImageObj) -> None:
    """A 128x256 image cannot support the default five scales."""
    with pytest.raises(ValueError, match="at least 176 pixels"):
        MSSSIM().similarity_score(non_square_image.array, non_square_image.array)


def test_mismatched_shapes_between_batches_raise(
    checkerboard_image: ImageObj, large_image: ImageObj
) -> None:
    """Comparing images of different sizes is rejected."""
    with pytest.raises(ValueError, match="same shape"):
        MSSSIM().similarity_score(checkerboard_image.array, large_image.array)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"weights": ()}, "at least one value"),
        ({"weights": (0.5, -0.5)}, "finite and > 0"),
        ({"weights": (0.5, 0.0)}, "finite and > 0"),
        ({"weights": (0.5, float("nan"))}, "finite and > 0"),
        ({"weights": (0.5, float("inf"))}, "finite and > 0"),
        ({"window_size": 8}, "odd integer"),
        ({"sigma": -1.0}, "sigma"),
        ({"k1": -0.01}, "k1 and k2"),
    ],
)
def test_invalid_parameters_raise(kwargs: dict[str, object], match: str) -> None:
    """Out-of-range constructor parameters are rejected."""
    with pytest.raises(ValueError, match=match):
        MSSSIM(**kwargs)  # type: ignore[arg-type]


def test_empty_input_raises(checkerboard_image: ImageObj) -> None:
    """An empty image collection is rejected."""
    with pytest.raises(ValueError, match="at least one image"):
        MSSSIM().similarity_score([], checkerboard_image.array)


def test_repr() -> None:
    """``repr`` reports the metric configuration."""
    metric = MSSSIM(weights=(0.5, 0.5))
    assert repr(metric) == (
        "MSSSIM(weights=(0.5, 0.5), window_size=11, sigma=1.5, "
        "k1=0.01, k2=0.03, batch_size=16)"
    )
