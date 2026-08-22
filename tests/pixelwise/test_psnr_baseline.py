"""Slow tests of :class:`pyvisim.pixelwise.PSNR` against its reference.

The baseline is the one the documented benchmark uses,
``skimage.metrics.peak_signal_noise_ratio(data_range=255)``, and so is the
setup: Oxford Flower images distorted with Gaussian noise for the accuracy
checks, square-resized galleries and a median over repeated calls for the
runtime one (see ``docs/pixelwise/benchmarks/generate_benchmark.py``).

The module is marked ``slow`` because it downloads the flower dataset.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np
import pytest
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio

from pyvisim.datasets import OxfordFlowerDataset
from pyvisim.pixelwise import PSNR
from pyvisim.typing import UInt8NumpyArray

pytestmark = pytest.mark.slow

#: Split, seed and distortion of the documented benchmark.
_DATASET_SPLIT = "train"
_SEED = 0
_NOISE_STD = 15.0

#: OpenMP team size the compiled kernel runs with, as in the benchmark.
_NUM_WORKERS = 4

#: Number of dataset images the accuracy tests score.
_ACCURACY_IMAGES = 8

#: Side length and gallery size of the runtime comparison.
_RUNTIME_SIZE = 256
_RUNTIME_IMAGES = 4

#: Timed calls per implementation, after one warm-up call.
_REPEATS = 7


def _add_noise(image: UInt8NumpyArray, rng: np.random.Generator) -> UInt8NumpyArray:
    """Add clipped Gaussian noise of ``_NOISE_STD`` to a uint8 image."""
    noisy = image.astype(np.float64) + rng.normal(0.0, _NOISE_STD, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _resize(image: UInt8NumpyArray, size: int) -> UInt8NumpyArray:
    """Resize an RGB uint8 image to a square side length."""
    resized = Image.fromarray(image).resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(resized)


def _baseline_psnr(image_a: UInt8NumpyArray, image_b: UInt8NumpyArray) -> float:
    """Score one pair with the scikit-image baseline, in decibels."""
    return float(peak_signal_noise_ratio(image_a, image_b, data_range=255))


def _median_ms(call: Callable[[], object]) -> float:
    """Median duration of ``_REPEATS`` calls in ms, after one warm-up call."""
    call()
    durations = []
    for _ in range(_REPEATS):
        start = time.perf_counter()
        call()
        durations.append(1000.0 * (time.perf_counter() - start))
    return float(np.median(durations))


@pytest.fixture(scope="module")
def flower_images() -> list[UInt8NumpyArray]:
    """Distinct RGB flower images, sampled like the benchmark does.

    :return: ``_ACCURACY_IMAGES`` images at their native resolution.
    """
    dataset = OxfordFlowerDataset(purpose=_DATASET_SPLIT)
    rng = np.random.default_rng(_SEED)
    indices = rng.choice(len(dataset), size=_ACCURACY_IMAGES, replace=False)
    return [dataset[int(index)][0] for index in indices]


@pytest.fixture(scope="module")
def gallery(flower_images: list[UInt8NumpyArray]) -> list[UInt8NumpyArray]:
    """A gallery of equally sized images, as the runtime scenarios use.

    :param flower_images: The sampled flower images.
    :return: ``_RUNTIME_IMAGES`` images resized to a common square size.
    """
    return [_resize(image, _RUNTIME_SIZE) for image in flower_images[:_RUNTIME_IMAGES]]


def test_scores_match_the_baseline_on_noisy_flowers(
    flower_images: list[UInt8NumpyArray],
) -> None:
    """Every image scored against a noisy copy matches scikit-image."""
    rng = np.random.default_rng(_SEED)
    metric = PSNR()
    for image in flower_images:
        distorted = _add_noise(image, rng)
        ours = metric.similarity_score(image, distorted)[0, 0]
        assert ours == pytest.approx(_baseline_psnr(image, distorted), abs=1e-9)


def test_score_matrix_matches_the_baseline_pairwise(
    gallery: Sequence[UInt8NumpyArray],
) -> None:
    """Every cell of the (N, M) matrix matches the pairwise baseline call."""
    queries = list(gallery[:2])
    scores = PSNR().similarity_score(gallery, queries)
    assert scores.shape == (len(gallery), len(queries))
    for row, image in enumerate(gallery):
        for column, query in enumerate(queries):
            if row == column:
                assert scores[row, column] == np.inf
                continue
            assert scores[row, column] == pytest.approx(
                _baseline_psnr(image, query), abs=1e-9
            )


def test_is_faster_than_the_baseline(
    gallery: Sequence[UInt8NumpyArray], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scoring a full gallery beats the baseline's pairwise Python loop.

    The compiled kernel scores the whole ``(N, N)`` grid in one call, which
    the documented benchmark measures at roughly an order of magnitude faster;
    the assertion only requires it not to be slower.
    """
    monkeypatch.setenv("PYVISIM_NUM_THREADS", str(_NUM_WORKERS))
    rng = np.random.default_rng(_SEED)
    # Distorted copies rather than the gallery itself: an identical pair makes
    # the baseline divide by zero and warn on its way to 'inf'.
    queries = [_add_noise(image, rng) for image in gallery]
    metric = PSNR()
    ours = _median_ms(lambda: metric.similarity_score(gallery, queries))
    baseline = _median_ms(
        lambda: [_baseline_psnr(a, b) for a in gallery for b in queries]
    )
    assert ours < baseline
