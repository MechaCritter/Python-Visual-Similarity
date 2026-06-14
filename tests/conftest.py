"""Shared pytest fixtures for pyvisim's encoder test suite."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

#: Directory where generated test images are dumped for manual inspection.
DEBUG_DIR = Path(__file__).parent / "debug"

#: Fixed side length used for most cases
MAIN_SIZE = 256

#: Gaussian noise configuration for each noise level. ``seed_a``/``seed_b``
#: are independent seeds used to generate the two images of a pair, so the
#: pair shares a noise *level* but not the actual noise pattern.
NOISE_LEVELS = {
    "noisy": {"std": 15.0, "seed_a": 101, "seed_b": 102},
    "very_noisy": {"std": 40.0, "seed_a": 201, "seed_b": 202},
    "extremely_noisy": {"std": 90.0, "seed_a": 301, "seed_b": 302},
}


@dataclass
class ImageObj:
    """Return object for all tests, holds the path and array of the image."""

    array: np.ndarray
    path: Path


def pytest_configure(config: pytest.Config) -> None:
    """Reset the debug image directory before the test session starts.

    Removes any leftover images from a previous run and re-creates an
    empty ``tests/debug/`` directory, so it always reflects only the
    most recent run.

    :param config: the pytest configuration object (unused).
    """
    if DEBUG_DIR.exists():
        shutil.rmtree(DEBUG_DIR)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def _save_debug_image(image: np.ndarray, name: str) -> None:
    """Write a generated test image to ``tests/debug/`` as a PNG file.

    :param image: image array to save, either ``(H, W)`` grayscale or
        ``(H, W, 3)`` RGB, with dtype ``uint8``.
    :param name: file name (without extension) to save the image under.
    """
    Image.fromarray(image).save(DEBUG_DIR / f"{name}.png")


def _make_solid_image(size: int, value: int) -> np.ndarray:
    """Generate a uniform, single-color grayscale image.

    :param size: width and height of the square image, in pixels.
    :param value: pixel intensity to fill the image with (0-255).
    :returns: a ``(size, size)`` ``uint8`` array filled with ``value``.
    """
    return np.full((size, size), value, dtype=np.uint8)


def _make_checkerboard(size: int, square: int = 16) -> np.ndarray:
    """Generate a black-and-white checkerboard pattern.

    :param size: width and height of the square image, in pixels.
    :param square: size of each checkerboard square, in pixels.
    :returns: a ``(size, size)`` ``uint8`` array with values 0 or 255.
    """
    rows, cols = np.indices((size, size)) // square
    pattern = (rows + cols) % 2
    return (pattern * 255).astype(np.uint8)


def _make_horizontal_gradient(size: int) -> np.ndarray:
    """Generate a smooth horizontal grayscale gradient.

    :param size: width and height of the square image, in pixels.
    :returns: a ``(size, size)`` ``uint8`` array, darkest on the left and
        brightest on the right.
    """
    row = np.linspace(0, 255, size, dtype=np.uint8)
    return np.tile(row, (size, 1))


def _make_stripes(size: int, stripe_width: int = 16) -> np.ndarray:
    """Generate alternating black-and-white vertical stripes.

    :param size: width and height of the square image, in pixels.
    :param stripe_width: width of each stripe, in pixels.
    :returns: a ``(size, size)`` ``uint8`` array with values 0 or 255.
    """
    _, cols = np.indices((size, size))
    pattern = (cols // stripe_width) % 2
    return (pattern * 255).astype(np.uint8)


def _add_gaussian_noise(image: np.ndarray, std: float, seed: int) -> np.ndarray:
    """Add reproducible gaussian noise to an image.

    :param image: the base ``uint8`` image to add noise to.
    :param std: standard deviation of the gaussian noise.
    :param seed: random seed, fixed for reproducibility across test runs.
    :returns: a ``uint8`` array with the same shape as ``image``, with
        noise added and values clipped to the valid ``[0, 255]`` range.
    """
    rng = np.random.default_rng(seed)
    noisy = image.astype(np.float64) + rng.normal(0.0, std, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _make_noise_pair(level: str) -> tuple[np.ndarray, np.ndarray]:
    """Build a pair of independently-noised images at a given noise level.

    Both images share the same checkerboard base pattern, so they differ
    *only* in their (independently sampled) noise. This makes them useful
    for testing that two images with similar noise levels score as more
    similar than two images with different noise levels.

    :param level: one of the keys of :data:`NOISE_LEVELS`.
    :returns: a tuple ``(image_a, image_b)`` of ``(256, 256)`` ``uint8``
        arrays.
    """
    config = NOISE_LEVELS[level]
    base = _make_checkerboard(size=MAIN_SIZE)
    image_a = _add_gaussian_noise(base, std=config["std"], seed=config["seed_a"])
    image_b = _add_gaussian_noise(base, std=config["std"], seed=config["seed_b"])
    return image_a, image_b


@pytest.fixture
def tiny_image() -> ImageObj:
    """An 8x8 checkerboard image, smaller than typical descriptor patches.

    :returns: an ``ImageObj`` object with the ``(8, 8)`` ``uint8`` array.
    """
    image = _make_checkerboard(size=8, square=2)
    _save_debug_image(image, "tiny_image")
    return ImageObj(array=image, path=DEBUG_DIR / "tiny_image.png")


@pytest.fixture
def small_image() -> ImageObj:
    """A 32x32 checkerboard image.

    :returns: an ``ImageObj`` object with the ``(32, 32)`` ``uint8`` array.
    """
    image = _make_checkerboard(size=32, square=4)
    _save_debug_image(image, "small_image")
    return ImageObj(array=image, path=DEBUG_DIR / "small_image.png")


@pytest.fixture
def large_image() -> ImageObj:
    """A 512x512 checkerboard image.

    :returns: an ``ImageObj`` object with the ``(512, 512)`` ``uint8`` array.
    """
    image = _make_checkerboard(size=512, square=32)
    _save_debug_image(image, "large_image")
    return ImageObj(array=image, path=DEBUG_DIR / "large_image.png")


@pytest.fixture
def non_square_image() -> ImageObj:
    """A 128x256 (height x width) checkerboard image.

    Used to confirm that the encoder does not silently assume square
    inputs.

    :returns: an ``ImageObj`` object with the ``(128, 256)`` ``uint8`` array.
    """
    rows, cols = np.indices((128, 256)) // 16
    pattern = ((rows + cols) % 2 * 255).astype(np.uint8)
    _save_debug_image(pattern, "non_square_image")
    return ImageObj(array=pattern, path=DEBUG_DIR / "non_square_image.png")


@pytest.fixture
def grayscale_image() -> ImageObj:
    """A single-channel (grayscale) checkerboard image at the edge-case size.

    :returns: an ``ImageObj`` object with the ``(256, 256)`` ``uint8`` array.
    """
    image = _make_checkerboard(size=MAIN_SIZE)
    _save_debug_image(image, "grayscale_image")
    return ImageObj(array=image, path=DEBUG_DIR / "grayscale_image.png")


@pytest.fixture()
def black_image() -> ImageObj:
    """A completely black image (all RGB channels zero).

    :returns: an ``ImageObj`` object with the ``(256, 256)`` ``uint8`` array
        filled with zeros.
    """
    image = _make_solid_image(size=MAIN_SIZE, value=0)
    _save_debug_image(image, "black_image")
    return ImageObj(array=image, path=DEBUG_DIR / "black_image.png")


@pytest.fixture()
def white_image() -> ImageObj:
    """A completely white image (all RGB channels 255).

    :returns: an ``ImageObj`` object with the ``(256, 256)`` ``uint8`` array
        filled with 255.
    """
    image = _make_solid_image(size=MAIN_SIZE, value=255)
    _save_debug_image(image, "white_image")
    return ImageObj(array=image, path=DEBUG_DIR / "white_image.png")


@pytest.fixture
def rgb_image() -> ImageObj:
    """A 3-channel (RGB) checkerboard image at the edge-case size.

    The same pattern as :func:`grayscale_image`, replicated across all
    three color channels.

    :returns: an ``ImageObj`` object with the ``(256, 256, 3)`` ``uint8`` array.
    """
    gray = _make_checkerboard(size=MAIN_SIZE)
    rgb = np.stack([gray, gray, gray], axis=-1)
    _save_debug_image(rgb, "rgb_image")
    return ImageObj(array=rgb, path=DEBUG_DIR / "rgb_image.png")


@pytest.fixture
def solid_image() -> ImageObj:
    """A uniform mid-gray image with no texture at all.

    Most keypoint detectors will find zero descriptors in this image --
    use it to verify that the encoder handles the "no descriptors found"
    case gracefully (e.g. by raising a clear error or returning a zero
    vector), rather than crashing.

    :returns: an ``ImageObj`` object with the ``(256, 256)`` ``uint8`` array
        filled with the value 128.
    """
    image = _make_solid_image(size=MAIN_SIZE, value=128)
    _save_debug_image(image, "solid_image")
    return ImageObj(array=image, path=DEBUG_DIR / "solid_image.png")


@pytest.fixture
def checkerboard_image() -> ImageObj:
    """A black-and-white checkerboard image with strong, predictable corners.

    :returns: an ``ImageObj`` object with the ``(256, 256)`` ``uint8`` array.
    """
    image = _make_checkerboard(size=MAIN_SIZE)
    _save_debug_image(image, "checkerboard_image")
    return ImageObj(array=image, path=DEBUG_DIR / "checkerboard_image.png")


@pytest.fixture
def horizontal_gradient_image() -> ImageObj:
    """A smooth horizontal gradient with very few sharp features.

    :returns: an ``ImageObj`` object with the ``(256, 256)`` ``uint8`` array.
    """
    image = _make_horizontal_gradient(size=MAIN_SIZE)
    _save_debug_image(image, "horizontal_gradient_image")
    return ImageObj(array=image, path=DEBUG_DIR / "horizontal_gradient_image.png")


@pytest.fixture
def stripes_image() -> ImageObj:
    """A repeating vertical stripe pattern.

    :returns: an ``ImageObj`` object with the ``(256, 256)`` ``uint8`` array.
    """
    image = _make_stripes(size=MAIN_SIZE)
    _save_debug_image(image, "stripes_image")
    return ImageObj(array=image, path=DEBUG_DIR / "stripes_image.png")


# Similarity test pairs (fixed 256x256 size)


@pytest.fixture
def noisy_image_pair() -> tuple[ImageObj, ImageObj]:
    """Two checkerboard images with mild, independent gaussian noise.

    Both images share the same base pattern and noise standard deviation
    (``std=15``), so an encoder should consider them highly similar.

    :returns: a tuple of two ``ImageObj`` objects with ``(256, 256)`` ``uint8``
        arrays.
    """
    image_a, image_b = _make_noise_pair("noisy")
    _save_debug_image(image_a, "noisy_image_a")
    _save_debug_image(image_b, "noisy_image_b")
    return (
        ImageObj(array=image_a, path=DEBUG_DIR / "noisy_image_a.png"),
        ImageObj(array=image_b, path=DEBUG_DIR / "noisy_image_b.png"),
    )


@pytest.fixture
def very_noisy_image_pair() -> tuple[ImageObj, ImageObj]:
    """Two checkerboard images with moderate, independent gaussian noise.

    Both images share the same base pattern and noise standard deviation
    (``std=40``).

    :returns: a tuple of two ``ImageObj`` objects with ``(256, 256)`` ``uint8``
        arrays.
    """
    image_a, image_b = _make_noise_pair("very_noisy")
    _save_debug_image(image_a, "very_noisy_image_a")
    _save_debug_image(image_b, "very_noisy_image_b")
    return (
        ImageObj(array=image_a, path=DEBUG_DIR / "very_noisy_image_a.png"),
        ImageObj(array=image_b, path=DEBUG_DIR / "very_noisy_image_b.png"),
    )


@pytest.fixture
def extremely_noisy_image_pair() -> tuple[ImageObj, ImageObj]:
    """Two checkerboard images with heavy, independent gaussian noise.

    Both images share the same base pattern and noise standard deviation
    (``std=90``), pushing the pattern close to indistinguishable from
    random noise.

    :returns: a tuple of two ``ImageObj`` objects with ``(256, 256)`` ``uint8``
        arrays.
    """
    image_a, image_b = _make_noise_pair("extremely_noisy")
    _save_debug_image(image_a, "extremely_noisy_image_a")
    _save_debug_image(image_b, "extremely_noisy_image_b")
    return (
        ImageObj(array=image_a, path=DEBUG_DIR / "extremely_noisy_image_a.png"),
        ImageObj(array=image_b, path=DEBUG_DIR / "extremely_noisy_image_b.png"),
    )


@pytest.fixture
def identical_image_pair() -> tuple[np.ndarray, np.ndarray]:
    """Two pixel-identical checkerboard images.

    Used to verify that encoding the same image twice yields (near)
    perfect similarity.

    :returns: a tuple ``(image_a, image_b)`` of ``(256, 256)`` ``uint8``
        arrays with identical contents.
    """
    image = _make_checkerboard(size=MAIN_SIZE)
    _save_debug_image(image, "identical_image_pair")
    return image, image.copy()


@pytest.fixture
def rotated_image_pair() -> tuple[np.ndarray, np.ndarray]:
    """A stripe image and a 90-degree rotation of itself.

    Rotating vertical stripes by 90 degrees turns them into horizontal
    stripes, giving a meaningful (non-trivial) rotation test.

    :returns: a tuple ``(image, rotated_image)`` of ``(256, 256)``
        ``uint8`` arrays.
    """
    image = _make_stripes(size=MAIN_SIZE)
    rotated = np.rot90(image).copy()
    _save_debug_image(image, "rotated_image_pair_original")
    _save_debug_image(rotated, "rotated_image_pair_rotated")
    return image, rotated


@pytest.fixture
def shifted_image_pair() -> tuple[ImageObj, ImageObj]:
    """A checkerboard image and a translated (shifted) version of itself.

    The shift is smaller than one checkerboard square, so the pattern
    remains recognizable but no longer pixel-aligned.

    :returns: a tuple of two ``ImageObj`` objects with ``(256, 256)``
        ``uint8`` arrays.
    """
    image = _make_checkerboard(size=MAIN_SIZE)
    shifted = np.roll(image, shift=8, axis=1)
    _save_debug_image(image, "shifted_image_pair_original")
    _save_debug_image(shifted, "shifted_image_pair_shifted")
    return (
        ImageObj(array=image, path=DEBUG_DIR / "shifted_image_pair_original.png"),
        ImageObj(array=shifted, path=DEBUG_DIR / "shifted_image_pair_shifted.png"),
    )


@pytest.fixture
def different_image_pair() -> tuple[ImageObj, ImageObj]:
    """Two clearly different patterns: a checkerboard and a stripe image.

    Used as a baseline for "these should NOT be considered similar".

    :returns: a tuple of two ``ImageObj`` objects with ``(256, 256)``
        ``uint8`` arrays.
    """
    image_a = _make_checkerboard(size=MAIN_SIZE)
    image_b = _make_stripes(size=MAIN_SIZE)
    _save_debug_image(image_a, "different_image_pair_a")
    _save_debug_image(image_b, "different_image_pair_b")
    return (
        ImageObj(array=image_a, path=DEBUG_DIR / "different_image_pair_a.png"),
        ImageObj(array=image_b, path=DEBUG_DIR / "different_image_pair_b.png"),
    )


@pytest.fixture
def image_batch(
    checkerboard_image: ImageObj,
    horizontal_gradient_image: ImageObj,
    stripes_image: ImageObj,
    solid_image: ImageObj,
) -> list[ImageObj]:
    """A small batch of varied images for testing :class:`Pipeline` encoding.

    :param checkerboard_image: fixture providing a checkerboard image.
    :param horizontal_gradient_image: fixture providing a horizontal gradient image.
    :param stripes_image: fixture providing a stripe image.
    :param solid_image: fixture providing a featureless solid image.
    :returns: a list of four ``ImageObj`` objects with ``(256, 256)`` ``uint8`` arrays.
    """
    return [checkerboard_image, horizontal_gradient_image, stripes_image, solid_image]
