"""Tests for the compiled :mod:`pyvisim.structural._kernel._ssim_kernels` module."""

from __future__ import annotations

import numpy as np
import pytest

from pyvisim.structural._filters import gaussian_kernel
from pyvisim.structural._kernel._ssim_kernels import ssim_plane_sums

C1 = (0.01 * 255.0) ** 2
C2 = (0.03 * 255.0) ** 2


def _random_plane_pair(
    n_planes: int = 3, height: int = 48, width: int = 56
) -> tuple[np.ndarray, np.ndarray]:
    """Create a reproducible pair of float32 plane batches in ``[0, 255]``."""
    rng = np.random.default_rng(1234)
    planes1 = rng.uniform(0.0, 255.0, (n_planes, height, width))
    planes2 = np.clip(planes1 + rng.normal(0.0, 20.0, planes1.shape), 0.0, 255.0)
    return planes1.astype(np.float32), planes2.astype(np.float32)


def _convolve_valid(images: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Reference "valid"-mode convolution along one axis (float64)."""
    taps = kernel.shape[0]
    n_out = images.shape[axis] - taps + 1
    window: list[slice] = [slice(None)] * images.ndim
    window[axis] = slice(0, n_out)
    result = kernel[0] * images[tuple(window)]
    for offset in range(1, taps):
        window[axis] = slice(offset, offset + n_out)
        result += kernel[offset] * images[tuple(window)]
    return result


def _reference_sums(
    planes1: np.ndarray, planes2: np.ndarray, kernel: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the kernel's output with plain float64 NumPy operations."""
    x = planes1.astype(np.float64)
    y = planes2.astype(np.float64)

    def window(images: np.ndarray) -> np.ndarray:
        return _convolve_valid(_convolve_valid(images, kernel, 1), kernel, 2)

    mu1, mu2 = window(x), window(y)
    var1 = window(x * x) - mu1 * mu1
    var2 = window(y * y) - mu2 * mu2
    covar = window(x * y) - mu1 * mu2
    luminance = (2.0 * mu1 * mu2 + C1) / (mu1 * mu1 + mu2 * mu2 + C1)
    contrast_structure = (2.0 * covar + C2) / (var1 + var2 + C2)
    return (
        (luminance * contrast_structure).sum(axis=(1, 2)),
        contrast_structure.sum(axis=(1, 2)),
    )


# ---------------------------------------------------------------------------
# Numerical behaviour
# ---------------------------------------------------------------------------


def test_matches_float64_reference() -> None:
    """The float32 kernel reproduces the float64 reference map sums."""
    planes1, planes2 = _random_plane_pair()
    kernel = gaussian_kernel(11, 1.5)
    sum_l_cs, sum_cs = ssim_plane_sums(
        planes1, planes2, kernel.astype(np.float32), C1, C2
    )
    ref_l_cs, ref_cs = _reference_sums(planes1, planes2, kernel)
    np.testing.assert_allclose(sum_l_cs, ref_l_cs, rtol=1e-4)
    np.testing.assert_allclose(sum_cs, ref_cs, rtol=1e-4)


def test_identical_planes_sum_to_map_size_exactly() -> None:
    """Identical inputs give luminance == cs == 1 at every output pixel."""
    planes, _ = _random_plane_pair(height=40, width=52)
    kernel = gaussian_kernel(11, 1.5).astype(np.float32)
    sum_l_cs, sum_cs = ssim_plane_sums(planes, planes, kernel, C1, C2)
    map_size = float((40 - 11 + 1) * (52 - 11 + 1))
    assert (sum_l_cs == map_size).all()
    assert (sum_cs == map_size).all()


def test_bitwise_symmetry() -> None:
    """Swapping the two plane batches changes no output bit."""
    planes1, planes2 = _random_plane_pair()
    kernel = gaussian_kernel(11, 1.5).astype(np.float32)
    forward = ssim_plane_sums(planes1, planes2, kernel, C1, C2)
    backward = ssim_plane_sums(planes2, planes1, kernel, C1, C2)
    np.testing.assert_array_equal(forward[0], backward[0])
    np.testing.assert_array_equal(forward[1], backward[1])


def test_repeated_calls_are_bitwise_deterministic() -> None:
    """Two identical calls agree bit for bit despite multithreading."""
    planes1, planes2 = _random_plane_pair(n_planes=5)
    kernel = gaussian_kernel(11, 1.5).astype(np.float32)
    first = ssim_plane_sums(planes1, planes2, kernel, C1, C2)
    second = ssim_plane_sums(planes1, planes2, kernel, C1, C2)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_single_plane_slices_match_batch() -> None:
    """Each plane is computed independently of its batch neighbours."""
    planes1, planes2 = _random_plane_pair(n_planes=4)
    kernel = gaussian_kernel(11, 1.5).astype(np.float32)
    batch = ssim_plane_sums(planes1, planes2, kernel, C1, C2)
    for index in range(planes1.shape[0]):
        single = ssim_plane_sums(
            np.ascontiguousarray(planes1[index : index + 1]),
            np.ascontiguousarray(planes2[index : index + 1]),
            kernel,
            C1,
            C2,
        )
        assert single[0][0] == batch[0][index]
        assert single[1][0] == batch[1][index]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_shape_mismatch_raises() -> None:
    """Misaligned plane batches are rejected."""
    planes1, planes2 = _random_plane_pair()
    kernel = gaussian_kernel(11, 1.5).astype(np.float32)
    with pytest.raises(ValueError, match="same shape"):
        ssim_plane_sums(planes1, np.ascontiguousarray(planes2[:, :-1]), kernel, C1, C2)


def test_plane_smaller_than_window_raises() -> None:
    """Planes the window does not fit into are rejected."""
    planes1, planes2 = _random_plane_pair(height=8, width=8)
    kernel = gaussian_kernel(11, 1.5).astype(np.float32)
    with pytest.raises(ValueError, match="smaller than"):
        ssim_plane_sums(planes1, planes2, kernel, C1, C2)
