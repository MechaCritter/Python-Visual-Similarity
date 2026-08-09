"""Filtering and layout helpers for the structural similarity metrics."""

import numpy as np

from ..typing import Float32NumpyArray, Float64NumpyArray


def _as_planes(images: Float64NumpyArray) -> Float32NumpyArray:
    """
    Flatten a ``(B, H, W, C)`` batch into ``(B * C, H, W)`` ``float32`` planes.

    The compiled SSIM kernel operates on single-channel planes with
    unit-stride rows, so each channel becomes its own plane; a pooled-channel
    map mean is recovered by averaging the per-plane sums of one image.

    :param images: ``(B, H, W, C)`` ``float64`` batch.
    :return: The ``(B * C, H, W)`` C-contiguous ``float32`` planes.
    """
    n_images, height, width, n_channels = images.shape
    channel_first = images.transpose(0, 3, 1, 2).astype(np.float32, order="C")
    planes: Float32NumpyArray = channel_first.reshape(
        n_images * n_channels, height, width
    )
    return planes


def gaussian_kernel(size: int, sigma: float) -> Float64NumpyArray:
    """
    Build a normalized one-dimensional Gaussian kernel.

    :param size: Number of kernel taps.
    :param sigma: Standard deviation of the Gaussian, in pixels.
    :return: A ``(size,)`` ``float64`` kernel summing to 1.
    """
    offsets = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    kernel = np.exp(-(offsets**2) / (2.0 * sigma**2))
    normalized: Float64NumpyArray = kernel / kernel.sum()
    return normalized


def _downsample_planes(planes: Float32NumpyArray) -> Float32NumpyArray:
    """
    Halve the spatial resolution of image planes with 2x2 average pooling.

    Odd trailing rows/columns are dropped before pooling, mirroring the
    behaviour of ``torch.nn.functional.avg_pool2d`` with kernel and stride 2.

    :param planes: ``(N, H, W)`` ``float32`` image planes.
    :return: The ``(N, H // 2, W // 2)`` pooled planes.
    """
    n_planes, height, width = planes.shape
    half_h, half_w = height // 2, width // 2
    cropped = planes[:, : half_h * 2, : half_w * 2]
    pooled = cropped.reshape(n_planes, half_h, 2, half_w, 2)
    result: Float32NumpyArray = pooled.mean(axis=(2, 4))
    return result
