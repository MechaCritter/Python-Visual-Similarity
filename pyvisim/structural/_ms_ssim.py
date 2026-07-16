"""
Multi-Scale Structural Similarity (MS-SSIM) metric.

References
==========
Wang, Z., Simoncelli, E. P., & Bovik, A. C. (2003). Multiscale structural
similarity for image quality assessment. The 37th Asilomar Conference on
Signals, Systems & Computers, Vol. 2, 1398-1402.
https://doi.org/10.1109/ACSSC.2003.1292216
"""

from collections.abc import Sequence

import numpy as np

from ..base import CANONICAL_DATA_RANGE, DenseMetricBase
from ..typing import Float64NumpyArray
from ..utils.cython_utils import get_kernel_threads
from ._filters import _as_planes, _downsample_planes, gaussian_kernel
from ._kernel._ssim_kernels import ssim_plane_sums
from ._ssim import _validate_stabilizers, _validate_window

__all__ = ["MSSSIM"]

#: Per-scale exponents from Wang et al. (2003), obtained through the
#: cross-scale calibration experiment in the original paper.
WANG_WEIGHTS: tuple[float, ...] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def _validate_weights(weights: Sequence[float]) -> tuple[float, ...]:
    """
    Validate the per-scale MS-SSIM exponents.

    :param weights: One exponent per scale, coarsest scale last.
    :return: The weights as a tuple of floats.
    :raises ValueError: If no weight is given or any weight is not a finite
        positive number.
    """
    validated = tuple(float(weight) for weight in weights)
    if not validated:
        raise ValueError("weights must contain at least one value.")
    if any(not np.isfinite(weight) or weight <= 0 for weight in validated):
        raise ValueError(
            f"All MS-SSIM scale weights must be finite and > 0, got {validated}."
        )
    return validated


class MSSSIM(DenseMetricBase):
    """
    Multi-Scale Structural Similarity index (Wang et al., 2003).

    MS-SSIM evaluates the SSIM contrast-structure component on a dyadic image
    pyramid: after each scale the images are low-pass filtered and downsampled
    by two (2x2 average pooling), and the luminance component enters only at
    the coarsest scale. With per-scale map means ``cs_j`` and the full SSIM
    mean ``ssim_M`` at the coarsest scale ``M``, the score is the weighted
    geometric mean::

        MS-SSIM(x, y) = ssim_M ** w_M * prod_{j=1..M-1} cs_j ** w_j

    Negative per-scale means are clamped to zero before exponentiation so the
    fractional powers stay real (the usual "relu" normalization, as in
    torchmetrics); natural image pairs are unaffected. Because every input is
    normalized to the canonical ``[0, 255]`` range, the dynamic range ``L``
    is fixed at 255. The window statistics are computed by a compiled
    multithreaded kernel in ``float32`` precision (scores match a ``float64``
    computation to ~1e-5).

    Higher values mean more similar: identical images score 1 and unrelated
    images score near 0.

    :param weights: One positive exponent per scale, coarsest scale last. The
        number of weights sets the number of scales. Defaults to the five
        exponents calibrated in the original paper.
    :param window_size: Side length of the Gaussian window, an odd integer
        >= 3 (default 11).
    :param sigma: Standard deviation of the Gaussian window (default 1.5).
    :param k1: Luminance stabilization constant (default 0.01).
    :param k2: Contrast stabilization constant (default 0.03).
    :param batch_size: Maximum number of image pairs scored per vectorized
        chunk; ``-1`` (default) scores the whole input as one batch.
    :param num_workers: Number of threads to use for the computation. If ``None``,
    use the default `PYVISIM_NUM_THREADS` instead (can be overridden by
    setting os.environ["PYVISIM_NUM_THREADS"]).
    :raises ValueError: If any parameter is outside its valid range.
    """

    def __init__(
        self,
        weights: Sequence[float] = WANG_WEIGHTS,
        window_size: int = 11,
        sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        batch_size: int = 2,
        num_workers: int | None = None,
    ):
        super().__init__(batch_size=batch_size)
        _validate_window(window_size, sigma)
        _validate_stabilizers(k1, k2)
        self._weights = _validate_weights(weights)
        self._window_size = window_size
        self._sigma = sigma
        self._k1 = k1
        self._k2 = k2
        self._kernel = gaussian_kernel(window_size, sigma)
        # float32 copy for the compiled kernel, cast once instead of per call.
        self._kernel32 = self._kernel.astype(np.float32)
        self._num_workers = (
            num_workers if num_workers is not None else get_kernel_threads()
        )

    @property
    def weights(self) -> tuple[float, ...]:
        """Per-scale exponents, coarsest scale last (read-only)."""
        return self._weights

    @property
    def n_scales(self) -> int:
        """Number of pyramid scales, one per weight (read-only)."""
        return len(self._weights)

    @property
    def window_size(self) -> int:
        """Side length of the Gaussian window, in pixels (read-only)."""
        return self._window_size

    @property
    def sigma(self) -> float:
        """Standard deviation of the Gaussian window (read-only)."""
        return self._sigma

    @property
    def k1(self) -> float:
        """Luminance stabilization constant (read-only)."""
        return self._k1

    @property
    def k2(self) -> float:
        """Contrast stabilization constant (read-only)."""
        return self._k2

    def _validate_image_shape(self, height: int, width: int) -> None:
        """
        Reject images too small for the configured pyramid.

        The Gaussian window must still fit after ``n_scales - 1`` halvings,
        so each side must measure at least ``window_size * 2**(n_scales - 1)``
        pixels.

        :param height: Height of the images, in pixels.
        :param width: Width of the images, in pixels.
        :raises ValueError: If the coarsest scale would be smaller than the
            window.
        """
        min_size = self._window_size * 2 ** (self.n_scales - 1)
        if min(height, width) < min_size:
            raise ValueError(
                f"Images of size {height}x{width} are too small for "
                f"{self.n_scales} MS-SSIM scales with a "
                f"{self._window_size}x{self._window_size} window; each side "
                f"must measure at least {min_size} pixels. Use larger images, "
                "fewer weights or a smaller window_size."
            )

    def _score_pairs(
        self, images1: Float64NumpyArray, images2: Float64NumpyArray
    ) -> Float64NumpyArray:
        """
        Compute the MS-SSIM score of each aligned image pair.

        The pyramid is scored by the compiled fused window kernel in
        ``float32`` precision, one channel plane at a time, and every scale
        mean is pooled over all channels.

        :param images1: ``(B, H, W, C)`` ``float64`` batch, one image per pair.
        :param images2: ``(B, H, W, C)`` ``float64`` batch, aligned with
            ``images1``.
        :return: A ``(B,)`` array of MS-SSIM scores.
        """
        c1 = (self._k1 * CANONICAL_DATA_RANGE) ** 2
        c2 = (self._k2 * CANONICAL_DATA_RANGE) ** 2
        n_pairs, n_channels = images1.shape[0], images1.shape[3]
        scale_means = np.empty((self.n_scales, n_pairs), dtype=np.float64)
        planes1, planes2 = _as_planes(images1), _as_planes(images2)
        for scale in range(self.n_scales):
            sum_l_cs, sum_cs = ssim_plane_sums(
                planes1, planes2, self._kernel32, c1, c2, self._num_workers
            )
            out_h = planes1.shape[1] - self._window_size + 1
            out_w = planes1.shape[2] - self._window_size + 1
            map_size = n_channels * out_h * out_w
            if scale < self.n_scales - 1:
                pooled_cs = sum_cs.reshape(n_pairs, n_channels).sum(axis=1)
                scale_means[scale] = pooled_cs / map_size
                planes1 = _downsample_planes(planes1)
                planes2 = _downsample_planes(planes2)
            else:
                pooled_l_cs = sum_l_cs.reshape(n_pairs, n_channels).sum(axis=1)
                scale_means[scale] = pooled_l_cs / map_size
        exponents = np.asarray(self._weights, dtype=np.float64)[:, np.newaxis]
        clamped = np.maximum(scale_means, 0.0)
        scores: Float64NumpyArray = np.prod(clamped**exponents, axis=0)
        return scores

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(weights={self._weights}, "
            f"window_size={self._window_size}, sigma={self._sigma}, "
            f"k1={self._k1}, k2={self._k2}, batch_size={self.batch_size})"
        )
