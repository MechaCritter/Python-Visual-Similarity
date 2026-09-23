"""Peak signal-to-noise ratio (PSNR) similarity metric."""

from __future__ import annotations

import numpy as np

from ...base import CANONICAL_DATA_RANGE, DenseMetricBase
from ...typing import Float64NumpyArray, UInt8NumpyArray
from ...utils.cython_utils import get_kernel_threads
from ._kernel._ssd_kernel import ssd_matrix

__all__ = ["PSNR"]


def _mean_squared_error(
    batch1: UInt8NumpyArray, batch2: UInt8NumpyArray
) -> Float64NumpyArray:
    """
    Compute the pairwise mean squared error matrix between two image batches.

    :return: The ``(N, M)`` matrix of mean squared pixel errors.
    """
    flat1 = np.ascontiguousarray(batch1.reshape(len(batch1), -1))
    flat2 = np.ascontiguousarray(batch2.reshape(len(batch2), -1))
    totals = ssd_matrix(flat1, flat2, get_kernel_threads())
    error: Float64NumpyArray = totals / flat1.shape[1]
    return error


def _peak_signal_noise_ratio(
    batch1: UInt8NumpyArray, batch2: UInt8NumpyArray
) -> Float64NumpyArray:
    """
    Compute the pairwise PSNR matrix in decibels between two image batches.
    """
    error = _mean_squared_error(batch1, batch2)
    with np.errstate(divide="ignore"):
        return np.asarray(
            10.0 * np.log10(CANONICAL_DATA_RANGE**2 / error), dtype=np.float64
        )


class PSNR(DenseMetricBase):
    """
    Peak signal-to-noise ratio between two batches of images.

    The metric is computed pairwise: for ``N`` images in the first batch and
    ``M`` images in the second batch, :meth:`similarity_score` returns an
    ``(N, M)`` matrix of PSNR values in decibels, where identical pairs yield
    ``inf``. Every compared pair must share the same ``(H, W[, C])`` shape.

    The squared differences are summed by a compiled OpenMP kernel. Set the
    ``PYVISIM_NUM_THREADS`` environment variable to override the team size.

    For more information, see the documentation:
    ``file:///home/critter_cool_laptop/workspace/Python-Visual-Similarity-parallel/docs/_build/html/pixelwise/psnr/psnr.html``.

    :param batch_size: Maximum number of images processed in a single batch.
        Set to ``-1`` to process all images as a single batch.
    :raises ValueError: If ``batch_size`` is neither ``-1`` nor a positive
        integer.

    Example:

    >>> import numpy as np
    >>> from pyvisim.dense.pixelwise import PSNR
    >>> image = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    >>> PSNR().similarity_score(image, image)
    array([[inf]])
    """

    def _score_batches(
        self, batch1: UInt8NumpyArray, batch2: UInt8NumpyArray
    ) -> Float64NumpyArray:
        """
        Score both batches block by block, one kernel call per pair of blocks.

        Each batch is cut into blocks of at most :attr:`batch_size` images.

        :param batch1: ``(N, H, W, C)`` ``uint8`` batch.
        :param batch2: ``(M, H, W, C)`` ``uint8`` batch of the same image shape.
        :return: An ``(N, M)`` matrix of PSNR values in decibels.
        """
        scores: Float64NumpyArray = np.empty(
            (len(batch1), len(batch2)), dtype=np.float64
        )
        whole_input = self._batch_size == -1
        step1 = len(batch1) if whole_input else self._batch_size
        step2 = len(batch2) if whole_input else self._batch_size
        for start1 in range(0, len(batch1), step1):
            chunk1 = batch1[start1 : start1 + step1]
            for start2 in range(0, len(batch2), step2):
                chunk2 = batch2[start2 : start2 + step2]
                scores[start1 : start1 + step1, start2 : start2 + step2] = (
                    _peak_signal_noise_ratio(chunk1, chunk2)
                )
        return scores
