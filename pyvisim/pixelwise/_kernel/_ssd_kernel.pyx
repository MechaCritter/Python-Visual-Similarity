# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Compiled kernel for the pixel-wise PSNR metric."""

import numpy as np

from cython.parallel cimport prange

__all__ = ["ssd_matrix"]


cdef enum:
    # Pixels per work block; 32768 * 255**2 stays below the uint32 limit.
    BLOCK = 32768


def ssd_matrix(
    const unsigned char[:, ::1] flat1,
    const unsigned char[:, ::1] flat2,
    int num_threads,
):
    """
    Compute the pairwise summed-squared-difference matrix of two batches.

    :param flat1: First batch flattened to a C-contiguous ``(N, P)``
        ``uint8`` array.
    :param flat2: Second batch flattened to a C-contiguous ``(M, P)``
        ``uint8`` array.
    :param num_threads: OpenMP team size used for the block loop.
    :return: The ``(N, M)`` ``int64`` matrix of summed squared pixel
        differences.
    :raises ValueError: If the batches differ in pixel count or
        ``num_threads`` is not positive.
    """
    cdef Py_ssize_t count1 = flat1.shape[0]
    cdef Py_ssize_t count2 = flat2.shape[0]
    cdef Py_ssize_t pixels = flat1.shape[1]
    if flat2.shape[1] != pixels:
        raise ValueError(
            "Batches must share the pixel count, got "
            f"{pixels} and {flat2.shape[1]}."
        )
    if num_threads < 1:
        raise ValueError(f"num_threads must be positive, got {num_threads}.")
    if count1 == 0 or count2 == 0 or pixels == 0:
        return np.zeros((count1, count2), dtype=np.int64)

    cdef Py_ssize_t blocks = (pixels + BLOCK - 1) // BLOCK
    cdef Py_ssize_t work_items = count1 * count2 * blocks
    partials_arr = np.empty((count1 * count2, blocks), dtype=np.int64)
    cdef long long[:, ::1] partials = partials_arr

    cdef Py_ssize_t work, pair, block, row1, row2, start, stop, p
    cdef unsigned int acc
    cdef int diff
    with nogil:
        for work in prange(work_items, num_threads=num_threads, schedule="static"):
            pair = work // blocks
            block = work - pair * blocks
            row1 = pair // count2
            row2 = pair - row1 * count2
            start = block * BLOCK
            stop = start + BLOCK
            if stop > pixels:
                stop = pixels
            acc = 0
            for p in range(start, stop):
                diff = <int>flat1[row1, p] - <int>flat2[row2, p]
                acc = acc + <unsigned int>(diff * diff)
            partials[pair, block] = <long long>acc
    return partials_arr.sum(axis=1).reshape(count1, count2)
