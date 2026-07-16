# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

"""Compiled kernels for the structural similarity metrics."""

import numpy as np

from cython.parallel cimport prange, threadid


def ssim_plane_sums(
    const float[:, :, ::1] planes1,
    const float[:, :, ::1] planes2,
    const float[::1] kernel,
    float c1,
    float c2,
    int num_threads
):
    """
    Reduce the SSIM component maps of aligned image planes to per-plane sums.

    For every plane pair the five local moments are estimated under the
    separable window ``kernel x kernel`` in "valid" mode, the luminance and
    contrast-structure values are formed per output pixel, and their products
    and the contrast-structure values are summed over the whole map. Dividing
    the sums by ``(H - k + 1) * (W - k + 1)`` yields the map means used by
    SSIM (``luminance * contrast_structure``) and MS-SSIM
    (``contrast_structure`` at intermediate scales).

    :param planes1: ``(N, H, W)`` C-contiguous ``float32`` image planes.
    :param planes2: ``(N, H, W)`` planes aligned with ``planes1``.
    :param kernel: One-dimensional, symmetric window kernel of length ``k``.
    :param c1: Luminance stabilization constant ``(k1 * L) ** 2``.
    :param c2: Contrast stabilization constant ``(k2 * L) ** 2``.
    :return: A ``(sum_l_cs, sum_cs)`` tuple of ``(N,)`` ``float64`` arrays
        holding the map sums of ``luminance * contrast_structure`` and of
        ``contrast_structure``.
    :raises ValueError: If the plane batches differ in shape or a plane is
        smaller than the window.
    """
    cdef Py_ssize_t n_planes = planes1.shape[0]
    cdef Py_ssize_t height = planes1.shape[1]
    cdef Py_ssize_t width = planes1.shape[2]
    cdef Py_ssize_t taps = kernel.shape[0]
    if (
        planes2.shape[0] != n_planes
        or planes2.shape[1] != height
        or planes2.shape[2] != width
    ):
        raise ValueError(
            "planes1 and planes2 must have the same shape, got "
            f"{(n_planes, height, width)} vs "
            f"{(planes2.shape[0], planes2.shape[1], planes2.shape[2])}."
        )
    if taps < 1 or height < taps or width < taps:
        raise ValueError(
            f"Planes of size {height}x{width} are smaller than the "
            f"{taps}x{taps} window."
        )
    cdef Py_ssize_t out_h = height - taps + 1
    cdef Py_ssize_t out_w = width - taps + 1

    sum_l_cs_arr = np.empty(n_planes, dtype=np.float64)
    sum_cs_arr = np.empty(n_planes, dtype=np.float64)
    # One horizontally-filtered buffer per input row (5 moment streams), plus
    # per-thread scratch rows; threads write disjoint row slots, so no
    # reduction clauses are needed and results stay bit-deterministic.
    hsum_arr = np.empty((height, 5, out_w), dtype=np.float32)
    prod_arr = np.empty((num_threads, 3, width), dtype=np.float32)
    vrow_arr = np.empty((num_threads, 5, out_w), dtype=np.float32)
    row_l_cs_arr = np.empty(out_h, dtype=np.float64)
    row_cs_arr = np.empty(out_h, dtype=np.float64)

    cdef double[::1] sum_l_cs = sum_l_cs_arr
    cdef double[::1] sum_cs = sum_cs_arr
    cdef float[:, :, ::1] hsum = hsum_arr
    cdef float[:, :, ::1] prod = prod_arr
    cdef float[:, :, ::1] vrow = vrow_arr
    cdef double[::1] row_l_cs = row_l_cs_arr
    cdef double[::1] row_cs = row_cs_arr

    cdef Py_ssize_t n, h, ho, w, wo, j, t, r
    cdef float k0, kj, a, b
    cdef float mu1, mu2, e11, e22, e12, mu12, mu11, mu22, lum, cs
    cdef float two = 2.0
    cdef double acc_l_cs, acc_cs, total_l_cs, total_cs

    with nogil:
        for n in range(n_planes):
            # Pass A: horizontal window over the two planes and their three
            # pixel products, one output row per input row.
            for h in prange(height, num_threads=num_threads, schedule="static"):
                t = threadid()
                for w in range(width):
                    a = planes1[n, h, w]
                    b = planes2[n, h, w]
                    prod[t, 0, w] = a * a
                    prod[t, 1, w] = b * b
                    prod[t, 2, w] = a * b
                k0 = kernel[0]
                for wo in range(out_w):
                    hsum[h, 0, wo] = k0 * planes1[n, h, wo]
                    hsum[h, 1, wo] = k0 * planes2[n, h, wo]
                    hsum[h, 2, wo] = k0 * prod[t, 0, wo]
                    hsum[h, 3, wo] = k0 * prod[t, 1, wo]
                    hsum[h, 4, wo] = k0 * prod[t, 2, wo]
                for j in range(1, taps):
                    kj = kernel[j]
                    for wo in range(out_w):
                        hsum[h, 0, wo] += kj * planes1[n, h, wo + j]
                        hsum[h, 1, wo] += kj * planes2[n, h, wo + j]
                        hsum[h, 2, wo] += kj * prod[t, 0, wo + j]
                        hsum[h, 3, wo] += kj * prod[t, 1, wo + j]
                        hsum[h, 4, wo] += kj * prod[t, 2, wo + j]
            # Pass B: vertical window over the filtered rows, then the SSIM
            # arithmetic, reduced into one float64 slot per output row.
            for ho in prange(out_h, num_threads=num_threads, schedule="static"):
                t = threadid()
                k0 = kernel[0]
                for wo in range(out_w):
                    vrow[t, 0, wo] = k0 * hsum[ho, 0, wo]
                    vrow[t, 1, wo] = k0 * hsum[ho, 1, wo]
                    vrow[t, 2, wo] = k0 * hsum[ho, 2, wo]
                    vrow[t, 3, wo] = k0 * hsum[ho, 3, wo]
                    vrow[t, 4, wo] = k0 * hsum[ho, 4, wo]
                for j in range(1, taps):
                    kj = kernel[j]
                    for wo in range(out_w):
                        vrow[t, 0, wo] += kj * hsum[ho + j, 0, wo]
                        vrow[t, 1, wo] += kj * hsum[ho + j, 1, wo]
                        vrow[t, 2, wo] += kj * hsum[ho + j, 2, wo]
                        vrow[t, 3, wo] += kj * hsum[ho + j, 3, wo]
                        vrow[t, 4, wo] += kj * hsum[ho + j, 4, wo]
                acc_l_cs = 0.0
                acc_cs = 0.0
                for wo in range(out_w):
                    mu1 = vrow[t, 0, wo]
                    mu2 = vrow[t, 1, wo]
                    e11 = vrow[t, 2, wo]
                    e22 = vrow[t, 3, wo]
                    e12 = vrow[t, 4, wo]
                    mu12 = mu1 * mu2
                    mu11 = mu1 * mu1
                    mu22 = mu2 * mu2
                    lum = (two * mu12 + c1) / (mu11 + mu22 + c1)
                    cs = (two * (e12 - mu12) + c2) / (
                        (e11 - mu11) + (e22 - mu22) + c2
                    )
                    acc_cs = acc_cs + cs
                    acc_l_cs = acc_l_cs + lum * cs
                row_l_cs[ho] = acc_l_cs
                row_cs[ho] = acc_cs
            # Fixed-order sequential reduction: floating
            # point arithmetic is not associative, so this
            # is done to be able to deterministically compare
            # results to baseline methods
            total_l_cs = 0.0
            total_cs = 0.0
            for r in range(out_h):
                total_l_cs += row_l_cs[r]
                total_cs += row_cs[r]
            sum_l_cs[n] = total_l_cs
            sum_cs[n] = total_cs

    return sum_l_cs_arr, sum_cs_arr
