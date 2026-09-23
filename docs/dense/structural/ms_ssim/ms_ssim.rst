MSSSIM
======

``MS-SSIM`` goes one step further than :doc:`SSIM <../ssim/ssim>` by computing
the SSIM at multiple scales. Hence, images are first downsampled by half (up to
5 times) and the SSIM is computed at each scale, then aggregated. Given two
images x and y, ``MS-SSIM(x, y)`` is defined as:

.. math::

   \text{MS-SSIM}(x, y) = \left[l_M(x, y)\right]^{\alpha_M} \cdot \prod_{j=1}^{M} \left[c_j(x, y)\right]^{\beta_j} \left[s_j(x, y)\right]^{\gamma_j}

where:

- :math:`l_j`, :math:`c_j` and :math:`s_j` are the luminance, contrast and
  structure components at scale :math:`j`, as defined in the ``SSIM`` formula.
- :math:`M` is the coarsest scale, which is set by the number of entries in
  ``weights``.
- :math:`\alpha_M`, :math:`\beta_j` and :math:`\gamma_j` are the exponents that
  weight the contribution of each scale. In ``pyvisim``, the default weights,
  proposed by Wang et al. (2003), are used.

At the end of each scale, the images are low-pass filtered and downsampled by a
factor of two through 2x2 average pooling. The luminance component is evaluated
at the coarsest scale :math:`M` only, so the product above is computed from the
per-scale contrast-structure map means :math:`cs_j` and the full SSIM mean
:math:`\text{ssim}_M`:

.. math::

   \text{MS-SSIM}(x, y) = \text{ssim}_M^{\,w_M} \cdot \prod_{j=1}^{M-1} cs_j^{\,w_j}

Each entry of ``weights`` is one exponent :math:`w_j`, ordered with the coarsest
scale last, and the number of entries sets the number of scales. 

The Gaussian window has to fit the images after ``n_scales - 1`` halvings, so
each image side has to measure at least ``window_size * 2 ** (n_scales - 1)``
pixels.

Usage
-----

.. code-block:: python

   from pyvisim.dense.structural import MSSSIM

   msssim = MSSSIM(batch_size=16)
   matrix = msssim.similarity_score(gallery, queries)   # (N, M) matrix

.. include:: benchmark.md
   :parser: myst_parser.sphinx_

API reference
-------------

.. autoclass:: pyvisim.dense.structural.MSSSIM
   :members:
   :inherited-members:
   :show-inheritance:
