SSIM
====

``SSIM`` captures the perceptual similarity of two images. It is used, for
example, to test out the quality of image compression or denoising algorithms.

Given two images x and y, ``SSIM(x, y)`` is defined as:

.. math::

   \text{SSIM}(x, y) = \underbrace{\left[\frac{2\mu_x \mu_y + C_1}{\mu_x^2 + \mu_y^2 + C_1}\right]}_{\text{luminance}} \cdot \underbrace{\left[\frac{2\sigma_x \sigma_y + C_2}{\sigma_x^2 + \sigma_y^2 + C_2}\right]}_{\text{contrast}} \cdot \underbrace{\left[\frac{\sigma_{xy} + C_3}{\sigma_x \sigma_y + C_3}\right]}_{\text{structure}}

where:

- :math:`\mu_x` and :math:`\mu_y` are the local luminance of x and y.
- :math:`\sigma_x` and :math:`\sigma_y` are the local contrast of x and y.
- :math:`\sigma_{xy}` is the joint variation of x and y, which carries how far
  their local structures agree.
- :math:`C_1 = (k_1 L)^2` and :math:`C_2 = (k_2 L)^2` are stabilization
  constants that keep the fractions well-defined where their denominators come
  close to zero. Both :math:`k_1` and :math:`k_2` are exposed as parameters.
- ``window_size`` and ``sigma`` are the side length and the standard deviation
  of the Gaussian window that defines the neighborhood of a pixel.

With :math:`C_3 = C_2 / 2`, the three components collapse into one fraction:

.. math::

   \text{SSIM}(x, y) = \frac{\left(2\mu_x \mu_y + C_1\right)\left(2\sigma_{xy} + C_2\right)}{\left(\mu_x^2 + \mu_y^2 + C_1\right)\left(\sigma_x^2 + \sigma_y^2 + C_2\right)}

Here, :math:`L` is the dynamic range of the pixel values. In ``pyvisim``, it is fixed at
255 (every input is normalized to the ``[0, 255]`` range).

Identical images score 1, unrelated images score near 0, and inverted structures score
below 0.

Usage
-----

.. code-block:: python

   from pyvisim.dense.structural import SSIM

   ssim = SSIM()
   scores = ssim.similarity_score(image1, image2)   # (1, 1) matrix

.. include:: benchmark.md
   :parser: myst_parser.sphinx_

API reference
-------------

.. autoclass:: pyvisim.dense.structural.SSIM
   :members:
   :inherited-members:
   :show-inheritance:
