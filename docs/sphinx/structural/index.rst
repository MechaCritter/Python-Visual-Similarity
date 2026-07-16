Structural
==========

.. include:: ../../structural/README.md
   :parser: myst_parser.sphinx_
   :start-line: 1
   :end-before: <!-- benchmark:begin -->

Benchmark
---------

Accuracy and runtime of both metrics against their reference
implementations (scikit-image for SSIM, torchmetrics for MS-SSIM) are
tracked in `this file
<https://github.com/MechaCritter/Python-Visual-Similarity/blob/main/docs/structural/README.md#benchmark>`__.

API reference
-------------

.. autoclass:: pyvisim.structural.SSIM
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: pyvisim.structural.MSSSIM
   :members:
   :inherited-members:
   :show-inheritance:
