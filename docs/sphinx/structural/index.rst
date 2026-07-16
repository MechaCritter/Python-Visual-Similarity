Structural metrics
==================

.. include:: ../../structural/README.md
   :parser: myst_parser.sphinx_
   :start-line: 1
   :end-before: <!-- benchmark:begin -->

Benchmark
---------

Accuracy and runtime of both metrics against their reference
implementations (scikit-image for SSIM, torchmetrics for MS-SSIM) are
tracked in the auto-generated benchmark section of the
`structural docs page on GitHub
<https://github.com/MechaCritter/Python-Visual-Similarity/blob/main/docs/structural/README.md#benchmark>`__,
refreshed with ``python -m pyvisim.structural.generate_benchmark``.

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
